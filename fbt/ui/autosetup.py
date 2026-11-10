"""Auto-setup wizard: camera probe + PC benchmark + target app -> recommendations.

Heavy work runs in a background thread; results come back through Qt signals.
Nothing is applied until the user presses "Apply" on the summary page; rows
for settings the user changed by hand start unchecked.
"""
from __future__ import annotations

import copy
import logging
import threading
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QDialog, QHBoxLayout,
                               QHeaderView,
                               QLabel, QProgressBar, QPushButton, QRadioButton, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from .. import autotune
from ..camera import (best_from_profile, camera_label, find_device, is_video_file,
                      list_camera_devices, probe_camera)
from ..config import Config
from ..pose import ModelManifest, ensure_model
from ..presets import PRESETS, detect_target
from . import theme

log = logging.getLogger(__name__)


class _Bridge(QObject):
    progress = Signal(float, str)
    step = Signal(int, str)        # step index, state text
    instruction = Signal(str)
    done = Signal(dict)
    failed = Signal(str)
    cameras = Signal(object)       # list of CameraDevice or None


class _CameraError(RuntimeError):
    """The chosen camera can't be used — the user can go back and pick another."""


STEPS = ["Модели нейросети", "Возможности камеры", "Скорость вашего ПК", "Подбор настроек"]


class AutoSetupDialog(QDialog):
    def __init__(self, cfg: Config, models_dir, search_dirs, manifest: ModelManifest,
                 first_run: bool = False, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.models_dir, self.search_dirs, self.manifest = models_dir, search_dirs, manifest
        self.setWindowTitle("Автонастройка — Nexovara Pose")
        self.resize(820, 600)
        self.result_values: Dict[str, Any] = {}
        self.result_extra: Dict[str, Any] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._bridge = _Bridge()
        self._bridge.progress.connect(self._on_progress)
        self._bridge.step.connect(self._on_step)
        self._bridge.instruction.connect(lambda t: self.instr.setText(t))
        self._bridge.done.connect(self._on_done)
        self._bridge.failed.connect(self._on_failed)
        self._bridge.cameras.connect(self._on_cameras)
        self._step_note: Dict[int, str] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(22, 18, 22, 16)
        self.stack = QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.stack.addWidget(self._page_intro(first_run))
        self.stack.addWidget(self._page_run())
        self.stack.addWidget(self._page_result())

    # ---------------------------------------------------------------- pages
    def _page_intro(self, first_run: bool) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        t = QLabel("Добро пожаловать в Nexovara Pose!" if first_run else "Автонастройка")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        intro = QLabel(
            "Программа сама подберёт настройки под вашу камеру, компьютер и игру:<br>"
            "1. проверит все режимы камеры и измерит реальный FPS;<br>"
            "2. попросит вас встать в кадр и измерит скорость нейросети на этом ПК;<br>"
            "3. предложит настройки — вы увидите, что и почему меняется, и сами решите, "
            "что применить.<br><br>Займёт около минуты. Камера должна быть свободна "
            "(закройте Discord/OBS/Zoom, если они её используют).")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        lay.addWidget(intro)

        lay.addWidget(self._section("Камера"))
        row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setMinimumWidth(360)
        row.addWidget(self.cam_combo, 1)
        self.cam_refresh = QPushButton("Обновить список")
        self.cam_refresh.clicked.connect(self._scan_cameras)
        row.addWidget(self.cam_refresh)
        lay.addLayout(row)
        self.cam_note = QLabel("")
        self.cam_note.setObjectName("muted")
        self.cam_note.setWordWrap(True)
        lay.addWidget(self.cam_note)

        lay.addWidget(self._section("Где будете использовать трекинг?"))
        self.target_group = QButtonGroup(self)
        detected, reason = detect_target()
        current = self.cfg.target if self.cfg.target != "auto" else detected
        for i, p in enumerate(PRESETS.values()):
            rb = QRadioButton(p.title + ("  (обнаружено)" if p.id == detected else ""))
            rb.setToolTip(p.description)
            rb.setProperty("pid", p.id)
            rb.setChecked(p.id == current)
            self.target_group.addButton(rb, i)
            lay.addWidget(rb)
        note = QLabel(f"Сейчас: {reason}.")
        note.setObjectName("muted")
        lay.addWidget(note)

        lay.addWidget(self._section("Дополнительно"))
        self.fingers_cb = QCheckBox("Отслеживать пальцы (нагружает процессор; выключено по "
                                    "умолчанию)")
        self.fingers_cb.setChecked(self.cfg.fingers)
        lay.addWidget(self.fingers_cb)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        skip = QPushButton("Пропустить")
        skip.clicked.connect(self.reject)
        go = QPushButton("Начать")
        go.setObjectName("primary")
        go.clicked.connect(self._start)
        self.go_btn = go
        row.addWidget(skip)
        row.addWidget(go)
        lay.addLayout(row)
        self._scan_cameras()
        return w

    # ---------------------------------------------------------------- cameras
    def _scan_cameras(self) -> None:
        self.cam_combo.clear()
        self.cam_combo.addItem("Поиск камер…", None)
        self.cam_combo.setEnabled(False)
        self.cam_refresh.setEnabled(False)
        self.go_btn.setEnabled(False)
        self.cam_note.setText("")
        threading.Thread(target=lambda: self._bridge.cameras.emit(
            self._list_cameras()), daemon=True, name="camera-scan").start()

    @staticmethod
    def _list_cameras():
        try:
            return list_camera_devices(allow_open=True)
        except Exception:
            log.exception("camera scan failed")
            return None

    def _on_cameras(self, devs) -> None:
        cfg = self.cfg
        self.cam_combo.clear()
        self.cam_combo.setEnabled(True)
        self.cam_refresh.setEnabled(True)
        devs = list(devs or [])
        for d in devs:
            self.cam_combo.addItem(d.label(), (str(d.index), d.name))
        note = ""
        sel = -1
        if is_video_file(cfg.camera):
            self.cam_combo.addItem(camera_label(cfg.camera), (cfg.camera, ""))
            sel = self.cam_combo.count() - 1
        else:
            dev, st = find_device(devs, cfg.camera, cfg.camera_name)
            if dev is not None:
                sel = devs.index(dev)
                if st == "moved":
                    note = f"Камера «{dev.name}» теперь под номером {dev.index}."
            elif devs:
                sel = 0
                note = (f"«{camera_label(cfg.camera, cfg.camera_name)}» из настроек не "
                        f"найдена — выбрана «{devs[0].label()}».")
        if sel >= 0:
            self.cam_combo.setCurrentIndex(sel)
            self.go_btn.setEnabled(True)
        else:
            self.cam_combo.addItem("Камеры не найдены", None)
            note = ("Камера не найдена. Подключите веб-камеру (или телефон через DroidCam / "
                    "Iriun / Camo) и нажмите «Обновить список».")
            self.go_btn.setEnabled(False)
        if len(devs) > 1 and not note:
            note = f"Найдено камер: {len(devs)}. Выберите ту, что будет снимать вас целиком."
        self.cam_note.setText(note)

    @staticmethod
    def _section(text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("sectionTitle")
        return lab

    def _page_run(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        t = QLabel("Идёт автонастройка…")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        self.step_labels: List[QLabel] = []
        for s in STEPS:
            lab = QLabel(f"○  {s}")
            lab.setObjectName("muted")
            self.step_labels.append(lab)
            lay.addWidget(lab)
        self.instr = QLabel("")
        self.instr.setObjectName("hint")
        self.instr.setWordWrap(True)
        lay.addWidget(self.instr)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        lay.addWidget(self.bar)
        self.bar_text = QLabel("")
        self.bar_text.setObjectName("muted")
        lay.addWidget(self.bar_text)
        lay.addStretch(1)
        row = QHBoxLayout()
        row.addStretch(1)
        self.back_btn = QPushButton("← Назад (выбрать другую камеру)")
        self.back_btn.setVisible(False)
        self.back_btn.clicked.connect(self._back)
        row.addWidget(self.back_btn)
        cancel = QPushButton("Отмена")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        lay.addLayout(row)
        return w

    def _back(self) -> None:
        self.back_btn.setVisible(False)
        self.instr.setText("")
        self.instr.setStyleSheet("")
        self.bar.setValue(0)
        self.bar_text.setText("")
        self._step_note = {}
        self._cur_step = None
        for i in range(len(STEPS)):
            self._on_step(i, "")
        self.stack.setCurrentIndex(0)
        self._scan_cameras()

    def _page_result(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        t = QLabel("Рекомендуемые настройки")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        self.sys_label = QLabel()
        self.sys_label.setObjectName("muted")
        self.sys_label.setWordWrap(True)
        lay.addWidget(self.sys_label)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Применить / настройка", "Сейчас", "Рекомендуем",
                                              "Почему"])
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.res_note = QLabel()
        self.res_note.setObjectName("muted")
        self.res_note.setWordWrap(True)
        lay.addWidget(self.res_note)
        row = QHBoxLayout()
        row.addStretch(1)
        cancel = QPushButton("Не применять")
        cancel.clicked.connect(self.reject)
        ok = QPushButton("Применить выбранное")
        ok.setObjectName("primary")
        ok.clicked.connect(self._apply)
        row.addWidget(cancel)
        row.addWidget(ok)
        lay.addLayout(row)
        return w

    # ---------------------------------------------------------------- run
    def _start(self) -> None:
        btn = self.target_group.checkedButton()
        self._target = btn.property("pid") if btn else "vrchat_osc"
        self._fingers = self.fingers_cb.isChecked()
        data = self.cam_combo.currentData()
        if not data:
            return
        self._cam_src, self._cam_name = data
        self._cam_label = camera_label(self._cam_src, self._cam_name)
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self.stack.setCurrentIndex(1)
        self._thread = threading.Thread(target=self._work, daemon=True, name="autosetup")
        self._thread.start()

    def _work(self) -> None:
        b = self._bridge
        try:
            cfg = copy.deepcopy(self.cfg)
            cfg.fingers = self._fingers
            cfg.camera, cfg.camera_name = self._cam_src, self._cam_name
            label = self._cam_label
            sysinfo = autotune.system_info()

            # 1. models
            b.step.emit(0, "run")
            paths = {}
            wanted = ["lite", "full", "heavy"] + (["hand"] if self._fingers else [])
            for i, v in enumerate(wanted):
                if self._stop.is_set():
                    return
                b.progress.emit(i / len(wanted), f"Модель {v}…")
                try:
                    paths[v] = ensure_model(v, self.models_dir, self.search_dirs,
                                            lambda p, i=i: b.progress.emit((i + p) / len(wanted),
                                                                           f"Скачивание {v}…"),
                                            self.manifest)
                except Exception as e:
                    log.warning("model %s unavailable: %s", v, e)
            if not any(k in paths for k in ("lite", "full", "heavy")):
                raise RuntimeError("Не удалось получить ни одну модель нейросети "
                                   "(нет интернета?). Положите файлы в папку models.")
            b.step.emit(0, "ok")

            # 2. camera
            self._step_note[1] = f" — {label}"
            b.step.emit(1, "run")
            profile = None
            if not is_video_file(cfg.camera):
                b.instruction.emit(f"Проверяю режимы камеры «{label}»… Можно пока встать в "
                                   "кадр.")
                try:
                    profile = probe_camera(int(cfg.camera),
                                           lambda p, m: b.progress.emit(p, f"{label}: {m}"),
                                           self._stop.is_set)
                except Exception as e:
                    log.warning("camera probe failed: %s", e)
                    raise _CameraError(str(e)) from e
                best = best_from_profile(profile, prefer_hd=self._fingers)
                if best is not None:
                    cfg.width, cfg.height = best.width, best.height
                    cfg.fps = int(round(min(best.fps, 120)))
                    cfg.fourcc = best.fourcc if best.fourcc in ("MJPG", "YUY2") else "auto"
                b.step.emit(1, "ok")
            else:
                b.step.emit(1, "skip")
            if self._stop.is_set():
                return

            # 3. benchmark (user in view)
            b.step.emit(2, "run")
            for s in range(5, 0, -1):
                b.instruction.emit(f"Встаньте в кадр камеры «{label}» целиком (2–3 м от камеры)"
                                   f" — замер начнётся через {s}…")
                b.progress.emit(0.0, "")
                if self._stop.wait(1.0):
                    return
            b.instruction.emit("Идёт замер. Стойте в кадре, можно двигаться.")
            try:
                bench = autotune.benchmark(
                    cfg, {k: v for k, v in paths.items() if k != "hand"}, paths.get("hand"),
                    progress=lambda p, m: b.progress.emit(p, m), stop=self._stop.is_set)
            except Exception as e:
                if "камер" in str(e).lower():
                    raise _CameraError(str(e)) from e
                raise
            b.step.emit(2, "ok")

            # 4. recommendations
            b.step.emit(3, "run")
            recs = autotune.recommend(self.cfg, sysinfo, profile, bench, self._target,
                                      self._fingers)
            from ..model_catalog import AUTO_ORDER
            from ..onnx_pose import find_extra
            from ..rtmpose import onnxruntime_available
            if (self.cfg.ensemble_mode == "legacy" and self.cfg.refiner == "off"
                    and onnxruntime_available()
                    and any(find_extra(m, [*self.search_dirs, self.models_dir]) for m in AUTO_ORDER)):
                recs.append(autotune.Recommendation(
                    "ensemble_mode", "auto", "Ансамбль моделей",
                    "Использовать скачанные модели, если замеры укладываются в бюджет кадра."))
            b.step.emit(3, "ok")
            b.done.emit({"sysinfo": sysinfo, "profile": profile, "bench": bench, "recs": recs,
                         "camera": cfg.camera, "camera_name": cfg.camera_name})
        except _CameraError as e:
            b.failed.emit("camera:" + str(e))
        except Exception as e:
            log.exception("auto-setup failed")
            b.failed.emit(str(e) or e.__class__.__name__)

    def _on_progress(self, p: float, text: str) -> None:
        self.bar.setValue(int(max(0.0, min(1.0, p)) * 1000))
        self.bar_text.setText(text)

    def _on_step(self, i: int, state: str) -> None:
        if state == "run":
            self._cur_step = i
        icon = {"run": "◐", "ok": "✔", "skip": "–", "fail": "✖"}.get(state, "○")
        col = {"run": theme.ACCENT, "ok": theme.GOOD, "fail": theme.BAD}.get(state, theme.MUTED)
        lab = self.step_labels[i]
        lab.setText(f"<span style='color:{col}'>{icon}</span>  {STEPS[i]}"
                    + self._step_note.get(i, "")
                    + ("  (пропущено: видеофайл)" if state == "skip" else ""))

    def _on_failed(self, msg: str) -> None:
        camera = msg.startswith("camera:")
        if camera:
            msg = msg[len("camera:"):]
        self.instr.setText(f"Автонастройка не удалась: {msg}")
        self.instr.setStyleSheet(f"color: {theme.BAD};")
        if getattr(self, "_cur_step", None) is not None:
            self._on_step(self._cur_step, "fail")
        self.back_btn.setVisible(True)
        if not camera:
            self.back_btn.setText("← Назад")

    def _on_done(self, res: Dict[str, Any]) -> None:
        self._res = res
        si = res["sysinfo"]
        bench = res["bench"] or {}
        fingers = getattr(self, "_fingers", self.cfg.fingers)
        best = best_from_profile(res["profile"], fingers) if res.get("profile") else None
        cam = res.get("camera", self.cfg.camera)
        cam_name = res.get("camera_name", self.cfg.camera_name)
        lines = [f"Камера: {camera_label(cam, cam_name)}"
                 + (f" · лучший режим {best.label()}" if best else ""),
                 f"Процессор: {si['cpu']} · {si['threads']} потоков"
                 + (f" · ОЗУ {si['ram_gb']} ГБ" if si.get("ram_gb") else "")]
        if bench.get("results"):
            lines.append("Замер: " + ", ".join(
                f"{r['variant']}{' GPU' if r['delegate'] == 'GPU' else ''} {r['ms']:.0f} мс"
                f" (тело найдено в {r['detect_rate'] * 100:.0f}% кадров)"
                for r in bench["results"]))
        self.sys_label.setText("<br>".join(lines))
        recs: List[autotune.Recommendation] = res["recs"]
        self.table.setRowCount(0)
        self._rows = []
        same = 0
        for r in recs:
            cur = getattr(self.cfg, r.key, None)
            if cur == r.value:
                same += 1
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            user = r.key in self.cfg.user_set
            it = QTableWidgetItem(r.label + ("  ● ваше" if user else ""))
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Unchecked if user else Qt.Checked)
            if user:
                it.setToolTip("Вы меняли это вручную — по умолчанию не трогаем")
            self.table.setItem(row, 0, it)
            self.table.setItem(row, 1, QTableWidgetItem(self._fmt(r.key, cur)))
            self.table.setItem(row, 2, QTableWidgetItem(self._fmt(r.key, r.value)))
            why = QTableWidgetItem(r.reason)
            why.setToolTip(r.reason)
            self.table.setItem(row, 3, why)
            self._rows.append(r)
        self.table.resizeRowsToContents()
        self.res_note.setText(
            (f"Ещё {same} настроек уже выставлены оптимально. " if same else "")
            + ("Отмеченные настройки будут применены; снятые галочки — останутся как есть."
               if self._rows else "Менять ничего не нужно."))
        self.stack.setCurrentIndex(2)

    @staticmethod
    def _fmt(key: str, v) -> str:
        if key == "ensemble_mode":
            return {"legacy": "Одна модель / RTMPose", "auto": "Авто", "manual": "Вручную"}.get(v, str(v))
        if key == "target":
            p = PRESETS.get(v)
            return p.title if p else ("Авто" if v == "auto" else str(v))
        if key == "trackers" and isinstance(v, dict):
            from ..config import TRACKER_LABELS_RU
            return ", ".join(TRACKER_LABELS_RU[k] for k, on in v.items() if on) or "—"
        if key == "finger_output":
            return {"preview": "только показ",
                    "driver": "драйвер Nexovara Pose"}.get(v, str(v))
        if key == "fourcc":
            return "авто" if v == "auto" else str(v)
        return autotune.format_value(v)

    def _apply(self) -> None:
        vals = {}
        for row, r in enumerate(self._rows):
            if self.table.item(row, 0).checkState() == Qt.Checked:
                vals[r.key] = r.value
        self.result_values = vals
        res = self._res
        self.result_extra = {
            "camera": res.get("camera", self.cfg.camera),
            "camera_name": res.get("camera_name", self.cfg.camera_name),
            "camera_profile": res["profile"],
            "pc_profile": autotune.pc_profile_from(res["sysinfo"], res["bench"]),
        }
        self.accept()

    def done(self, r: int) -> None:  # also covers reject()/close
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        super().done(r)

