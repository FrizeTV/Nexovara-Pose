"""Main window."""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox, QPushButton, QSplitter,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, profiles, steamvr
from ..camera import (SETTLE_S, camera_label, list_camera_devices, notify_device_change,
                      short_label)
from ..config import Config, apply_auto, app_dir, remap_camera_sources, save_config
from ..engine import Engine
from ..pose import ModelManifest
from ..recording import is_recording
from ..presets import PRESETS
from . import theme
from .autosetup import AutoSetupDialog
from .model_manager import ModelDownloadDialog
from .settings_dialog import SettingsDialog
from ..guidance import LABELS_RU as CONF_LABELS
from .widgets import ConfidenceBars, FingerBars, Skeleton3DView, StatCard, VideoView

log = logging.getLogger(__name__)

# changing these needs a restart of the tracking thread (done automatically);
# "target" re-applies the game preset on start
RESTART_KEYS = {"camera", "camera_name", "width", "height", "fps", "fourcc", "model",
                "use_gpu", "refiner", "refiner_gpu", "ensemble_mode", "ensemble_models",
                "min_detection", "min_presence", "min_tracking", "target",
                "person_lock", "max_people", "extra_cameras"}


class _Signals(QObject):
    cameras = Signal(object)   # List[CameraDevice] or None (names unknown, cameras busy)
    vrchat_desktop = Signal(bool)
    tuned = Signal(object)     # recording.TuneResult (or an error string)
    tune_progress = Signal(float)
    replug = Signal(object)    # camera names after a device change (or None)


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config, models_dir: Path, model_search_dirs: List[Path]):
        super().__init__()
        self.cfg = cfg
        self.models_dir = models_dir
        self.model_search_dirs = model_search_dirs
        self.data_dir = app_dir()
        self.manifest = ModelManifest(self.data_dir / "models.json")
        self.engine: Optional[Engine] = None
        self.settings: Optional[SettingsDialog] = None
        self._last_seq = -1
        self._countdown = 0
        self._status = "Готово к запуску"
        self._target_note = ""
        self._sig = _Signals()
        self._sig.cameras.connect(self._cameras_found)
        self._sig.vrchat_desktop.connect(self._set_vrchat_desktop)
        self._sig.tuned.connect(self._tune_done)
        self._sig.replug.connect(self._replug_checked)
        self._sig.tune_progress.connect(
            lambda p: setattr(self, "_status", f"Подбор сглаживания по записи… {p * 100:.0f}%"))
        self._tuning = False
        self._vrchat_desktop = False
        self._env_busy = False
        self._scan_quiet = False
        self._run_cam = (cfg.camera, cfg.camera_name)
        self._recording = False
        self._camera_before_replay: Optional[tuple] = None

        self.setWindowTitle(f"{APP_NAME} {__version__} — трекинг тела с веб-камеры")
        self.resize(1360, 820)
        self.setMinimumSize(1000, 640)
        self._build()

        self._save_timer = QTimer(self, singleShot=True, interval=400)
        self._save_timer.timeout.connect(self._save_and_push)
        self._restart_timer = QTimer(self, singleShot=True, interval=900)
        self._restart_timer.timeout.connect(self._restart_engine)
        # what else is running (VRChat without VR?) — checked in the background
        self._env_timer = QTimer(self, interval=5000)
        self._env_timer.timeout.connect(self._check_environment)
        self._env_timer.start()
        self._tick = QTimer(self, interval=16)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()
        # a camera plugged in / out: leave the cameras alone until Windows is done
        from PySide6.QtWidgets import QApplication
        from . import devwatch
        self._dev_rescan = QTimer(self, singleShot=True, interval=int(SETTLE_S * 1000) + 500)
        self._dev_rescan.timeout.connect(self._devices_settled)
        try:
            self._dev_filter = devwatch.install(QApplication.instance(), self._device_changed)
        except Exception:
            log.exception("device watcher unavailable")
            self._dev_filter = None
        for key, fn in (("F5", self.toggle_run), ("F6", self.start_calibration),
                        ("F7", self.toggle_pause), ("Ctrl+,", self.open_settings)):
            QShortcut(QKeySequence(key), self, activated=fn)
        for i in range(1, 10):
            QShortcut(QKeySequence(f"Ctrl+{i}"), self,
                      activated=lambda i=i: self._profile_hotkey(i))
        self._refresh_header()
        if not cfg.first_run_done:
            QTimer.singleShot(400, self.open_first_run)

    def open_models(self) -> None:
        ModelDownloadDialog(self.models_dir, self.model_search_dirs, self.manifest, self).exec()

    def open_first_run(self) -> None:
        self.open_models()
        self.open_autosetup(first_run=True)

    # ------------------------------------------------------------------ UI
    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QFrame()
        header.setObjectName("header")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 10, 16, 10)
        title = QLabel("Nexovara Pose")
        title.setObjectName("appTitle")
        hl.addWidget(title)
        self.chip = QLabel()
        self.chip.setObjectName("chip")
        hl.addWidget(self.chip)
        hl.addSpacing(8)
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumContentsLength(12)
        self.profile_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.profile_combo.setToolTip("Профиль: игра / аватар / человек (рост, калибровка, "
                                      "трекеры, сглаживание).\nCtrl+1…Ctrl+9 — быстро "
                                      "переключить. Создать — Настройки → Профили.\n"
                                      "* — текущие настройки отличаются от профиля.")
        self.profile_combo.activated.connect(self._profile_chosen)
        hl.addWidget(self.profile_combo)
        hl.addSpacing(8)
        self.btn_run = QPushButton("▶  Старт")
        self.btn_run.setObjectName("primary")
        self.btn_run.setToolTip("Запустить / остановить трекинг (F5)")
        self.btn_run.clicked.connect(self.toggle_run)
        self.btn_cal = QPushButton("🎯  Калибровка")
        self.btn_cal.setToolTip("Встаньте ровно лицом к камере (F6)")
        self.btn_cal.clicked.connect(self.start_calibration)
        self.btn_pause = QPushButton("⏸  Пауза")
        self.btn_pause.setToolTip("Заморозить трекеры в игре (F7)")
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_rec = QPushButton("⏺  Запись")
        self.btn_rec.setToolTip("Записать ваши движения — только цифры позы, без видео — "
                                "чтобы подобрать сглаживание под вас или проверить настройки "
                                "без камеры (Настройки → Запись и повтор)")
        self.btn_rec.clicked.connect(self.toggle_recording)
        for b in (self.btn_run, self.btn_cal, self.btn_pause, self.btn_rec):
            hl.addWidget(b)
        hl.addStretch(1)
        b_auto = QPushButton("✨  Автонастройка")
        b_auto.setToolTip("Подобрать настройки под камеру, ПК и игру")
        b_auto.clicked.connect(lambda: self.open_autosetup(False))
        b_set = QPushButton("⚙  Настройки")
        b_set.setToolTip("Все настройки в отдельном окне (Ctrl+,)")
        b_set.clicked.connect(self.open_settings)
        hl.addWidget(b_auto)
        hl.addWidget(b_set)
        root.addWidget(header)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(14, 12, 14, 12)
        bl.setSpacing(10)
        self.hint = QLabel()
        self.hint.setObjectName("hint")
        self.hint.setWordWrap(True)
        bl.addWidget(self.hint)

        split = QSplitter(Qt.Horizontal)
        self.video = VideoView()
        self.video.clicked.connect(self._video_clicked)
        split.addWidget(self.video)
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(10)
        self.view3d = Skeleton3DView()
        rl.addWidget(self.view3d, 1)
        grid = QGridLayout()
        grid.setSpacing(8)
        self.c_cam = StatCard("Камера")
        self.c_nn = StatCard("Нейросеть")
        self.c_track = StatCard("Трекинг")
        self.c_out = StatCard("Вывод")
        self.c_cal = StatCard("Калибровка")
        self.c_lat = StatCard("Задержка")
        for i, c in enumerate((self.c_cam, self.c_nn, self.c_track, self.c_out,
                               self.c_cal, self.c_lat)):
            grid.addWidget(c, i // 2, i % 2)
        rl.addLayout(grid)
        self.c_conf = StatCard("Уверенность трекеров")
        self.c_conf.value.hide()
        self.c_conf.sub.hide()
        self.c_conf.setToolTip("Насколько хорошо камера видит то, из чего строится каждый "
                               "трекер.\nЗелёный — уверенно, жёлтый — так себе, красный — "
                               "трекер достраивается или удерживается.")
        self.conf_bars = ConfidenceBars()
        self.c_conf.layout().addWidget(self.conf_bars)
        rl.addWidget(self.c_conf)
        self.c_fingers = StatCard("Пальцы (сгиб)")
        self.c_fingers.value.hide()
        self.finger_bars = FingerBars()
        self.c_fingers.layout().addWidget(self.finger_bars)
        rl.addWidget(self.c_fingers)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([800, 520])
        bl.addWidget(split, 1)
        root.addWidget(body, 1)
        self.statusBar().showMessage("")
        self._idle_cards()

    def _idle_cards(self) -> None:
        for c in (self.c_cam, self.c_nn, self.c_track, self.c_lat):
            c.set("—", "")
        self._set_cam_card_name()
        self._out_card_idle()
        cal = self.cfg.calibration
        self.c_cal.set("Есть" if cal else "Нужна", "F6 — откалибровать",
                       theme.GOOD if cal else theme.WARN)
        self.c_fingers.setVisible(self.cfg.fingers)
        self.conf_bars.set_values([(CONF_LABELS.get(n, n), None)
                                   for n in self.cfg.enabled_trackers()])

    def _set_cam_card_name(self) -> None:
        """Camera card title row: which camera is used (full name in the tooltip)."""
        full = camera_label(self.cfg.camera, self.cfg.camera_name)
        self.c_cam.sub.setText(short_label(self.cfg.camera, self.cfg.camera_name))
        self.c_cam.setToolTip(full)

    def _cam_label(self, source: str) -> str:
        """Name of any configured camera (main or extra) for messages."""
        if source == self.cfg.camera:
            return camera_label(source, self.cfg.camera_name)
        for c in self.cfg.extra_cameras:
            if c.get("source") == source:
                return camera_label(source, c.get("name", ""))
        return camera_label(source)

    def _out_card_idle(self) -> None:
        outs = self._outputs_text()
        self.c_out.set(outs or "нет", "выключено" if not outs else "ожидание")

    def _outputs_text(self) -> str:
        o = []
        if self.cfg.output_vrchat:
            o.append("VRChat OSC")
        if self.cfg.output_driver:
            o.append("SteamVR (драйвер" + (", без шлема)" if self.cfg.driver_hmd else ")"))
        return " + ".join(o)

    def _refresh_header(self) -> None:
        if self.cfg.target == "auto":
            txt = "Режим: авто" + (f" → {self._target_note}" if self._target_note else "")
        else:
            p = PRESETS.get(self.cfg.target)
            txt = f"Режим: {p.title if p else self.cfg.target}"
        self.chip.setText(txt)
        self._refresh_profiles()

    # ------------------------------------------------------------ devices
    def _device_changed(self, _kind: int) -> None:
        """Called from the native event filter (GUI thread) on WM_DEVICECHANGE."""
        notify_device_change()
        if not self._dev_rescan.isActive():
            log.info("device change: cameras are left alone for %.0f s", SETTLE_S)
        self._dev_rescan.start()            # restarts: waits for the LAST change

    def _devices_settled(self) -> None:
        if self.settings is not None and self.settings.isVisible():
            self._scan_cameras(quiet=True)   # show a newly plugged camera in the lists
        # running on a stand-in camera because YOUR camera was missing at start?
        name = self.cfg.camera_name
        if (self.engine and self.engine.running and name
                and getattr(self, "_run_cam", (None, name))[1] != name):
            def work():
                try:
                    self._sig.replug.emit(list_camera_devices(allow_open=False))
                except Exception:
                    self._sig.replug.emit(None)
            threading.Thread(target=work, daemon=True, name="camera-replug").start()

    def _replug_checked(self, devs) -> None:
        name = self.cfg.camera_name
        if not devs or not (self.engine and self.engine.running):
            return
        if any(d.name == name for d in devs) and self._run_cam[1] != name:
            self._status = f"Подключена ваша камера «{name}» — переключаюсь на неё"
            log.info("remembered camera %s is back: restarting tracking", name)
            self._restart_engine()

    # ------------------------------------------------------------ profiles
    def _refresh_profiles(self) -> None:
        c = self.profile_combo
        c.blockSignals(True)
        c.clear()
        c.addItem("Профиль: —", "")
        for i, name in enumerate(profiles.names(self.cfg)):
            mark = " *" if name == self.cfg.active_profile and profiles.differs(self.cfg) else ""
            c.addItem(f"Профиль: {name}{mark}", name)
            if i < 9:
                c.setItemData(c.count() - 1, f"Ctrl+{i + 1}", Qt.ToolTipRole)
        i = c.findData(self.cfg.active_profile)
        c.setCurrentIndex(i if i >= 0 else 0)
        c.setVisible(bool(self.cfg.profiles))
        c.blockSignals(False)

    def _profile_chosen(self, _i: int) -> None:
        name = self.profile_combo.currentData()
        if name:
            self.switch_profile(name)
        else:
            self.cfg.active_profile = ""
            self._after_change()

    def _profile_hotkey(self, i: int) -> None:
        names = profiles.names(self.cfg)
        if 0 < i <= len(names):
            self.switch_profile(names[i - 1])

    def switch_profile(self, name: str) -> None:
        try:
            changed = profiles.apply_profile(self.cfg, name)
        except KeyError:
            return
        self._status = f"Профиль «{name}»" + ("" if changed else " (настройки те же)")
        if self.settings:
            self.settings.refresh(self._settings_extra())
        self._after_change(restart=bool(profiles.RESTART_KEYS.intersection(changed)))

    # -------------------------------------------------------------- actions
    def toggle_run(self) -> None:
        if self.engine and self.engine.running:
            self._stop_engine()
            return
        self._start_engine()

    def _start_engine(self) -> None:
        self._save_timer.stop()
        save_config(self.cfg)
        self._run_cam = (self.cfg.camera, self.cfg.camera_name)
        self.engine = Engine(self.cfg, self.models_dir, self.model_search_dirs, self.data_dir)
        self.engine.start()
        self.btn_run.setText("■  Стоп")
        self._status = "Запуск…"

    def _stop_engine(self) -> None:
        if self.engine:
            self.engine.stop(timeout=3.0)
            self._drain_events()
        self.engine = None
        self._on_stopped()

    def _restart_engine(self) -> None:
        if self.engine and self.engine.running:
            self._status = "Применяю новые настройки камеры/модели…"
            self._stop_engine()
            self._start_engine()

    def _on_stopped(self) -> None:
        self._recording = False
        self.btn_rec.setText("⏺  Запись")
        self.btn_run.setText("▶  Старт")
        self.btn_pause.setText("⏸  Пауза")
        self._countdown = 0
        self._last_seq = -1
        self.video.overlay_text = ""
        self.video.clear("Трекинг остановлен\nНажмите «Старт» (F5)")
        self.view3d.set_data(None, [], None, message="")
        self._idle_cards()

    def _video_clicked(self, x: float, y: float) -> None:
        if self.engine and self.engine.running and self.cfg.person_lock:
            self.engine.pick_person(x, y)
            self._status = "Слежу за выбранным человеком"

    def toggle_pause(self) -> None:
        if not self.engine:
            return
        self.engine.paused = not self.engine.paused
        self.btn_pause.setText("▶  Продолжить" if self.engine.paused else "⏸  Пауза")

    # ------------------------------------------------------------ recording
    def toggle_recording(self) -> None:
        if not (self.engine and self.engine.running):
            QMessageBox.information(self, APP_NAME, "Сначала запустите трекинг (F5).")
            return
        if self._recording:
            self.engine.stop_recording()
        else:
            self.engine.start_recording()
            self._status = ("Запись: двигайтесь как обычно в игре — постойте спокойно, "
                            "походите, присядьте. Нажмите «Стоп записи», когда хватит "
                            "(1–3 минуты).")

    def _recording_event(self, p: dict) -> None:
        st = p.get("state")
        if st == "started":
            self._recording = True
            self.btn_rec.setText("⏹  Стоп записи")
            return
        self._recording = False
        self.btn_rec.setText("⏺  Запись")
        if st == "saved":
            self._status = (f"Запись сохранена ({p.get('seconds', 0):.0f} с): "
                            f"{Path(p['path']).name}"
                            + (" — достигнут предел 15 минут" if p.get("full") else "")
                            + ". Подбор сглаживания: Настройки → Запись и повтор.")
            if self.settings:
                self.settings.refresh(self._settings_extra())
        elif st == "empty":
            self._status = "Запись пустая — вас не было видно в кадре"
        else:
            self._status = f"Запись не сохранилась: {p.get('error', '')}"

    def _own_recording(self, path: str) -> Optional[Path]:
        """Only files inside <data>/recordings are touched."""
        try:
            p = Path(path).resolve()
            root = (self.data_dir / "recordings").resolve()
        except OSError:
            return None
        return p if p.parent == root and p.suffix.lower() == ".npz" and p.is_file() else None

    def _recording_action(self, act: str) -> None:
        kind, _, arg = act.partition(":")
        if kind == "replay_stop":
            src, name = self._camera_before_replay or ("0", "")
            self._camera_before_replay = None
            self.cfg.camera, self.cfg.camera_name = src, name
            self._status = "Снова камера: " + camera_label(src, name)
            self._after_change(restart=True)
            return
        p = self._own_recording(arg)
        if p is None:
            QMessageBox.warning(self, APP_NAME, "Запись не найдена.")
            return
        if kind == "replay":
            if not is_recording(self.cfg.camera):
                self._camera_before_replay = (self.cfg.camera, self.cfg.camera_name)
            self.cfg.camera, self.cfg.camera_name = str(p), ""
            self._status = (f"Воспроизведение записи «{p.name}» — камера и нейросеть не "
                            "используются. «Вернуть камеру» — в Настройки → Запись и повтор.")
            if self.engine and self.engine.running:
                self._after_change(restart=True)
            else:
                save_config(self.cfg)
                self._start_engine()
        elif kind == "rec_delete":
            if QMessageBox.question(self, APP_NAME, f"Удалить запись «{p.name}»?") \
                    != QMessageBox.Yes:
                return
            if str(p) == self.cfg.camera:
                self._recording_action("replay_stop")
            try:
                p.unlink()
            except OSError as e:
                QMessageBox.warning(self, APP_NAME, f"Не получилось удалить: {e}")
        elif kind == "tune":
            if self._tuning:
                return
            self._tuning = True
            self._status = "Подбор сглаживания по записи…"
            cfg = self.cfg

            def work():
                from ..recording import load_recording, tune_filter
                try:
                    rec = load_recording(p)
                    r = tune_filter(rec, cfg.filter_min_cutoff, cfg.filter_beta, cfg.lag_comp,
                                    progress=self._sig.tune_progress.emit)
                except Exception as e:
                    log.exception("filter tuning failed")
                    r = f"Подбор не удался: {e}"
                self._sig.tuned.emit(r)
            threading.Thread(target=work, daemon=True, name="tune").start()

    def _tune_done(self, r) -> None:
        self._tuning = False
        if isinstance(r, str):
            self._status = r
            QMessageBox.warning(self, APP_NAME, r)
            return
        self._status = r.message
        if not (r.ok and r.better):
            QMessageBox.information(self, APP_NAME, r.message)
            return
        if QMessageBox.question(
                self, APP_NAME, r.message + f"\n\nМин. частота среза: "
                f"{self.cfg.filter_min_cutoff:.2f} → {r.min_cutoff:.2f}\nБета: "
                f"{self.cfg.filter_beta:.2f} → {r.beta:.2f}\n\nПрименить?") == QMessageBox.Yes:
            self.cfg.filter_min_cutoff, self.cfg.filter_beta = r.min_cutoff, r.beta
            self.cfg.mark_user_set(["filter_min_cutoff", "filter_beta"])
            self._after_change()
            self._status = "Сглаживание подобрано по вашей записи и применено"
            if self.settings:
                self.settings.refresh(self._settings_extra())

    def start_calibration(self) -> None:
        if not (self.engine and self.engine.running):
            QMessageBox.information(self, APP_NAME, "Сначала запустите трекинг (F5).")
            return
        self._countdown = 3
        self._countdown_step()

    def _countdown_step(self) -> None:
        if not self.engine:
            self._countdown = 0
            return
        if self._countdown > 0:
            self.video.overlay_text = str(self._countdown)
            self._status = "Калибровка: встаньте ровно лицом к камере, руки вниз"
            self._countdown -= 1
            QTimer.singleShot(1000, self._countdown_step)
        else:
            self.video.overlay_text = ""
            self.engine.request_calibration(30)
            self._status = "Калибровка: не двигайтесь…"

    def open_settings(self) -> None:
        if self.settings is None:
            self.settings = SettingsDialog(self.cfg, self)
            self.settings.changed.connect(self._settings_changed)
            self.settings.action.connect(self._settings_action)
            self._scan_cameras(quiet=True)  # names for the camera lists
        self.settings.refresh(self._settings_extra())
        self.settings.show()
        self.settings.raise_()
        self.settings.activateWindow()

    def _settings_extra(self) -> dict:
        if not steamvr.is_installed():
            sv = "SteamVR не найден на этом ПК."
        else:
            sv = "SteamVR установлен" + (", запущен." if steamvr.is_running() else ", не запущен.")
        if not steamvr.driver_built():
            drv = "Драйвер Nexovara Pose ещё не собран — запустите build_driver.bat."
        else:
            reg = steamvr.driver_registered()
            drv = "Драйвер Nexovara Pose собран" + (
                (", установлен в SteamVR" + ("" if steamvr.installed_copy_current() else
                                             " (устарел — нажмите «Установить / обновить»)"))
                if reg else ", НЕ установлен в SteamVR")
            drv += "; режим без шлема: " + ("включён" if steamvr.no_hmd_mode_enabled() else "выключен")
            drv += "."
        return {"steamvr": sv, "driver": drv, "data_dir": str(self.data_dir)}

    def open_autosetup(self, first_run: bool = False) -> None:
        was_running = bool(self.engine and self.engine.running)
        if was_running:
            self._stop_engine()  # the camera can only be used by one at a time
        dlg = AutoSetupDialog(self.cfg, self.models_dir, self.model_search_dirs, self.manifest,
                              first_run, self)
        accepted = dlg.exec() == AutoSetupDialog.Accepted
        self.cfg.first_run_done = True
        if accepted:
            extra = dlg.result_extra
            cam = extra.get("camera")
            if cam and (cam, extra.get("camera_name", "")) != (self.cfg.camera,
                                                               self.cfg.camera_name):
                self.cfg.camera, self.cfg.camera_name = cam, extra.get("camera_name", "")
                self.cfg.mark_user_set(["camera"])  # chosen in the wizard
                self.cfg.extra_cameras = [c for c in self.cfg.extra_cameras
                                          if c.get("source") != cam]
            if extra.get("camera_profile"):
                self.cfg.camera_profiles[self.cfg.camera] = extra["camera_profile"]
            if extra.get("pc_profile"):
                self.cfg.pc_profile = extra["pc_profile"]
            apply_auto(self.cfg, dlg.result_values, force=True)
            self._status = (f"Автонастройка применена ({len(dlg.result_values)} изменений)"
                            if dlg.result_values else "Автонастройка: менять ничего не нужно")
        save_config(self.cfg)
        self._refresh_header()
        self._idle_cards()
        if self.settings:
            self.settings.refresh(self._settings_extra())
        if was_running or (accepted and first_run):
            self._start_engine()

    # ------------------------------------------------------------ settings
    def _settings_changed(self, keys: list) -> None:
        self._refresh_header()   # also the profile list (* when changed)
        if self.settings and any(k.startswith("key_") for k in keys):
            self.settings.refresh(self._settings_extra())  # duplicates were corrected
        self.c_fingers.setVisible(self.cfg.fingers)
        self._save_timer.start()
        if self.engine and self.engine.running and RESTART_KEYS.intersection(keys):
            self._restart_timer.start()

    def _save_and_push(self) -> None:
        save_config(self.cfg)
        if self.engine and self.engine.running:
            self.engine.update_config(self.cfg)

    def _settings_action(self, act: str) -> None:
        if act == "models":
            self.open_models()
            return
        if act.startswith("profile_apply:"):
            self.switch_profile(act.split(":", 1)[1])
            return
        if act.startswith(("replay:", "tune:", "rec_delete:")) or act == "replay_stop":
            self._recording_action(act)
            if self.settings:
                self.settings.refresh(self._settings_extra())
            return
        if act == "reset_auto":
            if QMessageBox.question(self, APP_NAME, "Забыть все ручные изменения? При следующем "
                                    "запуске программа снова подберёт камеру и режим "
                                    "игры сама.") == QMessageBox.Yes:
                self.cfg.user_set = []
                self.cfg.target = "auto"
                self._after_change(restart=True)
        elif act == "reset_calibration":
            self.cfg.calibration = None
            self._after_change()
        elif act == "scan_cameras":
            self._scan_cameras()
        elif act == "pick_video":
            p, _ = QFileDialog.getOpenFileName(self, "Видеофайл для теста", "",
                                               "Видео (*.mp4 *.avi *.mkv *.mov *.webm *.m4v)")
            if p:
                self.cfg.camera = p
                self.cfg.camera_name = ""
                self.cfg.mark_user_set(["camera"])
                self._after_change(restart=True)
        elif act == "reprobe_camera":
            self.cfg.camera_profiles.pop(self.cfg.camera, None)
            self.cfg.user_set = [k for k in self.cfg.user_set
                                 if k not in ("width", "height", "fps", "fourcc")]
            self._status = "Режимы камеры будут измерены при запуске"
            self._after_change(restart=True)
        elif act == "relock_person":
            if self.engine and self.engine.running:
                self.engine.relock_person()
            self._status = "Выбираю человека заново — встаньте в центр кадра"
        elif act == "calibrate_rig":
            if not (self.engine and self.engine.running):
                QMessageBox.information(self, APP_NAME, "Сначала запустите трекинг (F5).")
            elif not self.cfg.extra_cameras:
                QMessageBox.information(self, APP_NAME, "Сначала добавьте дополнительную камеру.")
            else:
                self.engine.request_rig_calibration(8.0)
                self._status = ("Калибровка камер: встаньте так, чтобы вас видели все камеры, "
                                "медленно поворачивайтесь и поднимайте руки…")
        elif act in ("driver_install", "driver_uninstall", "driver_nohmd"):
            self._driver_action(act)
        elif act == "steamvr_check":
            self.open_steamvr_check()
        elif act == "vrchat_align":
            if not (self.engine and self.engine.running):
                QMessageBox.information(self, APP_NAME, "Сначала запустите трекинг (F5).")
            else:
                self.engine.request_vrchat_align()
                self._status = "VRChat: направление выровнено по вашему корпусу"
        elif act == "assign_roles":
            self._assign_roles()
        elif act == "refresh_steamvr":
            pass
        elif act == "open_data":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.data_dir)))
        elif act == "clear_logs":
            self._clear_logs()
        elif act == "delete_data":
            self._delete_data()
        if self.settings:
            self.settings.refresh(self._settings_extra())

    def _after_change(self, restart: bool = False) -> None:
        save_config(self.cfg)
        self._refresh_header()
        if not (self.engine and self.engine.running):
            self._idle_cards()
        else:
            if restart:
                self._restart_engine()
            else:
                self.engine.update_config(self.cfg)

    def _scan_cameras(self, quiet: bool = False) -> None:
        """Camera list with names. Names come from the OS without opening the
        cameras; without them the cameras are probed — not while tracking."""
        running = bool(self.engine and self.engine.running)
        if not quiet:
            self._status = "Поиск камер…"
        self._scan_quiet = quiet

        def work():
            try:
                devs = list_camera_devices(allow_open=not running)
            except Exception:
                log.exception("camera scan failed")
                devs = None
            self._sig.cameras.emit(devs)
        threading.Thread(target=work, daemon=True, name="camera-scan").start()

    def _cameras_found(self, devs) -> None:
        quiet = getattr(self, "_scan_quiet", False)
        if devs is None:
            if not quiet:
                self._status = ("Остановите трекинг перед поиском камер (названия камер в этой "
                                "системе недоступны, нужно открывать каждую камеру)")
            return
        if not quiet:
            self._status = ("Найдены камеры: " + ", ".join(d.label() for d in devs)
                            if devs else "Камеры не найдены")
        if self.settings:
            self.settings.set_cameras(devs)
        if not (self.engine and self.engine.running):
            self._idle_cards()

    def _assign_roles(self) -> None:
        enabled = self.cfg.enabled_trackers()
        if not enabled:
            QMessageBox.information(self, APP_NAME, "Не включено ни одного трекера.")
            return
        plan = steamvr.planned_roles(enabled)
        text = "\n".join(f"  {k}  →  {v.replace('TrackerRole_', '')}" for k, v in plan.items())
        if QMessageBox.question(
                self, APP_NAME,
                "Записать роли трекеров в настройки SteamVR?\n\n" + text +
                "\n\nПеред изменением будет создана резервная копия steamvr.vrsettings. "
                "SteamVR должен быть закрыт.") != QMessageBox.Yes:
            return
        try:
            backup = steamvr.assign_tracker_roles(enabled)
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не получилось: {e}")
            return
        QMessageBox.information(self, APP_NAME, "Готово! Роли записаны. Резервная копия:\n"
                                                f"{backup}")

    def open_steamvr_check(self) -> None:
        from .steamvr_check import SteamVRCheckDialog
        dlg = SteamVRCheckDialog(self.cfg, self._app_checks, self._after_steamvr_fix,
                                 self._assign_roles, self)
        dlg.exec()

    def _after_steamvr_fix(self) -> None:
        if self.cfg.target == "steamvr_nohmd" or self.cfg.driver_hmd:
            from ..presets import PRESETS
            self.cfg.target = "steamvr_nohmd"
            apply_auto(self.cfg, PRESETS["steamvr_nohmd"].values, force=True)
        save_config(self.cfg)
        self._refresh_header()
        if self.settings:
            self.settings.refresh(self._settings_extra())

    def _app_checks(self) -> list:
        """Nexovara Pose's own part of the chain, for the SteamVR check dialog."""
        C = steamvr.Check
        out = []
        p = PRESETS.get(self.cfg.target)
        mode = p.title if p else "Авто"
        if not self.cfg.output_driver:
            out.append(C("Nexovara Pose отправляет в драйвер", False,
                         f"режим «{mode}» — вывод в драйвер выключен. Выберите «SteamVR без "
                         "шлема» или «SteamVR — драйвер Nexovara Pose» (Настройки → Игра и вывод)"))
        else:
            out.append(C("Nexovara Pose отправляет в драйвер", True, f"режим «{mode}», порт "
                                                               f"{self.cfg.driver_port}"))
        snap = self.engine.snapshot() if (self.engine and self.engine.running) else None
        if snap is None:
            out.append(C("Трекинг Nexovara Pose", None, "не запущен — нажмите «Старт» (F5)"))
        else:
            st = snap["stats"]
            out.append(C("Трекинг Nexovara Pose", bool(snap["tracking"]),
                         "тело найдено" if snap["tracking"] else "встаньте в кадр целиком"))
            out.append(C("Пакеты в драйвер", st["send_rate"] > 0,
                         f"{st['send_rate']:.0f} в секунду"))
            out.append(C("Калибровка в Nexovara Pose", bool(snap["calibrated"]) or None,
                         "есть" if snap["calibrated"] else "нажмите F6, стоя ровно"))
            if self.cfg.output_driver and not self.cfg.driver_hmd and self.cfg.hmd_anchor:
                pose = self.engine.hmd_state.latest(1.0)
                out.append(C("Поза шлема (привязка тела)", pose is not None and not pose.virtual,
                             f"есть ({'от драйвера' if pose.source == 'driver' else 'через openvr'})"
                             if pose is not None else
                             "нет — шлем не подключён, SteamVR не запущен или драйвер старый "
                             "(пересоберите build_driver.bat и нажмите «Исправить»)"))
        if self._vrchat_desktop or steamvr.vrchat_desktop_running():
            out.append(C("VRChat в режиме VR", False, "VRChat запущен без VR — он не принимает "
                                                      "трекеры. Перезапустите VRChat в режиме VR "
                                                      "(SteamVR)"))
        return out

    def _nohmd_window(self):
        w, h = self.cfg.nohmd_window.split("x")
        return int(w), int(h)

    def _driver_action(self, act: str) -> None:
        try:
            if act == "driver_install":
                inst = steamvr.register_driver()
                msg = ("Драйвер установлен в SteamVR:\n" + str(inst) + "\n\nЗапустите SteamVR — "
                       "трекеры появятся, как только Nexovara Pose начнёт отправлять данные. Если "
                       "что-то не так — «Проверка и настройка SteamVR».")
            elif act == "driver_uninstall":
                steamvr.unregister_driver()
                msg = "Драйвер удалён из SteamVR."
            else:
                enable = not steamvr.no_hmd_mode_enabled()
                if enable and QMessageBox.question(
                        self, APP_NAME, "Включить режим «SteamVR без шлема»? SteamVR будет "
                        "использовать виртуальный шлем Nexovara Pose (картинка — в окне на мониторе). "
                        "Если у вас подключён настоящий шлем, не включайте этот режим.\n\n"
                        "Настройки SteamVR будут изменены (с резервной копией).") != QMessageBox.Yes:
                    return
                if enable and not steamvr.driver_registered():
                    steamvr.register_driver()
                backup = steamvr.set_no_hmd_mode(enable, window=self._nohmd_window(),
                                                 vertical_fov=self.cfg.nohmd_fov)
                if enable:
                    from ..config import apply_auto
                    from ..presets import PRESETS
                    self.cfg.target = "steamvr_nohmd"
                    apply_auto(self.cfg, PRESETS["steamvr_nohmd"].values, force=True)
                    save_config(self.cfg)
                    self._refresh_header()
                msg = (f"Режим без шлема {'включён' if enable else 'выключен'}. "
                       f"Резервная копия настроек SteamVR:\n{backup}")
        except Exception as e:
            QMessageBox.warning(self, APP_NAME, f"Не получилось: {e}")
            return
        QMessageBox.information(self, APP_NAME, msg)

    def _clear_logs(self) -> None:
        for p in [*self.data_dir.glob("nexovara_pose.log*"), self.data_dir / "crash.log"]:
            try:
                if not p.exists():
                    continue
                if p.name in ("nexovara_pose.log", "crash.log"):
                    p.write_text("", encoding="utf-8")
                else:
                    p.unlink()
            except OSError:
                pass
        self._status = "Журнал очищен"

    def _delete_data(self) -> None:
        if QMessageBox.question(
                self, APP_NAME, "Удалить настройки, калибровку, профили, записи движений, "
                                "результаты замеров и журнал? Модели нейросети останутся. "
                                "Программа вернётся к первому запуску.") != QMessageBox.Yes:
            return
        if self.engine:
            self._stop_engine()
        for name in ("config.json", "models.json"):
            try:
                (self.data_dir / name).unlink(missing_ok=True)
            except OSError:
                pass
        self._clear_logs()
        rec_dir = self.data_dir / "recordings"
        if rec_dir.is_dir():
            for p in rec_dir.glob("*.npz"):
                try:
                    p.unlink()
                except OSError:
                    pass
        fresh = Config().validate()
        for k, v in fresh.to_dict().items():
            setattr(self.cfg, k, v)
        self.cfg.first_run_done = True  # don't pop the wizard right now
        self._status = "Данные удалены"
        self._refresh_header()
        self._idle_cards()

    # ---------------------------------------------------------------- loop
    def _drain_events(self) -> None:
        if not self.engine:
            return
        for kind, payload in self.engine.poll_events():
            self._handle_event(kind, payload)

    def _handle_event(self, kind: str, payload) -> None:
        if kind == "status":
            self._status = payload
        elif kind == "error":
            self._status = f"Ошибка: {payload}"
            QMessageBox.critical(self, APP_NAME, f"Трекинг остановлен:\n{payload}\n\n"
                                                 f"Подробности в журнале:\n"
                                                 f"{self.data_dir / 'nexovara_pose.log'}")
        elif kind == "stopped":
            if self.engine and not self.engine.running:
                self.engine = None
                self._on_stopped()
        elif kind == "calibrated":
            self.cfg.calibration = payload
            save_config(self.cfg)
            self._refresh_profiles()
            self._status = "✔ Калибровка выполнена. Теперь откалибруйте FBT в игре."
            if self.settings:
                self.settings.refresh(self._settings_extra())
        elif kind == "calib_failed":
            self._status = payload
        elif kind == "auto_applied":
            changed = apply_auto(self.cfg, payload["values"])
            if changed:
                save_config(self.cfg)
                self._status = f"Автонастройка: {payload['why']}"
                if self.settings:
                    self.settings.refresh(self._settings_extra())
        elif kind == "recording":
            self._recording_event(payload)
        elif kind == "camera_profile":
            self.cfg.camera_profiles[payload["camera"]] = payload["profile"]
            save_config(self.cfg)
        elif kind == "cameras_resolved":
            self._cameras_resolved(payload)
        elif kind == "camera_info":
            self._run_cam = (payload.get("camera", self.cfg.camera), payload.get("name", ""))
        elif kind == "rig_calibrated":
            ok, bad = [], []
            for src, e in payload.items():
                if "R" in e:
                    self.cfg.camera_rig[src] = e
                    ok.append(f"«{self._cam_label(src)}»: ±{e['rms_cm']:.0f} см"
                              + (" (автоматически)" if e.get("auto") else ""))
                else:
                    bad.append(f"«{self._cam_label(src)}»: {e.get('error')}")
            save_config(self.cfg)
            self._status = "Калибровка камер: " + "; ".join(ok + bad)
            if self.settings:
                self.settings.refresh(self._settings_extra())
        elif kind == "target":
            p = PRESETS.get(payload["id"])
            self._target_note = p.title if p else payload["id"]
            self._status = f"Режим: {self._target_note} ({payload['reason']})"
            self._refresh_header()

    def _cameras_resolved(self, upd: dict) -> None:
        """The engine found the cameras by name: keep numbers/names up to date."""
        changed = False
        if upd.get("mapping"):
            remap_camera_sources(self.cfg, upd["mapping"])
            changed = True
        if upd.get("persist") and upd.get("camera") and \
                (upd["camera"], upd.get("camera_name", "")) != (self.cfg.camera,
                                                               self.cfg.camera_name):
            self.cfg.camera, self.cfg.camera_name = upd["camera"], upd.get("camera_name", "")
            changed = True
        extras = upd.get("extras") or []
        if len(extras) == len(self.cfg.extra_cameras):
            new = []
            for old, ex in zip(self.cfg.extra_cameras, extras):
                if ex is not None and (ex["source"], ex.get("name", "")) != \
                        (old.get("source"), old.get("name", "")):
                    old = dict(old, source=ex["source"], name=ex.get("name", ""))
                    changed = True
                new.append(old)
            self.cfg.extra_cameras = new
        self._run_cam = (upd.get("camera", self.cfg.camera), upd.get("camera_name",
                                                                     self.cfg.camera_name))
        if changed:
            save_config(self.cfg)
            if self.settings:
                self.settings.refresh(self._settings_extra())

    def _on_tick(self) -> None:
        try:
            self._drain_events()
            self._render()
        except Exception:
            log.exception("UI update failed")

    def _render(self) -> None:
        self.statusBar().showMessage(self._status)
        snap = self.engine.snapshot() if self.engine else None
        self._update_hint(snap)
        if not snap or snap["seq"] == self._last_seq:
            return
        self._last_seq = snap["seq"]
        st = snap["stats"]
        if self.cfg.show_preview and snap["frame"] is not None:
            self.video.set_data(snap["frame"], snap["image_lm"], snap["visibility"],
                                snap.get("hands"), self.cfg.visibility_threshold,
                                snap.get("others"))
        elif not self.cfg.show_preview:
            self.video.clear("Видео скрыто (Настройки → Производительность)")
        if self.cfg.show_3d:
            msg = ""
            if snap["calib_progress"] is not None:
                msg = f"Калибровка… {snap['calib_progress'] * 100:.0f}%"
                self.video.overlay_text = f"{snap['calib_progress'] * 100:.0f}%"
            elif self._countdown == 0 and self.video.overlay_text.endswith("%"):
                self.video.overlay_text = ""
            if not msg and not snap["calibrated"]:
                msg = "Не откалибровано (F6)"
            elif not msg and snap["lost"]:
                msg = "Тело потеряно"
            self.view3d.set_data(snap["pts"], snap["trackers"], snap["head"], [], msg,
                                 snap.get("statuses"))
        cs = st["cam_size"]
        mc = snap.get("multicam") or {}
        cams = mc.get("cameras") or []
        src, name = getattr(self, "_run_cam", (self.cfg.camera, self.cfg.camera_name))
        self.c_cam.setToolTip(camera_label(src, name) + "".join(
            f"\n+ {camera_label(c['source'], c.get('name', ''))}"
            + (f" — {c['error']}" if c.get("error") else "") for c in cams))
        sub = f"{short_label(src, name)}\n{cs[0]}×{cs[1]}"
        if cams:
            used = sum(1 for c in cams if c["used"])
            sub += f" · ещё камер: {len(cams)}, в 3D: {used}"
            bad = [c for c in cams if c.get("error")]
            if bad:
                sub += " · ошибка: " + ", ".join(short_label(c["source"], c.get("name", ""), 24)
                                                 for c in bad)
            elif any(c.get("rms_cm") is None for c in cams):
                sub += (" · калибруются сами — подвигайтесь" if self.cfg.rig_auto
                        else " · нужна калибровка камер")
        self.c_cam.set(f"{st['cam_fps']:.0f} fps", sub)
        if mc.get("rig_progress") is not None:
            self.video.overlay_text = f"{mc['rig_progress'] * 100:.0f}%"
        self.c_nn.set(f"{st['infer_ms']:.0f} мс",
                      f"{st['model']} · {st['delegate']} · {st['proc_fps']:.0f} кадр/с"
                      + (f"\n{st['refiner']}" if st.get("refiner") else ""))
        self.c_nn.setToolTip("Время обработки кадра нейросетью." + (
            "\nRTMPose уточняет положение суставов на картинке. «Дрожание стоп X → Y» — "
            "замер, пока вы стоите: без уточнения → с уточнением (меньше — лучше)."
            if st.get("refiner") else ""))
        person = snap.get("person") or {}
        others = max(0, int(person.get("people", 0)) - (1 if snap["tracking"] else 0))
        extra = []
        if snap.get("pose_state"):
            extra.append(snap["pose_state"])
        est = [s for s, w in zip(("левая", "правая"), snap.get("legs_estimated", [])) if w > 0.5]
        if est:
            extra.append("оценка ног: " + ", ".join(est))
        if others:
            extra.append(f"ещё {others} чел. игнорируется")
        if snap["tracking"]:
            self.c_track.set("Тело найдено", " · ".join(extra) or
                             f"трекеров: {len(snap['trackers'])}", theme.GOOD)
        elif person.get("people"):
            self.c_track.set("Ищу вас…", person.get("status") or "в кадре другие люди",
                             theme.WARN)
        else:
            self.c_track.set("Нет тела", "встаньте в кадр целиком", theme.WARN)
        outs = self._outputs_text()
        if st["paused"]:
            self.c_out.set("Пауза", outs, theme.WARN)
        elif outs:
            self.c_out.set(outs, f"{st['send_rate']:.0f} пакетов/с"
                           + (f" · плавность {st['interp_ms']:.0f} мс" if st.get("interp_ms") else "")
                           + (" · тело привязано к шлему" if st["hmd"] else "")
                           + (" · тело привязано к шлему (VRChat)" if st.get("vrc_hmd") else ""),
                           theme.GOOD if st["send_rate"] > 0 else None)
        else:
            self.c_out.set("нет", "включите вывод в настройках", theme.WARN)
        self.c_cal.set("Есть" if snap["calibrated"] else "Нужна",
                       "F6 — перекалибровать" if snap["calibrated"] else "F6 — откалибровать",
                       theme.GOOD if snap["calibrated"] else theme.WARN)
        lat = st["latency_ms"]
        self.c_lat.set(f"{lat:.0f} мс", "от кадра камеры до отправки",
                       theme.GOOD if lat < 60 else (theme.WARN if lat < 120 else theme.BAD))
        if st.get("recording") is not None and self._recording:
            self.btn_rec.setText(f"⏹  Стоп записи ({st['recording']:.0f} с)")
        conf = snap.get("confidence") or {}
        self.conf_bars.set_values([(CONF_LABELS.get(n, n), conf.get(n))
                                   for n in self.cfg.enabled_trackers()])
        if self.cfg.fingers:
            self.finger_bars.set_curls({s: h.get("curls") for s, h in snap.get("hands", {}).items()})

    def _check_environment(self) -> None:
        if self._env_busy or not (self.engine and self.engine.running):
            return
        if not (self.cfg.output_vrchat or self.cfg.output_driver):
            return
        self._env_busy = True

        def work():
            try:
                self._sig.vrchat_desktop.emit(steamvr.vrchat_desktop_running())
            except Exception:
                self._sig.vrchat_desktop.emit(False)
        threading.Thread(target=work, daemon=True, name="env-check").start()

    def _set_vrchat_desktop(self, v: bool) -> None:
        self._env_busy = False
        if v and not self._vrchat_desktop:
            log.info("VRChat runs in desktop mode: it ignores body trackers")
        self._vrchat_desktop = v

    def _update_hint(self, snap) -> None:
        ctl = (snap or {}).get("controls") or ""
        if self.engine and self.engine.running and self._vrchat_desktop:
            t = ("<b>VRChat запущен без VR</b> — в этом режиме VRChat не принимает трекеры тела "
                 "(ограничение VRChat). Без шлема: режим «SteamVR без шлема» (Настройки → Игра "
                 "и вывод → «Проверка и настройка SteamVR»), затем VRChat в режиме VR.")
        elif not (self.engine and self.engine.running):
            t = "Нажмите <b>«Старт»</b> (F5) — программа сама подберёт режим камеры и игры."
        elif snap is None:
            t = "Запуск… " + self._status
        elif snap.get("camera_lost"):
            t = ("<b>Камера не отдаёт кадры</b> — она отключена или её занимает другая программа "
                 "(браузер, Discord, OBS, Zoom). Подключите / освободите её — трекинг продолжится "
                 "сам через несколько секунд, перезапускать ничего не нужно.")
        elif snap["stats"]["paused"]:
            t = "<b>Пауза</b> — трекеры в игре заморожены. F7 — продолжить."
        elif snap["calib_progress"] is not None or self._countdown:
            t = "Калибровка: стойте ровно лицом к камере, руки вниз, смотрите прямо."
        elif not snap["tracking"] and snap["stats"].get("holding"):
            t = ("<b>Вас не видно</b> — трекеры в игре удерживаются в последней позе (до "
                 f"{self.cfg.lost_hold_s:.0f} с), калибровка не теряется. Встаньте в кадр.")
        elif not snap["tracking"] and (snap.get("person") or {}).get("people"):
            t = ("В кадре есть люди, но это не вы (или вас не узнать). <b>Кликните по себе</b> "
                 "в окне видео, чтобы следить за вами.")
        elif not snap["tracking"]:
            t = "Встаньте в кадр <b>целиком</b> — от головы до стоп, в 2–3 м от камеры."
        elif not snap["calibrated"]:
            t = "Отлично, вас видно! Встаньте ровно и нажмите <b>«Калибровка»</b> (F6)."
        else:
            outs = self._outputs_text() or "никуда (включите вывод в настройках)"
            t = (f"<b>Готово.</b> Трекеры отправляются: {outs}. "
                 "Теперь выполните калибровку FBT в самой игре.")
            if self.cfg.output_driver and self.cfg.driver_hmd and self.cfg.kb_controls:
                t += (f" Без контроллеров: <b>{self.cfg.key_menu}</b> — меню, "
                      f"<b>{self.cfg.key_pointer}</b> — рука за мышью, "
                      f"<b>{self.cfg.key_triggers}</b> — оба курка.")
        if (snap and snap["tracking"] and self.cfg.output_driver and not self.cfg.driver_hmd
                and self.cfg.hmd_anchor and not snap["stats"].get("hmd")):
            t += (" <span style='color:#f0b429'>Шлем не найден — тело не привязано к шлему. "
                  "Запустите SteamVR со шлемом; если не помогло — пересоберите драйвер "
                  "(build_driver.bat) и нажмите «Проверка и настройка SteamVR» → "
                  "«Исправить».</span>")
        if (snap and snap["tracking"] and snap["calibrated"] and snap["stats"].get("hmd")
                and not snap.get("calib_has_hmd_yaw")):
            t += (" <span style='color:#f0b429'>Калибровка сделана без шлема — направление "
                  "тела подбирается само и может «уплывать». Наденьте шлем, встаньте лицом к "
                  "камере, смотрите прямо и нажмите F6.</span>")
        if (snap and snap["tracking"] and self.cfg.output_vrchat and self.cfg.vrchat_hmd
                and not snap["stats"].get("vrc_hmd")):
            t += (" <span style='color:#f0b429'>Поза шлема из VRChat: "
                  f"{snap['stats'].get('oscquery') or 'запуск…'}.</span>")
        tips = (snap or {}).get("hints") or []
        if tips and self.engine and self.engine.running and not (snap or {}).get(
                "stats", {}).get("paused"):
            t += f" <span style='color:{theme.WARN}'>Совет: {tips[0]}</span>"
        if ctl and self.engine and self.engine.running:
            t += f" <span style='color:{theme.ACCENT}'>[{ctl}]</span>"
        if self.hint.text() != t:
            self.hint.setText(t)

    def closeEvent(self, e) -> None:
        self._save_timer.stop()
        save_config(self.cfg)
        if self.engine:
            self.engine.stop(timeout=3.0)
        if self.settings:
            self.settings.close()
        super().closeEvent(e)

