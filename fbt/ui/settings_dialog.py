"""Settings window (separate, non-modal).

Every change is written to the shared Config immediately, marked as
"changed by you" (so automatic tuning won't overwrite it) and reported to the
main window through `changed`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QMessageBox, QPushButton, QScrollArea, QSpinBox, QStackedWidget,
                               QVBoxLayout, QWidget)

from ..camera import CameraDevice, camera_label, profile_for
from ..config import (KEY_CHOICES, MODEL_VARIANTS, NOHMD_WINDOWS, TRACKER_LABELS_RU, TRACKER_NAMES,
                      Config)
from .. import profiles
from ..presets import PRESETS
from . import theme

STD_SIZES = [(640, 480), (848, 480), (960, 540), (1280, 720), (1920, 1080)]
MARK = "●"


class SettingsDialog(QDialog):
    changed = Signal(list)          # keys
    action = Signal(str)            # see _act_* buttons

    def __init__(self, cfg: Config, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Настройки — Nexovara Pose")
        self.resize(860, 640)
        self.setModal(False)
        self._loading = False
        self._fields: Dict[str, Tuple[QWidget, Optional[QLabel], Callable]] = {}
        self._labels: Dict[str, QLabel] = {}
        self._known_devs: Optional[List[CameraDevice]] = None  # None = not scanned yet
        self._data_dir: Optional[str] = None

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.nav = QListWidget()
        self.nav.setObjectName("nav")
        self.nav.setFixedWidth(210)
        self.pages = QStackedWidget()
        root.addWidget(self.nav)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.pages, 1)
        foot = QHBoxLayout()
        foot.setContentsMargins(16, 8, 16, 12)
        self.legend = QLabel(f"<span style='color:{theme.ACCENT}'>{MARK}</span> — изменено вами "
                             "(автонастройка это не трогает)")
        self.legend.setObjectName("muted")
        b_reset = QPushButton("Вернуть всё к автонастройке")
        b_reset.setToolTip("Забыть ручные изменения: при следующем запуске программа сама "
                           "подберёт камеру, режим игры и т.д.")
        b_reset.clicked.connect(lambda: self.action.emit("reset_auto"))
        b_close = QPushButton("Закрыть")
        b_close.setObjectName("primary")
        b_close.clicked.connect(self.close)
        foot.addWidget(self.legend, 1)
        foot.addWidget(b_reset)
        foot.addWidget(b_close)
        right.addLayout(foot)
        rw = QWidget()
        rw.setLayout(right)
        root.addWidget(rw, 1)

        for title, builder in [("🎮  Игра и вывод", self._page_output),
                               ("🦵  Трекеры", self._page_trackers),
                               ("👥  Человек в кадре", self._page_person),
                               ("🌊  Плавность и реализм", self._page_smooth),
                               ("📷  Камера", self._page_camera),
                               ("⚡  Производительность", self._page_perf),
                               ("🎯  Точность", self._page_accuracy),
                               ("✋  Пальцы", self._page_fingers),
                               ("🧍  Тело и калибровка", self._page_body),
                               ("👤  Профили", self._page_profiles),
                               ("📼  Запись и повтор", self._page_recordings),
                               ("🔒  Конфиденциальность", self._page_privacy)]:
            self.nav.addItem(title)
            self.pages.addWidget(self._scroll(builder()))
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        self.refresh()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _scroll(w: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setWidget(w)
        return sa

    def _page(self, title: str, intro: str = ""):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(22, 18, 22, 18)
        lay.setSpacing(10)
        t = QLabel(title)
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        if intro:
            i = QLabel(intro)
            i.setObjectName("muted")
            i.setWordWrap(True)
            lay.addWidget(i)
        return w, lay

    def _group(self, lay: QVBoxLayout, title: str) -> QFormLayout:
        g = QGroupBox(title)
        f = QFormLayout(g)
        f.setLabelAlignment(Qt.AlignLeft)
        f.setHorizontalSpacing(16)
        f.setVerticalSpacing(8)
        lay.addWidget(g)
        return f

    @staticmethod
    def _hint(form: QFormLayout, text: str) -> None:
        h = QLabel(text)
        h.setObjectName("muted")
        h.setWordWrap(True)
        form.addRow(h)

    def _label(self, key: str, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setProperty("base", text)
        self._labels[key] = lab
        return lab

    def _set(self, keys: List[str], values: List) -> None:
        if self._loading:
            return
        for k, v in zip(keys, values):
            setattr(self.cfg, k, v)
        self.cfg.mark_user_set(keys)
        self.cfg.validate()
        self._update_marks()
        self.changed.emit(list(keys))

    def _spin(self, form, key, text, lo, hi, step=1.0, decimals=0, suffix="", tip=""):
        if decimals:
            w = QDoubleSpinBox()
            w.setDecimals(decimals)
            w.setRange(float(lo), float(hi))
            w.setSingleStep(float(step))
        else:
            w = QSpinBox()
            w.setRange(int(lo), int(hi))
            w.setSingleStep(max(1, int(step)))
        if suffix:
            w.setSuffix(" " + suffix)
        w.setKeyboardTracking(False)
        w.valueChanged.connect(lambda v, k=key: self._set([k], [v]))
        if tip:
            w.setToolTip(tip)
        form.addRow(self._label(key, text), w)
        self._fields[key] = (w, None, lambda v, w=w: w.setValue(v))
        return w

    def _check(self, form, key, text, tip=""):
        w = QCheckBox(text)
        w.toggled.connect(lambda v, k=key: self._set([k], [bool(v)]))
        if tip:
            w.setToolTip(tip)
        lab = self._label(key, "")
        form.addRow(lab, w)
        self._fields[key] = (w, None, lambda v, w=w: w.setChecked(bool(v)))
        return w

    def _combo(self, form, key, text, items: List[Tuple[object, str]], tip=""):
        w = QComboBox()
        w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        w.setMinimumContentsLength(16)
        for val, label in items:
            w.addItem(label, val)
        w.currentIndexChanged.connect(lambda _i, k=key, w=w: self._set([k], [w.currentData()]))
        if tip:
            w.setToolTip(tip)
        form.addRow(self._label(key, text), w)

        def setter(v, w=w):
            i = w.findData(v)
            if i >= 0:
                w.setCurrentIndex(i)
        self._fields[key] = (w, None, setter)
        return w

    # ------------------------------------------------------------ pages
    def _page_output(self):
        w, lay = self._page("Игра и вывод", "Куда отправлять трекеры. В режиме «Авто» программа при "
                                            "каждом запуске смотрит, что открыто (SteamVR, VRChat), "
                                            "и выбирает сама.")
        f = self._group(lay, "Режим")
        items = [("auto", "Авто (определять при запуске)")] + \
                [(p.id, p.title) for p in PRESETS.values()]
        self._combo(f, "target", "Игра / приложение", items)
        self.target_desc = QLabel()
        self.target_desc.setObjectName("muted")
        self.target_desc.setWordWrap(True)
        f.addRow(self.target_desc)

        f = self._group(lay, "VRChat OSC")
        self._check(f, "output_vrchat", "Отправлять в VRChat")
        self._spin(f, "vrchat_port", "Порт", 1, 65535)
        self._check(f, "send_head", "Отправлять голову (выравнивание позиции)",
                    "VRChat совмещает голову Nexovara Pose со своей. Поворот (курс) выравнивается "
                    "один раз — после калибровки (F6) или кнопкой ниже, чтобы тело не "
                    "«уплывало» и не кривилось.")
        self._check(f, "vrchat_hmd", "Брать позу шлема из VRChat (OSCQuery) — точнее",
                    "VRChat сам сообщает, где ваш шлем, и тело «приклеивается» к нему, как в "
                    "режиме драйвера SteamVR: голова и поворот берутся у шлема, а не с камеры "
                    "— меньше дрожания и тело не «уплывает». Nexovara Pose объявляет себя для VRChat "
                    "на этом ПК (mDNS, адрес 127.0.0.1). В VRChat появится сообщение — "
                    "подтвердите передачу данных трекинга. Если VRChat позу не присылает, "
                    "всё работает как раньше.")
        b = QPushButton("Выровнять направление в VRChat")
        b.setToolTip("Встаньте лицом к камере и смотрите прямо, затем нажмите. Трекинг должен "
                     "быть запущен.")
        b.clicked.connect(lambda: self.action.emit("vrchat_align"))
        f.addRow(b)
        self._hint(f, "В VRChat включите OSC: Action Menu → Options → OSC → Enabled, затем "
                      "Quick Menu → Tracking & IK → Toggle FBT и Calibrate FBT. Пальцы VRChat "
                      "по OSC не принимает. Работает только в VR (со шлемом).")

        f = self._group(lay, "Драйвер Nexovara Pose для SteamVR (C++)")
        self._check(f, "output_driver", "Отправлять в драйвер Nexovara Pose",
                    "Трекеры и руки-контроллеры с пальцами напрямую в SteamVR (любые "
                    "SteamVR-игры с FBT).")
        self._check(f, "hmd_anchor", "Привязывать тело к настоящему шлему",
                    "Драйвер сообщает, где ваш шлем, и тело «приклеивается» к нему (курс — "
                    "по калибровке F6). Выключите, только если трекеры нужны без шлема.")
        self._check(f, "steamvr_prediction", "Упреждение трекеров в SteamVR (меньше отставание)",
                    "Драйвер сообщает SteamVR скорость трекеров, и SteamVR рисует их с тем же "
                    "упреждением, что шлем и контроллеры, и плавно между пакетами. Скорость "
                    "сглажена, у стоящего трекера она ровно 0. Если трекеры дрожат — выключите.")
        self.steamvr_state = QLabel()
        self.steamvr_state.setObjectName("muted")
        self.steamvr_state.setWordWrap(True)
        f.addRow(self.steamvr_state)
        row = QHBoxLayout()
        b = QPushButton("Назначить роли трекерам в SteamVR…")
        b.setToolTip("Пропишет Талия / Левая нога / Правая нога и т.д. для трекеров Nexovara Pose в "
                     "настройках SteamVR (с резервной копией). SteamVR должен быть закрыт.")
        b.clicked.connect(lambda: self.action.emit("assign_roles"))
        b2 = QPushButton("Обновить")
        b2.setObjectName("flat")
        b2.clicked.connect(lambda: self.action.emit("refresh_steamvr"))
        row.addWidget(b)
        row.addWidget(b2)
        row.addStretch(1)
        f.addRow(row)
        self._check(f, "driver_hmd", "SteamVR без шлема (виртуальный шлем с камеры)",
                    "Голова и руки тоже с камеры, игра показывается в окне на мониторе.")
        self._check(f, "hmd_head_rotation", "Виртуальный шлем поворачивается вместе с головой",
                    "Выключено: вид ровный и следует за корпусом — удобно, когда смотрите в "
                    "монитор.")
        self._spin(f, "driver_port", "Порт драйвера", 1024, 65535)
        self.driver_state = QLabel()
        self.driver_state.setObjectName("muted")
        self.driver_state.setWordWrap(True)
        f.addRow(self.driver_state)
        b = QPushButton("Проверка и настройка SteamVR…")
        b.setObjectName("primary")
        b.setToolTip("Пошагово проверяет: драйвер собран → установлен → включён → SteamVR его "
                     "загрузил → данные доходят. Кнопка «Исправить» чинит то, что можно.")
        b.clicked.connect(lambda: self.action.emit("steamvr_check"))
        f.addRow(b)
        buttons = [
            ("Установить / обновить в SteamVR", "driver_install",
             "Копирует собранный драйвер в папку без русских букв (C:\\ProgramData\\Nexovara Pose) "
             "и прописывает его в SteamVR. SteamVR должен быть закрыт."),
            ("Удалить из SteamVR", "driver_uninstall", ""),
            ("Включить/выключить режим без шлема", "driver_nohmd",
             "Меняет настройки SteamVR (с резервной копией). SteamVR должен быть закрыт.")]
        for chunk in (buttons[:2], buttons[2:]):   # two rows: the page never gets too wide
            row = QHBoxLayout()
            for text, act, tip in chunk:
                b = QPushButton(text)
                if tip:
                    b.setToolTip(tip)
                b.clicked.connect(lambda _c=False, a=act: self.action.emit(a))
                row.addWidget(b)
            row.addStretch(1)
            f.addRow(row)
        self._hint(f, "Драйвер собирается один раз: запустите build_driver.bat в папке программы "
                      "(нужны бесплатные Visual Studio Build Tools — скрипт предложит поставить). "
                      "После каждой пересборки нажмите «Установить / обновить».")

        f = self._group(lay, "Без шлема: окно игры и кнопки контроллеров")
        self._combo(f, "nohmd_window", "Размер окна шлема",
                    [(v, v.replace("x", "×")) for v in NOHMD_WINDOWS])
        self._spin(f, "nohmd_fov", "Угол обзора (верт.)", 50, 120, 1, 0, "°")
        self._check(f, "kb_controls", "Кнопки контроллеров с клавиатуры и мыши",
                    "Без настоящих контроллеров VRChat нельзя открыть меню и откалибровать FBT. "
                    "Клавиши работают, даже когда активно окно игры.")
        keys = [(k, "выкл." if k == "off" else k) for k in KEY_CHOICES]
        self._combo(f, "key_menu", "Меню (кнопка B, VRChat Quick Menu)", keys)
        self._combo(f, "key_triggers", "Оба курка (калибровка FBT)", keys)
        self._combo(f, "key_pointer", "Указка мышью (вкл/выкл)", keys)
        self._hint(f, "VRChat без шлема: запустите SteamVR, затем VRChat в режиме VR. Встаньте в "
                      "кадр → F8 (меню) → F10 и наведите правую руку мышью на «Calibrate FBT» → "
                      "клик → встаньте ровно → F9. Ходить можно клавишами WASD. Размер окна и "
                      "угол применяются кнопкой «Включить режим без шлема» / «Исправить» при "
                      "закрытом SteamVR.")

        f = self._group(lay, "Общее")
        self._check(f, "root_motion", "Оценивать перемещение по кадру",
                    "Для режимов без шлема/головы: приседания и шаги считаются по картинке.")
        self._hint(f, "Программа работает только на этом ПК: данные трекеров отправляются "
                      "исключительно сюда же (127.0.0.1) и никогда не уходят в сеть.")
        lay.addStretch(1)
        return w

    def _page_person(self):
        w, lay = self._page("Человек в кадре", "Если в кадре несколько людей, программа следит "
                                               "только за вами: запоминает цвет одежды, "
                                               "пропорции тела и где вы стоите. Остальные "
                                               "показываются серым пунктиром и игнорируются.")
        f = self._group(lay, "Слежение")
        self._check(f, "person_lock", "Следить только за одним человеком")
        self._spin(f, "max_people", "Сколько людей искать в кадре", 1, 5,
                   tip="Больше — надёжнее отличает вас от других, но чуть больше нагрузка, "
                       "когда в кадре несколько человек.")
        self._spin(f, "relock_single_s", "Перезахват, если в кадре один", 0, 30, 0.5, 1, "с",
                   tip="Если вас не узнали, а в кадре ровно один человек дольше этого времени — "
                       "следить за ним. 0 — никогда (строгий режим).")
        b = QPushButton("Выбрать человека заново")
        b.setToolTip("Забыть, за кем следим: возьмём самого крупного/центрального")
        b.clicked.connect(lambda: self.action.emit("relock_person"))
        f.addRow(b)
        self._hint(f, "Самый простой способ выбрать — кликнуть по нужному человеку в окне "
                      "видео. Калибровка (F6) тоже закрепляет того, кто калибровался.")
        lay.addStretch(1)
        return w

    def _page_smooth(self):
        w, lay = self._page("Плавность и реализм", "Математика, которая делает движения "
                                                   "плавными и естественными (идеи из программ "
                                                   "оценки тела вроде Standable).")
        f = self._group(lay, "Плавный вывод")
        self._check(f, "output_smoothing", "Сглаживать вывод (интерполяция между кадрами)",
                    "Камера даёт 30–60 кадров/с, а трекеры отправляются с частотой ниже — "
                    "промежуточные положения вычисляются. Движение без «ступенек».")
        self._spin(f, "send_rate", "Частота отправки", 10, 240, suffix="Гц")
        self._spin(f, "interp_delay_ms", "Задержка интерполяции (0 = авто)", 0, 100, suffix="мс")
        self._spin(f, "prediction_ms", "Предсказание движения", 0, 60, suffix="мс",
                   tip="Компенсирует задержку камеры, но при больших значениях может "
                       "«перелетать» при резких остановках.")
        self._check(f, "rotation_smoothing", "Плавные повороты трекеров")
        f = self._group(lay, "Против дрожания")
        self._check(f, "lag_comp", "Сглаживание с компенсацией задержки",
                    "Каждая часть тела сглаживается по-своему (стопы сильнее), а задержка "
                    "сглаживания компенсируется — без «перелёта» при остановке. По замерам: "
                    "дрожание −10…17 %, задержка на ~20 мс меньше, чем раньше.")
        self._check(f, "stabilizer", "Стабилизатор (трекеры стоят, когда вы стоите)",
                    "Мелкая дрожь в несколько миллиметров гасится, движения проходят без "
                    "задержки. Размер «мёртвой зоны» подстраивается под шум каждого трекера.")
        self._check(f, "leg_swap_fix", "Исправлять перепутанные левую и правую ногу",
                    "Нейросеть иногда путает ноги (ноги скрещены, вы стоите боком) — тогда "
                    "трекеры стоп прыгают друг на друга. Программа это замечает и исправляет.")
        self._check(f, "posture_smoothing", "Сидя и лёжа — ноги спокойнее",
                    "Когда вы сидите или лежите, ноги почти не двигаются, но хуже всего видны — "
                    "их сглаживание усиливается. Быстрые движения проходят как обычно. Стоя "
                    "ничего не меняется. Нужна калибровка (F6) и «Позы стоя / сидя / лёжа».")
        f = self._group(lay, "Реализм")
        self._check(f, "joint_limits", "Колени сгибаются только вперёд",
                    "Исправляет частую ошибку одной камеры, когда колено «выгибается» назад.")
        self._check(f, "leg_estimation", "Достраивать ноги, которых не видно",
                    "Если ноги закрыты (стол, край кадра), они оцениваются по позе тела: стоя "
                    "— прямые, сидя — согнутые. Плавно переключается обратно на камеру.")
        self._check(f, "pose_states", "Позы стоя / сидя / лёжа с плавными переходами")
        self._check(f, "auto_floor", "Автоматически уточнять уровень пола")
        self._hint(f, "Работает после калибровки (F6). В 3D-виде цвет трекера показывает: "
                      "оранжевый — по камере, фиолетовый — оценка, красный — удерживается.")
        lay.addStretch(1)
        return w

    def _page_trackers(self):
        w, lay = self._page("Трекеры", "С одной камерой лучше всего работают 3 трекера (таз + "
                                       "стопы): чем меньше трекеров, тем лучше IK игры исправляет "
                                       "неточности. У каждого трекера постоянный номер.")
        f = self._group(lay, "Включённые трекеры")
        self._tracker_boxes = {}
        for i, n in enumerate(TRACKER_NAMES):
            cb = QCheckBox(f"{TRACKER_LABELS_RU[n]}")
            cb.toggled.connect(lambda _v: self._trackers_changed())
            sub = QLabel(f"VRChat #{i + 1} · SteamVR Nexovara Pose-{n}")
            sub.setObjectName("muted")
            f.addRow(cb, sub)
            self._tracker_boxes[n] = cb
        self._labels["trackers"] = QLabel()
        lay.addStretch(1)
        return w

    def _trackers_changed(self):
        if self._loading:
            return
        self._set(["trackers"], [{n: cb.isChecked() for n, cb in self._tracker_boxes.items()}])

    def _page_camera(self):
        w, lay = self._page("Камера", "При первом запуске программа сама проверяет все режимы "
                                      "камеры, измеряет реальный FPS и выбирает лучший.")
        f = self._group(lay, "Источник")
        row = QHBoxLayout()
        self.cam_combo = QComboBox()
        self.cam_combo.setEditable(True)
        self.cam_combo.lineEdit().editingFinished.connect(self._camera_edited)
        self.cam_combo.activated.connect(lambda _i: self._camera_edited())
        self.cam_combo.setMinimumWidth(300)
        self.cam_combo.setToolTip("Камера (название и номер) или путь к видеофайлу")
        b_find = QPushButton("Найти")
        b_find.setToolTip("Обновить список камер")
        b_find.clicked.connect(lambda: self.action.emit("scan_cameras"))
        b_file = QPushButton("Видеофайл…")
        b_file.clicked.connect(lambda: self.action.emit("pick_video"))
        row.addWidget(self.cam_combo, 1)
        row.addWidget(b_find)
        row.addWidget(b_file)
        f.addRow(self._label("camera", "Камера"), row)

        f = self._group(lay, "Режим")
        self.mode_combo = QComboBox()
        self.mode_combo.activated.connect(self._mode_chosen)
        f.addRow(self._label("width", "Разрешение и FPS"), self.mode_combo)
        self._spin(f, "fps", "Запрашивать FPS", 5, 240)
        self._combo(f, "fourcc", "Формат кадров", [("auto", "Авто (MJPG)"), ("MJPG", "MJPG"),
                                                    ("YUY2", "YUY2 (без сжатия)")])
        b = QPushButton("Заново определить возможности камеры")
        b.clicked.connect(lambda: self.action.emit("reprobe_camera"))
        f.addRow(b)
        self.cam_note = QLabel()
        self.cam_note.setObjectName("muted")
        self.cam_note.setWordWrap(True)
        f.addRow(self.cam_note)

        f = self._group(lay, "Изображение")
        self._combo(f, "rotate", "Поворот", [(0, "0°"), (90, "90°"), (180, "180°"), (270, "270°")])
        self._check(f, "mirror", "Отразить изображение",
                    "Включите, если в игре левая и правая сторона перепутаны.")
        self._spin(f, "camera_vfov", "Угол обзора (верт.)", 20, 120, 1, 1, "°")

        f = self._group(lay, "Дополнительные камеры (точнее в 3D)")
        self._hint(f, "Вторая камера под углом 45–90° к первой исправляет главную слабость "
                      "одной камеры — глубину. Подойдут USB-вебкамеры и телефон как веб-камера "
                      "(DroidCam, Iriun, Camo — он появится в списке камер как обычная).")
        self._check(f, "multicam_fusion", "Объединять данные камер")
        self._check(f, "rig_auto", "Калибровать камеры автоматически",
                    "Пока вы просто двигаетесь, программа сама вычисляет, как повёрнуты "
                    "камеры друг относительно друга (нужно ~2 с, когда вас видят обе камеры и "
                    "в кадре больше никого). Если камеру сдвинули — пересчитывает. Плохой "
                    "результат никогда не сохраняется. Кнопка ниже — ручная калибровка.")
        self._extra_box = QVBoxLayout()
        holder = QWidget()
        holder.setLayout(self._extra_box)
        f.addRow(holder)
        row = QHBoxLayout()
        b_add = QPushButton("Добавить камеру")
        b_add.clicked.connect(self._extra_add)
        b_rig = QPushButton("Калибровка камер (8 с)")
        b_rig.setToolTip("Встаньте так, чтобы вас видели все камеры, и медленно двигайтесь: "
                         "поворачивайтесь, поднимайте руки. Трекинг должен быть запущен.")
        b_rig.clicked.connect(lambda: self.action.emit("calibrate_rig"))
        row.addWidget(b_add)
        row.addWidget(b_rig)
        row.addStretch(1)
        f.addRow(row)
        self.rig_state = QLabel()
        self.rig_state.setObjectName("muted")
        self.rig_state.setWordWrap(True)
        f.addRow(self.rig_state)
        self._labels["extra_cameras"] = QLabel()
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------ extra cameras
    def _extra_rebuild(self) -> None:
        while self._extra_box.count():
            it = self._extra_box.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, spec in enumerate(self.cfg.extra_cameras):
            roww = QWidget()
            r = QHBoxLayout(roww)
            r.setContentsMargins(0, 0, 0, 0)
            src = QComboBox()
            src.setEditable(True)
            src.setMinimumWidth(220)
            self._fill_cam_combo(src, spec["source"], spec.get("name", ""))
            src.setToolTip(camera_label(spec["source"], spec.get("name", ""))
                           + "\nКамера из списка, её номер или путь к видеофайлу")
            src.lineEdit().editingFinished.connect(lambda i=i, w=src: self._extra_set_source(i, w))
            src.activated.connect(lambda _x, i=i, w=src: self._extra_set_source(i, w))
            rot = QComboBox()
            for v in (0, 90, 180, 270):
                rot.addItem(f"{v}°", v)
            rot.setCurrentIndex((0, 90, 180, 270).index(spec["rotate"]))
            rot.currentIndexChanged.connect(lambda _x, i=i, w=rot: self._extra_set(i, "rotate", w.currentData()))
            mir = QCheckBox("зеркало")
            mir.setChecked(spec["mirror"])
            mir.toggled.connect(lambda v, i=i: self._extra_set(i, "mirror", bool(v)))
            en = QCheckBox("вкл")
            en.setChecked(spec["enabled"])
            en.toggled.connect(lambda v, i=i: self._extra_set(i, "enabled", bool(v)))
            rm = QPushButton("✕")
            rm.setObjectName("flat")
            rm.setToolTip("Убрать камеру")
            rm.clicked.connect(lambda _c=False, i=i: self._extra_remove(i))
            r.addWidget(QLabel(f"Камера {i + 2}:"))
            r.addWidget(src, 1)
            r.addWidget(rot)
            r.addWidget(mir)
            r.addWidget(en)
            r.addWidget(rm)
            self._extra_box.addWidget(roww)
        rigs = []
        for spec in self.cfg.extra_cameras:
            e = self.cfg.camera_rig.get(spec["source"])
            rigs.append(f"«{camera_label(spec['source'], spec.get('name', ''))}»: " + (
                f"откалибрована, точность ±{e.get('rms_cm', 0):.0f} см" if e and e.get("R")
                else "нужна калибровка камер"))
        self.rig_state.setText("; ".join(rigs) if rigs else "Дополнительных камер нет.")

    def _extra_apply(self, cams) -> None:
        if self._loading:
            return
        self._set(["extra_cameras"], [cams])
        self._extra_rebuild()

    def _extra_set(self, i: int, key: str, value) -> None:
        cams = [dict(c) for c in self.cfg.extra_cameras]
        if i < len(cams) and cams[i].get(key) != value and value != "":
            cams[i][key] = value
            self._extra_apply(cams)

    def _extra_set_source(self, i: int, w: QComboBox) -> None:
        src, name = self._combo_source(w)
        cams = [dict(c) for c in self.cfg.extra_cameras]
        if not src or i >= len(cams) or (cams[i]["source"], cams[i].get("name", "")) == (src, name):
            return
        if src == self.cfg.camera:
            self._fill_cam_combo(w, cams[i]["source"], cams[i].get("name", ""))
            return  # the main camera can't be an extra camera at the same time
        cams[i]["source"], cams[i]["name"] = src, name
        self._extra_apply(cams)

    def _extra_remove(self, i: int) -> None:
        cams = [dict(c) for c in self.cfg.extra_cameras]
        if i < len(cams):
            cams.pop(i)
            self._extra_apply(cams)

    def _extra_add(self) -> None:
        cams = [dict(c) for c in self.cfg.extra_cameras]
        if len(cams) >= 3:
            return
        used = {self.cfg.camera, *[c["source"] for c in cams]}
        free = [d for d in (self._known_devs or []) if str(d.index) not in used]
        if free:
            src, name = str(free[0].index), free[0].name
        else:
            src, name = next((str(i) for i in range(10) if str(i) not in used), "1"), ""
        cams.append({"source": src, "name": name, "rotate": 0, "mirror": False,
                     "enabled": True})
        self._extra_apply(cams)

    # ------------------------------------------------------------ camera lists
    def _dev_name(self, src: str) -> str:
        for d in self._known_devs or []:
            if str(d.index) == src:
                return d.name
        return ""

    def _combo_source(self, w: QComboBox):
        """(source, name) of a camera combo: a listed camera or typed number/path."""
        txt = w.currentText().strip()
        i = w.findText(txt)
        if i >= 0 and w.itemData(i):
            return tuple(w.itemData(i))
        if txt.startswith("#") and txt[1:].isdigit():
            txt = txt[1:]
        return txt, (self._dev_name(txt) if txt.isdigit() else "")

    def _fill_cam_combo(self, w: QComboBox, cur: str, cur_name: str) -> None:
        """Known cameras by name; the configured one is selected (and added with a
        note when it isn't connected right now)."""
        w.blockSignals(True)
        w.clear()
        sel = -1
        for d in self._known_devs or []:
            w.addItem(d.label(), (str(d.index), d.name))
            if str(d.index) == cur and (not cur_name or d.name == cur_name):
                sel = w.count() - 1
        if sel < 0 and cur:
            label = camera_label(cur, cur_name)
            if self._known_devs is not None and cur.isdigit():
                label += " — не подключена"
            w.addItem(label, (cur, cur_name))
            sel = w.count() - 1
        w.setCurrentIndex(sel)
        w.blockSignals(False)

    def _camera_edited(self):
        src, name = self._combo_source(self.cam_combo)
        if src and (src, name) != (self.cfg.camera, self.cfg.camera_name):
            self._set(["camera", "camera_name"], [src, name])
            self._fill_cam_combo(self.cam_combo, self.cfg.camera, self.cfg.camera_name)
            self._extra_rebuild()
            self._fill_modes()

    def _mode_chosen(self, _i):
        d = self.mode_combo.currentData()
        if d:
            w, h, fps, fcc = d
            keys, vals = ["width", "height"], [w, h]
            if fps:
                keys += ["fps", "fourcc"]
                vals += [int(round(min(fps, 120))), fcc if fcc in ("MJPG", "YUY2") else "auto"]
            self._set(keys, vals)

    def _fill_modes(self):
        self.mode_combo.blockSignals(True)
        self.mode_combo.clear()
        prof = profile_for(self.cfg.camera_profiles, self.cfg.camera, self.cfg.camera_name)
        cur = (self.cfg.width, self.cfg.height)
        idx = -1
        if prof and prof.get("modes"):
            modes = sorted(prof["modes"], key=lambda m: (-m["fps"], -m["width"]))
            for m in modes:
                self.mode_combo.addItem(f"{m['width']}×{m['height']} · {m['fps']:.0f} fps · "
                                        f"{m['fourcc']}",
                                        (m["width"], m["height"], m["fps"], m["fourcc"]))
                if (m["width"], m["height"]) == cur and idx < 0:
                    idx = self.mode_combo.count() - 1
            self.cam_note.setText(f"Список — результат замера камеры "
                                  f"«{camera_label(self.cfg.camera, self.cfg.camera_name)}» "
                                  "(реальный FPS).")
        else:
            for (w, h) in STD_SIZES:
                self.mode_combo.addItem(f"{w}×{h}", (w, h, None, None))
                if (w, h) == cur:
                    idx = self.mode_combo.count() - 1
            self.cam_note.setText(f"Камера «{camera_label(self.cfg.camera, self.cfg.camera_name)}»"
                                  " ещё не проверялась — режимы будут измерены при "
                                  "следующем запуске трекинга.")
        if idx < 0:
            self.mode_combo.addItem(f"{cur[0]}×{cur[1]} (текущий)", (cur[0], cur[1], None, None))
            idx = self.mode_combo.count() - 1
        self.mode_combo.setCurrentIndex(idx)
        self.mode_combo.blockSignals(False)

    def set_cameras(self, devs: List[CameraDevice]) -> None:
        self._known_devs = list(devs)
        self._fill_cam_combo(self.cam_combo, self.cfg.camera, self.cfg.camera_name)
        self._extra_rebuild()

    def _page_perf(self):
        w, lay = self._page("Производительность", "Автонастройка измеряет скорость нейросети на "
                                                  "вашем ПК и сама выбирает модель с запасом для игры.")
        f = self._group(lay, "Нейросеть")
        self._combo(f, "model", "Модель", [(v, {"lite": "lite — быстрая",
                                                 "full": "full — баланс",
                                                 "heavy": "heavy — точная, тяжёлая"}[v])
                                            for v in MODEL_VARIANTS])
        self._check(f, "use_gpu", "MediaPipe на GPU (только Linux/macOS)",
                    "На Windows MediaPipe работает только на процессоре. Видеокарту на Windows "
                    "использует вторая нейросеть RTMPose (ниже).")
        self._spin(f, "min_detection", "Порог обнаружения", 0.1, 0.95, 0.05, 2)
        self._spin(f, "min_presence", "Порог присутствия", 0.1, 0.95, 0.05, 2)
        self._spin(f, "min_tracking", "Порог трекинга", 0.1, 0.95, 0.05, 2)
        self.perf_note = QLabel()
        self.perf_note.setObjectName("muted")
        self.perf_note.setWordWrap(True)
        f.addRow(self.perf_note)
        from ..model_catalog import MAX_EXTRA_MODELS, available_options

        f = self._group(lay, "Ансамбль моделей")
        mode = self._combo(f, "ensemble_mode", "Режим",
                           [("legacy", "Одна модель / RTMPose"),
                            ("auto", "Авто — по скорости обработки"),
                            ("manual", "Выбрать модели вручную")])
        legacy = self._combo(f, "refiner", "Уточнение в обычном режиме",
                             [("off", "Выключено"), ("rtmpose_m", "RTMPose-m"),
                              ("rtmpose_x", "RTMPose-x")])
        choices = QWidget()
        boxes = QVBoxLayout(choices)
        boxes.setContentsMargins(0, 0, 0, 0)
        extras = {}
        for option in available_options():
            if option.runtime != "ONNX Runtime":
                continue
            box = QCheckBox(option.title)
            extras[option.id] = box
            boxes.addWidget(box)
        f.addRow(self._label("ensemble_models", "До четырёх дополнительных"), choices)

        def update_choices():
            manual = mode.currentData() == "manual"
            at_limit = sum(box.isChecked() for box in extras.values()) >= MAX_EXTRA_MODELS
            for box in extras.values():
                box.setEnabled(manual and (box.isChecked() or not at_limit))
            legacy.setEnabled(mode.currentData() == "legacy")

        def set_choices(values):
            for key, box in extras.items():
                box.setChecked(key in values)
            update_choices()

        def on_choice(_checked):
            if not self._loading:
                self._set(["ensemble_models"], [[k for k, b in extras.items() if b.isChecked()]])
                update_choices()

        self._fields["ensemble_models"] = (choices, None, set_choices)
        for box in extras.values():
            box.toggled.connect(on_choice)
        mode.currentIndexChanged.connect(update_choices)
        self._hint(f, "MediaPipe даёт глубину. Дополнительные модели уточняют суставы на том же "
                      "кадре. В авто используются только скачанные модели, которые укладываются "
                      "в бюджет кадра. Модели для пальцев включаются отдельно.")
        downloads = QPushButton("Модели: скачать заранее")
        downloads.clicked.connect(lambda: self.action.emit("models"))
        f.addRow(downloads)
        self._check(f, "refiner_gpu", "Использовать GPU для ONNX, если доступен")
        self._spin(f, "refiner_strength", "Сила уточнения", 0, 1, 0.05, 2,
                   tip="0 — только MediaPipe. 1 — максимальный вклад согласованных 2D-точек.")
        self.refiner_note = QLabel()
        self.refiner_note.setObjectName("muted")
        self.refiner_note.setWordWrap(True)
        f.addRow(self.refiner_note)
        f = self._group(lay, "Отображение")
        self._check(f, "show_preview", "Показывать видео с камеры")
        self._check(f, "show_3d", "Показывать 3D-скелет")
        self._hint(f, "Отключите, чтобы сэкономить немного процессора во время игры.")
        lay.addStretch(1)
        return w

    _providers: Optional[List[str]] = None

    def _refiner_text(self) -> str:
        if SettingsDialog._providers is None:
            from ..rtmpose import available_providers
            SettingsDialog._providers = available_providers()
        prov = SettingsDialog._providers
        if not prov:
            return ("Не установлен ONNX Runtime. Запустите setup.bat или setup.sh "
                    "и установите дополнительные модули.")
        from ..rtmpose import GPU_PROVIDERS
        gpu = [p.replace("ExecutionProvider", "") for p in prov if p in GPU_PROVIDERS]
        device = ", ".join(gpu) or "CPU"
        return f"Доступные вычисления ONNX: {device}. Состав ансамбля виден в карточке «Нейросеть»."

    def _page_accuracy(self):
        w, lay = self._page("Точность", "Эти функции включаются после калибровки (F6).")
        f = self._group(lay, "Стабилизация тела")
        self._check(f, "bone_constraints", "Постоянная длина костей",
                    "Длины ног/рук, измеренные при калибровке, сохраняются в каждом кадре — "
                    "убирает «растягивание» ног из-за ошибок глубины.")
        self._check(f, "foot_lock", "Фиксация стоп на полу",
                    "Стоящая на полу нога не «скользит».")
        self._check(f, "floor_clamp", "Не опускать стопы ниже пола")
        f = self._group(lay, "Фильтр (One Euro)")
        self._spin(f, "filter_min_cutoff", "Мин. частота среза", 0.05, 20, 0.1, 2,
                   tip="Меньше — плавнее в покое, но больше задержка.")
        self._spin(f, "filter_beta", "Бета (скорость)", 0, 50, 0.1, 2,
                   tip="Больше — меньше задержка при быстрых движениях.")
        self._spin(f, "root_min_cutoff", "Сглаживание перемещения", 0.05, 10, 0.05, 2)
        self._spin(f, "visibility_threshold", "Порог видимости точек", 0, 0.99, 0.05, 2,
                   tip="Плохо видимые точки «замораживаются», а не прыгают.")
        self._spin(f, "max_speed", "Макс. скорость", 0.5, 50, 0.5, 1, "м/с")
        self._spin(f, "hold_time", "Удержание при потере", 0, 10, 0.1, 1, "с")
        self._spin(f, "lost_hold_s", "Держать трекеры в игре, если вас не видно", 0, 120, 1, 0,
                   "с", tip="Отошли, подошли слишком близко, отвернулись: трекеры остаются в "
                            "последней позе и не пропадают из игры — калибровка в игре не "
                            "сбивается. 0 — сразу убирать.")
        lay.addStretch(1)
        return w

    def _page_fingers(self):
        w, lay = self._page("Пальцы", "Отслеживание пальцев по камере. По умолчанию выключено: "
                                      "нагружает процессор и требует, чтобы руки были хорошо видны.")
        f = self._group(lay, "Пальцы")
        self._check(f, "fingers", "Отслеживать пальцы")
        self._combo(f, "finger_output", "Куда отправлять",
                    [("preview", "Только показывать в программе"),
                     ("driver", "SteamVR (драйвер Nexovara Pose): руки Index + пальцы")])
        self._spin(f, "hand_every", "Распознавать каждый N-й кадр", 1, 6)
        self._spin(f, "finger_min_cutoff", "Сглаживание пальцев", 0.05, 20, 0.1, 2)
        self._hint(f, "VRChat не принимает пальцы через OSC. В SteamVR руки появятся как "
                      "дополнительные контроллеры Index — используйте это, только если у вас "
                      "нет своих контроллеров, иначе они будут конфликтовать.")
        lay.addStretch(1)
        return w

    def _page_body(self):
        w, lay = self._page("Тело и калибровка")
        f = self._group(lay, "Размеры и смещения")
        self._spin(f, "user_height", "Ваш рост", 0.8, 2.5, 0.01, 2, "м")
        self._spin(f, "scale_multiplier", "Множитель масштаба", 0.3, 3, 0.01, 2)
        self._spin(f, "yaw_offset", "Поворот", -180, 180, 1, 0, "°")
        self._spin(f, "offset_x", "Смещение X", -10, 10, 0.01, 2, "м")
        self._spin(f, "offset_y", "Смещение Y", -10, 10, 0.01, 2, "м")
        self._spin(f, "offset_z", "Смещение Z", -10, 10, 0.01, 2, "м")
        f = self._group(lay, "Калибровка")
        self.calib_state = QLabel()
        self.calib_state.setWordWrap(True)
        f.addRow(self.calib_state)
        b = QPushButton("Сбросить калибровку")
        b.clicked.connect(lambda: self.action.emit("reset_calibration"))
        f.addRow(b)
        self._hint(f, "Встаньте ровно лицом к камере, чтобы было видно всё тело от головы до "
                      "стоп, руки вниз, смотрите прямо. Нажмите F6 в главном окне.")
        lay.addStretch(1)
        return w

    def _page_profiles(self):
        w, lay = self._page("Профили", "Профиль — набор настроек для игры, аватара или "
                                       "человека: режим игры, рост и масштаб, калибровка (длины "
                                       "ног и рук), включённые трекеры и сглаживание. "
                                       "Переключение — список в верхней панели или Ctrl+1…9.")
        f = self._group(lay, "Мои профили")
        self.prof_list = QListWidget()
        self.prof_list.setMinimumHeight(140)
        self.prof_list.itemDoubleClicked.connect(lambda _it: self._prof_apply())
        f.addRow(self.prof_list)
        row = QHBoxLayout()
        for text, fn, tip in [
                ("Применить", self._prof_apply, "Переключиться на выбранный профиль"),
                ("Сохранить текущие как новый…", self._prof_new,
                 "Запомнить текущие настройки и калибровку под новым именем"),
                ("Обновить выбранный", self._prof_update,
                 "Записать текущие настройки и калибровку в выбранный профиль")]:
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)
        row = QHBoxLayout()
        for text, fn, obj in [("Переименовать…", self._prof_rename, ""),
                              ("Удалить", self._prof_delete, "danger")]:
            b = QPushButton(text)
            if obj:
                b.setObjectName(obj)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)
        self._hint(f, "Пример: «VRChat — Аватар 1», «VRChat — маленький аватар» (другой "
                      "масштаб), «SteamVR — Beat Saber» (без трекеров локтей), «Брат» (другой "
                      "рост и своя калибровка). После перестановки камеры откалибруйтесь (F6) и "
                      "нажмите «Обновить выбранный».")
        lay.addStretch(1)
        return w

    def _prof_fill(self) -> None:
        cur = self._prof_selected() or self.cfg.active_profile
        self.prof_list.clear()
        for name in profiles.names(self.cfg):
            p = self.cfg.profiles[name]
            tp = PRESETS.get(p.get("target", ""))
            parts = [tp.title if tp else "режим: авто",
                     f"рост {p.get('user_height', 0):.2f} м"]
            if p.get("calibration"):
                parts.append("с калибровкой")
            mark = "✔ " if name == self.cfg.active_profile else "    "
            self.prof_list.addItem(f"{mark}{name}  —  " + ", ".join(parts))
            it = self.prof_list.item(self.prof_list.count() - 1)
            it.setData(Qt.UserRole, name)
            if name == cur:
                self.prof_list.setCurrentItem(it)

    def _prof_selected(self) -> Optional[str]:
        lst = getattr(self, "prof_list", None)
        it = lst.currentItem() if lst is not None else None
        return it.data(Qt.UserRole) if it is not None else None

    def _prof_changed(self) -> None:
        self._prof_fill()
        self.changed.emit(["profiles"])

    def _prof_new(self) -> None:
        name, ok = QInputDialog.getText(self, "Новый профиль", "Название профиля:")
        if not ok or not name.strip():
            return
        if profiles.clean_profile_name(name) in self.cfg.profiles and QMessageBox.question(
                self, "Профили", "Профиль с таким именем уже есть. Перезаписать?") \
                != QMessageBox.Yes:
            return
        try:
            profiles.save_profile(self.cfg, name)
        except ValueError as e:
            QMessageBox.warning(self, "Профили", str(e))
            return
        self._prof_changed()

    def _prof_update(self) -> None:
        name = self._prof_selected()
        if name:
            profiles.save_profile(self.cfg, name)
            self._prof_changed()

    def _prof_apply(self) -> None:
        name = self._prof_selected()
        if name:
            self.action.emit("profile_apply:" + name)

    def _prof_rename(self) -> None:
        name = self._prof_selected()
        if not name:
            return
        new, ok = QInputDialog.getText(self, "Переименовать профиль", "Новое название:",
                                       text=name)
        if ok and new.strip():
            try:
                profiles.rename_profile(self.cfg, name, new)
            except ValueError as e:
                QMessageBox.warning(self, "Профили", str(e))
                return
            self._prof_changed()

    def _prof_delete(self) -> None:
        name = self._prof_selected()
        if name and QMessageBox.question(self, "Профили", f"Удалить профиль «{name}»?") \
                == QMessageBox.Yes:
            profiles.delete_profile(self.cfg, name)
            self._prof_changed()

    def _page_recordings(self):
        w, lay = self._page("Запись и повтор", "Кнопка «Запись» в главном окне записывает "
                                               "ваши движения — только цифры позы (33 точки "
                                               "тела), без видео и без других людей. Записи "
                                               "хранятся только на этом ПК.")
        f = self._group(lay, "Записи")
        self.rec_list = QListWidget()
        self.rec_list.setMinimumHeight(140)
        f.addRow(self.rec_list)
        row = QHBoxLayout()
        for text, act, tip in [
                ("Подобрать сглаживание", "tune",
                 "Перебирает настройки сглаживания на вашей записи и предлагает лучшие — "
                 "только если они измеримо лучше текущих (меньше дрожание стоя и не больше "
                 "ошибка в движении)."),
                ("Воспроизвести", "replay",
                 "Запустить трекинг по записи вместо камеры: всё (сглаживание, калибровка, "
                 "вывод в игру) работает как вживую — удобно проверять настройки."),
                ("Удалить", "rec_delete", "")]:
            b = QPushButton(text)
            if tip:
                b.setToolTip(tip)
            if act == "rec_delete":
                b.setObjectName("danger")
            b.clicked.connect(lambda _c=False, a=act: self._rec_act(a))
            row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)
        b = QPushButton("Вернуть камеру вместо записи")
        b.clicked.connect(lambda: self.action.emit("replay_stop"))
        f.addRow(b)
        self._hint(f, "Для подбора сглаживания запишите 1–3 минуты: постойте спокойно секунд "
                      "20, потом походите, присядьте, подвигайте ногами. Подбор занимает около "
                      "минуты и ничего не меняет без вашего согласия.")
        lay.addStretch(1)
        return w

    def _rec_fill(self) -> None:
        from ..recording import list_recordings
        self.rec_list.clear()
        if not self._data_dir:
            return
        for p in list_recordings(Path(self._data_dir))[:200]:
            try:
                kb = p.stat().st_size / 1024
            except OSError:
                continue
            playing = " ▶ воспроизводится" if p.resolve() == Path(self.cfg.camera).resolve() \
                and self.cfg.camera.lower().endswith(".npz") else ""
            self.rec_list.addItem(f"{p.stem}  ({kb:.0f} КБ){playing}")
            self.rec_list.item(self.rec_list.count() - 1).setData(Qt.UserRole, str(p))
        if self.rec_list.count():
            self.rec_list.setCurrentRow(0)

    def _rec_act(self, act: str) -> None:
        it = self.rec_list.currentItem()
        if it is None:
            QMessageBox.information(self, "Записи", "Сначала сделайте запись: кнопка "
                                                    "«Запись» в главном окне.")
            return
        self.action.emit(f"{act}:{it.data(Qt.UserRole)}")

    def _page_privacy(self):
        w, lay = self._page("Конфиденциальность и данные")
        info = QLabel(
            "• Видео с камеры обрабатывается только на этом ПК и нигде не сохраняется.<br>"
            "• Координаты трекеров отправляются только в программы на этом же ПК "
            "(127.0.0.1) — в сеть они не уходят вообще.<br>"
            "• Распознавание людей в кадре (кто есть кто) делается в памяти по цвету одежды "
            "и пропорциям и нигде не сохраняется.<br>"
            "• Программа ничего не принимает из сети и не отправляет статистику. Принимаются "
            "только данные от программ на этом ПК (127.0.0.1): поза шлема от драйвера Nexovara Pose и, "
            "если включено, от VRChat.<br>"
            "• «Поза шлема из VRChat» (выключено по умолчанию): Nexovara Pose объявляет себя в "
            "локальной сети по mDNS — только имя «Nexovara Pose-xxxx», порты и адрес 127.0.0.1, "
            "чтобы VRChat на этом ПК его нашёл.<br>"
            "• Записи движений (кнопка «Запись») — только цифры позы того, за кем следит "
            "программа, без видео; хранятся в папке данных, удаляются кнопкой ниже.<br>"
            "• Профили хранятся в config.json вместе с остальными настройками.<br>"
            "• Автонастройка сохраняет только цифры (потоки CPU, объём ОЗУ, время "
            "обработки кадра), без названий устройств и серийных номеров.<br>"
            "• Список процессов проверяется только на имена SteamVR/VRChat/ChilloutVR/"
            "Resonite и никуда не записывается.<br>"
            "• Из интернета скачиваются только модели нейросетей с официальных адресов: Google "
            "(MediaPipe) и, если вы включили RTMPose, OpenMMLab (download.openmmlab.com), по "
            "HTTPS; их контрольные суммы запоминаются и проверяются при каждом запуске.")
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        lay.addWidget(info)
        f = self._group(lay, "Папка данных")
        self.data_path = QLabel()
        self.data_path.setObjectName("muted")
        self.data_path.setWordWrap(True)
        self.data_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        f.addRow(self.data_path)
        row = QHBoxLayout()
        for text, act, name in [("Открыть папку", "open_data", ""),
                                ("Очистить журнал", "clear_logs", ""),
                                ("Удалить все мои данные…", "delete_data", "danger")]:
            b = QPushButton(text)
            if name:
                b.setObjectName(name)
            b.clicked.connect(lambda _c=False, a=act: self.action.emit(a))
            row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------ state
    def _update_marks(self) -> None:
        for key, lab in self._labels.items():
            base = lab.property("base") or ""
            if key in self.cfg.user_set:
                lab.setText(f"<span style='color:{theme.ACCENT}'>{MARK}</span> {base}")
                lab.setToolTip("Изменено вами — автонастройка не трогает")
            else:
                lab.setText(base)
                lab.setToolTip("")
        if "width" in self._labels:  # resolution row also covers height
            if "width" in self.cfg.user_set or "height" in self.cfg.user_set:
                self._labels["width"].setText(f"<span style='color:{theme.ACCENT}'>{MARK}</span> "
                                              "Разрешение и FPS")
        tp = PRESETS.get(self.cfg.target)
        self.target_desc.setText(tp.description if tp else
                                 "Программа выберет режим сама при запуске трекинга.")

    def refresh(self, extra: Optional[dict] = None) -> None:
        """Reload all widgets from the config (no change signals)."""
        self._loading = True
        try:
            for key, (_w, _l, setter) in self._fields.items():
                setter(getattr(self.cfg, key))
            for n, cb in self._tracker_boxes.items():
                cb.setChecked(bool(self.cfg.trackers.get(n)))
            self._fill_cam_combo(self.cam_combo, self.cfg.camera, self.cfg.camera_name)
            self._fill_modes()
            self._extra_rebuild()
            self._prof_fill()
            self.refiner_note.setText(self._refiner_text())
            c = self.cfg.calibration
            if c:
                self.calib_state.setText(
                    f"<span style='color:{theme.GOOD}'>✔ Откалибровано</span> · масштаб "
                    f"{c.get('scale', 1):.2f}" + (" · привязка к шлему" if c.get("hmd_yaw")
                                                   is not None else ""))
            else:
                self.calib_state.setText(f"<span style='color:{theme.WARN}'>Не откалибровано</span>")
            pc = self.cfg.pc_profile
            if pc and pc.get("bench"):
                parts = [f"{r['variant']}: {r['ms']:.0f} мс" for r in pc["bench"]]
                self.perf_note.setText("Замер на вашем ПК: " + ", ".join(parts))
            else:
                self.perf_note.setText("Замер ещё не проводился — запустите «Автонастройку».")
            if extra:
                if "steamvr" in extra:
                    self.steamvr_state.setText(extra["steamvr"])
                if "driver" in extra:
                    self.driver_state.setText(extra["driver"])
                if "data_dir" in extra:
                    self.data_path.setText(extra["data_dir"])
                    self._data_dir = extra["data_dir"]
            self._rec_fill()
            self._update_marks()
        finally:
            self._loading = False
