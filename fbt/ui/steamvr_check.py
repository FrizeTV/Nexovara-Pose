"""'SteamVR check' dialog: step by step, where the chain
Nexovara Pose -> driver -> SteamVR -> game breaks, plus one button that repairs what
can be repaired (SteamVR closed) and the Nexovara Pose lines from SteamVR's own log."""
from __future__ import annotations

import logging
from typing import Callable, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
                               QScrollArea, QVBoxLayout, QWidget)

from .. import steamvr
from ..config import Config
from . import theme

log = logging.getLogger(__name__)


def _window(cfg: Config):
    w, h = cfg.nohmd_window.split("x")
    return int(w), int(h)


class SteamVRCheckDialog(QDialog):
    def __init__(self, cfg: Config, app_state: Callable[[], List[steamvr.Check]],
                 on_fixed: Optional[Callable[[], None]] = None,
                 on_roles: Optional[Callable[[], None]] = None, parent=None):
        """app_state(): checks about Nexovara Pose itself (mode, tracking running, packets/s)."""
        super().__init__(parent)
        self.cfg = cfg
        self._app_state = app_state
        self._on_fixed = on_fixed
        self._on_roles = on_roles
        self.setWindowTitle("Проверка SteamVR — Nexovara Pose")
        self.resize(820, 640)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 14)
        t = QLabel("Проверка SteamVR и драйвера Nexovara Pose")
        t.setObjectName("pageTitle")
        lay.addWidget(t)
        intro = QLabel("Цепочка: Nexovara Pose → драйвер Nexovara Pose → SteamVR → игра. Первый пункт с ✖ — "
                       "то, что мешает. «Исправить» установит/обновит драйвер, включит его и "
                       "настроит режим без шлема (SteamVR должен быть закрыт).")
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        lay.addWidget(self.area, 3)
        cap = QLabel("Строки Nexovara Pose из журнала SteamVR (vrserver.txt):")
        cap.setObjectName("muted")
        lay.addWidget(cap)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(400)
        lay.addWidget(self.log_view, 2)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        row = QHBoxLayout()
        b_again = QPushButton("Проверить снова")
        b_again.clicked.connect(self.refresh)
        self.b_fix = QPushButton("Исправить")
        self.b_fix.setObjectName("primary")
        self.b_fix.clicked.connect(self._fix)
        b_close = QPushButton("Закрыть")
        b_close.clicked.connect(self.close)
        row.addWidget(b_again)
        if on_roles is not None:
            b_roles = QPushButton("Назначить роли трекерам…")
            b_roles.clicked.connect(lambda: (self._on_roles(), self.refresh()))
            row.addWidget(b_roles)
        row.addStretch(1)
        row.addWidget(self.b_fix)
        row.addWidget(b_close)
        lay.addLayout(row)
        self.refresh()

    def _checks(self) -> List[steamvr.Check]:
        nohmd = self.cfg.target == "steamvr_nohmd" or self.cfg.driver_hmd
        try:
            checks = steamvr.diagnose(nohmd)
        except Exception as e:
            log.exception("SteamVR diagnose failed")
            checks = [steamvr.Check("Проверка SteamVR", False, str(e))]
        try:
            checks += self._app_state()
        except Exception:
            log.exception("app state check failed")
        return checks

    def refresh(self) -> None:
        holder = QWidget()                 # a fresh list every time (the old one is deleted)
        self.rows = QVBoxLayout(holder)
        first_bad = True
        for c in self._checks():
            icon, col = (("✔", theme.GOOD) if c.ok else ("✖", theme.BAD) if c.ok is False
                         else ("•", theme.MUTED))
            lab = QLabel(f"<span style='color:{col}; font-weight:600'>{icon}</span>&nbsp; "
                         f"<b>{c.title}</b>" + (f" — {_esc(c.text)}" if c.text else ""))
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            if c.ok is False and first_bad:
                lab.setStyleSheet("background: rgba(248, 113, 113, 0.14); border-radius: 6px; "
                                  "padding: 4px;")
                first_bad = False
            self.rows.addWidget(lab)
        self.rows.addStretch(1)
        self.area.setWidget(holder)
        lines = steamvr.driver_log_lines(80)
        self.log_view.setPlainText("\n".join(lines) if lines else
                                   "Нет строк Nexovara Pose. Если SteamVR уже запускался после установки "
                                   "драйвера — значит SteamVR его не загрузил: нажмите "
                                   "«Исправить» при закрытом SteamVR.")
        self.log_view.verticalScrollBar().setValue(self.log_view.verticalScrollBar().maximum())

    def _fix(self) -> None:
        nohmd = self.cfg.target == "steamvr_nohmd" or self.cfg.driver_hmd
        try:
            done = steamvr.fix_all(nohmd, _window(self.cfg), self.cfg.nohmd_fov)
        except Exception as e:
            self.result.setText(f"<span style='color:{theme.BAD}'>Не получилось: {_esc(str(e))}"
                                "</span>")
            self.refresh()
            return
        if self._on_fixed:
            self._on_fixed()
        self.result.setText(f"<span style='color:{theme.GOOD}'>"
                            + ("Готово: " + "; ".join(done) if done else "Чинить нечего.")
                            + "</span> Теперь запустите SteamVR, затем «Старт» в Nexovara Pose и "
                              "нажмите «Проверить снова».")
        self.refresh()


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
