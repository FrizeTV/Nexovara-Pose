"""Custom widgets: camera view with overlay, 3D skeleton view, status cards."""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..hands import HAND_CONNECTIONS
from ..mathutil import rot_x, rot_y
from ..skeleton import CONNECTIONS, Tracker
from . import theme

BONE_OK = QColor("#7ee2a8")
BONE_LOW = QColor("#e07a7a")
HAND_COL = QColor("#ffd166")


OTHER_COL = QColor(150, 155, 165, 150)
STATUS_COL = {"measured": QColor("#ffb347"), "estimated": QColor("#c08bff"),
              "held": QColor("#e07a7a")}


class VideoView(QWidget):
    """Shows the camera frame (RGB numpy) with the detected skeleton on top.
    Other people are drawn in grey; clicking a person emits `clicked(x, y)`
    with normalized image coordinates."""

    clicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
        self._others = []
        self._rect = None
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._frame: Optional[np.ndarray] = None
        self._img: Optional[QImage] = None
        self._lm = None
        self._vis = None
        self._hands = {}
        self.vis_threshold = 0.5
        self.overlay_text = ""
        self.placeholder = "Камера выключена\nНажмите «Старт» (F5)"

    def set_data(self, frame, lm, vis, hands, vis_threshold: float, others=None) -> None:
        self._others = others or []
        if frame is not None:
            frame = np.ascontiguousarray(frame)
            self._frame = frame  # keep the buffer alive while QImage uses it
            h, w = frame.shape[:2]
            self._img = QImage(frame.data, w, h, int(frame.strides[0]), QImage.Format_RGB888)
        self._lm, self._vis, self._hands = lm, vis, hands or {}
        self.vis_threshold = vis_threshold
        self.update()

    def clear(self, placeholder: Optional[str] = None) -> None:
        self._frame = self._img = self._lm = self._vis = None
        self._hands = {}
        self._others = []
        if placeholder is not None:
            self.placeholder = placeholder
        self.update()

    def paintEvent(self, _ev) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0d0f13"))
        if self._img is None:
            p.setPen(QColor(theme.MUTED))
            f = QFont(self.font())
            f.setPointSizeF(12)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            p.end()
            return
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        tw, th = iw * s, ih * s
        ox, oy = (self.width() - tw) / 2, (self.height() - th) / 2
        self._rect = (ox, oy, tw, th)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.drawImage(QRectF(ox, oy, tw, th), self._img)

        def pt(xy):
            return QPointF(ox + xy[0] * tw, oy + xy[1] * th)
        for other in self._others:   # people we ignore
            p.setPen(QPen(OTHER_COL, 2, Qt.DashLine, Qt.RoundCap))
            for a, b in CONNECTIONS:
                p.drawLine(pt(other[a]), pt(other[b]))
        if self._lm is not None and self._vis is not None:
            thr = self.vis_threshold
            for a, b in CONNECTIONS:
                ok = self._vis[a] >= thr and self._vis[b] >= thr
                p.setPen(QPen(BONE_OK if ok else BONE_LOW, 3, Qt.SolidLine, Qt.RoundCap))
                p.drawLine(pt(self._lm[a]), pt(self._lm[b]))
            p.setPen(Qt.NoPen)
            for i in range(11, 33):
                p.setBrush(QColor("#ffffff") if self._vis[i] >= thr else BONE_LOW)
                p.drawEllipse(pt(self._lm[i]), 3.5, 3.5)
        for hand in self._hands.values():
            img = hand.get("image")
            if img is None:
                continue
            p.setPen(QPen(HAND_COL, 2, Qt.SolidLine, Qt.RoundCap))
            for a, b in HAND_CONNECTIONS:
                p.drawLine(pt(img[a]), pt(img[b]))
        if self._others:
            f = QFont(self.font())
            f.setPointSizeF(9.5)
            p.setFont(f)
            p.setPen(QColor("#c8d0dc"))
            p.drawText(QRectF(ox + 10, oy + th - 28, tw - 20, 22), Qt.AlignLeft,
                       f"Другие люди в кадре: {len(self._others)} — игнорируются "
                       "(клик по человеку — следить за ним)")
        if self.overlay_text:
            f = QFont(self.font())
            f.setPointSizeF(54)
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("#ffd166"))
            p.drawText(self.rect(), Qt.AlignCenter, self.overlay_text)
        p.end()


    def mousePressEvent(self, e):
        if self._rect is None or e.button() != Qt.LeftButton:
            return
        ox, oy, tw, th = self._rect
        x = (e.position().x() - ox) / tw
        y = (e.position().y() - oy) / th
        if 0 <= x <= 1 and 0 <= y <= 1:
            self.clicked.emit(float(x), float(y))


class Skeleton3DView(QWidget):
    """Orbit view of the calibrated skeleton and the trackers' axes.
    Left drag — rotate, wheel — zoom, double click — reset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.reset_view()
        self._drag = None
        self._pts = None
        self._trackers: List[Tracker] = []
        self._head = None
        self._hands = []
        self._statuses = {}
        self.message = ""

    def reset_view(self) -> None:
        self.yaw, self.pitch, self.zoom = 200.0, -12.0, 1.0
        self.update()

    def set_data(self, pts, trackers, head, hands=None, message: str = "",
                 statuses=None) -> None:
        self._statuses = statuses or {}
        self._pts, self._trackers, self._head = pts, trackers or [], head
        self._hands = hands or []
        self.message = message
        self.update()

    def mousePressEvent(self, e):
        self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            d = e.position() - self._drag
            self.yaw = (self.yaw + d.x() * 0.5) % 360
            self.pitch = float(np.clip(self.pitch - d.y() * 0.4, -89, 89))
            self._drag = e.position()
            self.update()

    def mouseReleaseEvent(self, _e):
        self._drag = None

    def mouseDoubleClickEvent(self, _e):
        self.reset_view()

    def wheelEvent(self, e):
        f = 1.1 if e.angleDelta().y() > 0 else 1 / 1.1
        self.zoom = float(np.clip(self.zoom * f, 0.3, 5.0))
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#101318"))
        w, h = self.width(), self.height()
        V = rot_x(self.pitch) @ rot_y(-self.yaw)
        scale = min(w, h) / 2.4 * self.zoom
        center = np.array([0.0, 0.9, 0.0])

        def proj(pnt):
            q = (np.asarray(pnt, dtype=np.float64) - center) @ V.T
            return QPointF(w / 2 + q[0] * scale, h / 2 - q[1] * scale)
        p.setPen(QPen(QColor("#252a35"), 1))
        for v in np.arange(-1.5, 1.51, 0.5):
            p.drawLine(proj((v, 0, -1.5)), proj((v, 0, 1.5)))
            p.drawLine(proj((-1.5, 0, v)), proj((1.5, 0, v)))
        p.setPen(QPen(QColor("#3d4658"), 2))
        p.drawLine(proj((0, 0, 0)), proj((0, 0, 0.6)))  # where the user faces
        if self._pts is not None:
            P = self._pts
            p.setPen(QPen(QColor("#a9bddb"), 2.4, Qt.SolidLine, Qt.RoundCap))
            for a, b in CONNECTIONS:
                p.drawLine(proj(P[a]), proj(P[b]))
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#e6edf7"))
            for i in range(11, 33):
                p.drawEllipse(proj(P[i]), 2.5, 2.5)
        axes = [QColor("#ff6b6b"), QColor("#5ee08a"), QColor("#62a8ff")]
        for t in [*self._trackers, *self._hands, *([self._head] if self._head else [])]:
            c = proj(t.pos)
            col = QColor(theme.ACCENT) if t.name == "head" else (
                HAND_COL if t.name.endswith("hand")
                else STATUS_COL.get(self._statuses.get(t.name, "measured"), STATUS_COL["measured"]))
            p.setPen(QPen(col, 2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(c, 7, 7)
            if t.rot is not None:
                for k in range(3):
                    p.setPen(QPen(axes[k], 2))
                    p.drawLine(c, proj(t.pos + t.rot[:, k] * 0.12))
        p.setPen(QColor(theme.MUTED))
        if self.message:
            f = QFont(self.font())
            f.setPointSizeF(10.5)
            p.setFont(f)
            p.drawText(QRectF(10, 8, w - 20, 24), Qt.AlignLeft, self.message)
        f = QFont(self.font())
        f.setPointSizeF(8)
        p.setFont(f)
        p.setPen(QColor("#5b6475"))
        p.drawText(QRectF(10, h - 22, w - 20, 18), Qt.AlignLeft,
                   "ЛКМ — вращать · колесо — масштаб · двойной клик — сброс")
        x = w - 10
        for label, key in (("удержан", "held"), ("оценка", "estimated"), ("камера", "measured")):
            tw_ = p.fontMetrics().horizontalAdvance(label)
            x -= tw_
            p.setPen(STATUS_COL[key])
            p.drawText(QPointF(x, h - 9), label)
            p.setBrush(STATUS_COL[key])
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x - 8, h - 13), 4, 4)
            x -= 22
        p.end()


class StatCard(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 9, 12, 9)
        lay.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName("cardTitle")
        self.value = QLabel("—")
        self.value.setObjectName("cardValue")
        self.sub = QLabel("")
        self.sub.setObjectName("cardSub")
        self.sub.setWordWrap(True)
        for wdg in (self.title, self.value, self.sub):
            lay.addWidget(wdg)

    def set(self, value: str, sub: str = "", color: Optional[str] = None) -> None:
        self.value.setText(value)
        self.value.setStyleSheet(f"color: {color};" if color else "")
        if self.sub.text() != sub:
            self.sub.setText(sub)
            # never squeezed below its text (the 3D view above gives way instead)
            self.setMinimumHeight(self.layout().sizeHint().height())


class ConfidenceBars(QWidget):
    """How sure the program is about each tracker: one thin bar per tracker
    (green — sure, yellow — so-so, red — guessed or out of view)."""

    ROW_H = 17

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: List[tuple] = []
        self.setMinimumHeight(self.ROW_H)

    def _cols(self) -> int:
        return int(min(4, max(2, self.width() // 165)))

    def _fit(self) -> None:
        rows = -(-len(self._items) // self._cols()) if self._items else 1
        self.setMinimumHeight(max(1, rows) * self.ROW_H + 2)

    def resizeEvent(self, e) -> None:
        self._fit()
        super().resizeEvent(e)

    def set_values(self, items: List[tuple]) -> None:
        """items: [(label, value 0..1 or None)]"""
        if items != self._items:
            self._items = list(items)
            self._fit()
            self.update()

    @staticmethod
    def color(v: Optional[float]) -> QColor:
        if v is None:
            return QColor(theme.MUTED)
        return QColor(theme.GOOD) if v >= 0.65 else (QColor(theme.WARN) if v >= 0.35
                                                     else QColor(theme.BAD))

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        f = QFont(self.font())
        f.setPointSizeF(8.5)
        p.setFont(f)
        w = self.width()
        cols = self._cols()
        col_w = (w - 8 * (cols - 1)) / cols
        label_w = min(100.0, col_w * 0.6)
        for i, (label, v) in enumerate(self._items):
            x = (i % cols) * (col_w + 8)
            y = (i // cols) * self.ROW_H
            p.setPen(QColor(theme.MUTED))
            p.drawText(QRectF(x, y, label_w - 4, self.ROW_H), Qt.AlignVCenter | Qt.AlignLeft,
                       label)
            bx, bw = x + label_w, col_w - label_w - 4
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(theme.PANEL_2))
            p.drawRoundedRect(QRectF(bx, y + 5, bw, self.ROW_H - 10), 3, 3)
            if v is not None:
                p.setBrush(self.color(v))
                p.drawRoundedRect(QRectF(bx, y + 5, max(3.0, bw * float(np.clip(v, 0, 1))),
                                         self.ROW_H - 10), 3, 3)
        p.end()


class FingerBars(QWidget):
    """Five small bars per hand showing finger curl."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(46)
        self._curls = {}

    def set_curls(self, curls: dict) -> None:
        self._curls = curls
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        half = w / 2
        for k, side in enumerate(("left", "right")):
            c = self._curls.get(side)
            x0 = k * half + 4
            bw = (half - 16) / 5
            for i in range(5):
                x = x0 + i * bw
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(theme.PANEL_2))
                p.drawRoundedRect(QRectF(x, 2, bw - 4, h - 16), 3, 3)
                if c is not None:
                    v = float(np.clip(c[i], 0, 1))
                    p.setBrush(HAND_COL)
                    hh = (h - 16) * v
                    p.drawRoundedRect(QRectF(x, 2 + (h - 16) - hh, bw - 4, hh), 3, 3)
            p.setPen(QColor(theme.MUTED))
            f = QFont(self.font())
            f.setPointSizeF(8)
            p.setFont(f)
            p.drawText(QRectF(x0, h - 13, half - 12, 12), Qt.AlignCenter,
                       "левая" if side == "left" else "правая")
        p.end()
