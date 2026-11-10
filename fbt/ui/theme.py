"""Dark theme: palette + Qt style sheet."""
from __future__ import annotations

import tempfile
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

BG = "#14161b"
PANEL = "#1c1f26"
PANEL_2 = "#232731"
BORDER = "#2e3340"
TEXT = "#e6e9ef"
MUTED = "#8b93a5"
ACCENT = "#6d8bff"
ACCENT_2 = "#5a76e8"
GOOD = "#4ade80"
WARN = "#fbbf24"
BAD = "#f87171"

QSS = f"""
* {{ font-size: 10pt; }}
QWidget {{ background: {BG}; color: {TEXT}; }}
QLabel {{ background: transparent; }}
QGroupBox QWidget, QFrame#card QWidget {{ background: transparent; }}
QGroupBox QLineEdit, QGroupBox QSpinBox, QGroupBox QDoubleSpinBox, QGroupBox QComboBox {{
    background: {PANEL_2}; }}
QGroupBox QPushButton {{ background: {PANEL_2}; }}
QGroupBox QPushButton#danger {{ background: #4a2326; }}
QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; }}
QFrame#header {{ background: {PANEL}; border-bottom: 1px solid {BORDER}; }}
QLabel#appTitle {{ font-size: 15pt; font-weight: 700; background: transparent; }}
QLabel#chip {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 11px;
              padding: 3px 10px; color: {TEXT}; }}
QLabel#hint {{ background: #1e2638; border: 1px solid #2f3d63; border-radius: 10px;
              padding: 10px 14px; color: #c9d6ff; font-size: 10.5pt; }}
QLabel#muted, QLabel.muted {{ color: {MUTED}; background: transparent; }}
QLabel#pageTitle {{ font-size: 14pt; font-weight: 700; }}
QLabel#sectionTitle {{ font-size: 11pt; font-weight: 600; color: {TEXT}; margin-top: 6px; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 12px; }}
QLabel#cardTitle {{ color: {MUTED}; font-size: 9pt; background: transparent; }}
QLabel#cardValue {{ font-size: 12.5pt; font-weight: 600; background: transparent; }}
QLabel#cardSub {{ color: {MUTED}; font-size: 8.5pt; background: transparent; }}
QPushButton {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 8px;
              padding: 7px 14px; }}
QPushButton:hover {{ background: #2b303c; }}
QPushButton:pressed {{ background: #333948; }}
QPushButton:disabled {{ color: #5d6475; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: white; font-weight: 600;
                      padding: 8px 20px; }}
QPushButton#primary:hover {{ background: {ACCENT_2}; }}
QPushButton#danger {{ background: #4a2326; border: 1px solid #6b2f34; }}
QPushButton#flat {{ background: transparent; border: none; color: {MUTED}; padding: 4px 8px; }}
QPushButton#flat:hover {{ color: {TEXT}; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {ACCENT}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {BORDER};
    selection-background-color: {ACCENT}; }}
QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px;
    border: 1px solid #5a6275; background: #2b303c; }}
QCheckBox::indicator {{ border-radius: 4px; }}
QRadioButton::indicator {{ border-radius: 9px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT};
    image: url(@CHECK@); }}
QRadioButton::indicator:checked {{ border-color: {ACCENT};
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {ACCENT}, stop:0.45 {ACCENT}, stop:0.55 #2b303c, stop:1 #2b303c); }}
QTableWidget::indicator {{ width: 16px; height: 16px; border: 1px solid #5a6275;
    border-radius: 4px; background: #2b303c; }}
QTableWidget::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT};
    image: url(@CHECK@); }}
QSpinBox, QDoubleSpinBox {{ padding-right: 18px; }}
QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::down-button {{ width: 16px; border: none; background: transparent; }}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url(@UP@); width: 9px; height: 9px; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow,
QComboBox::down-arrow {{ image: url(@DOWN@); width: 9px; height: 9px; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QListWidget#nav {{ background: {PANEL}; border: none; border-right: 1px solid {BORDER};
                  padding: 8px; outline: none; }}
QListWidget#nav::item {{ padding: 9px 12px; border-radius: 8px; margin: 1px 0; }}
QListWidget#nav::item:selected {{ background: #283150; color: white; }}
QListWidget#nav::item:hover:!selected {{ background: {PANEL_2}; }}
QScrollArea {{ border: none; }}
QGroupBox {{ border: 1px solid {BORDER}; border-radius: 10px; margin-top: 14px; padding: 12px;
            background: {PANEL}; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; color: {MUTED}; }}
QProgressBar {{ background: {PANEL_2}; border: 1px solid {BORDER}; border-radius: 6px;
               text-align: center; height: 18px; }}
QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}
QTableWidget {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
               gridline-color: {BORDER}; }}
QHeaderView::section {{ background: {PANEL_2}; color: {MUTED}; border: none; padding: 6px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; border-top: 1px solid {BORDER}; }}
QSplitter::handle {{ background: {BG}; width: 6px; }}
QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
"""


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    for role, col in [(QPalette.Window, BG), (QPalette.WindowText, TEXT), (QPalette.Base, "#2b303c"),
                      (QPalette.AlternateBase, PANEL), (QPalette.Text, TEXT),
                      (QPalette.Button, PANEL_2), (QPalette.ButtonText, TEXT),
                      (QPalette.Highlight, ACCENT), (QPalette.HighlightedText, "#ffffff"),
                      (QPalette.ToolTipBase, PANEL_2), (QPalette.ToolTipText, TEXT),
                      (QPalette.PlaceholderText, MUTED)]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    icons = _icons()
    app.setStyleSheet(QSS.replace("@CHECK@", icons["check"]).replace("@UP@", icons["up"])
                      .replace("@DOWN@", icons["down"]))


def _icons() -> dict:
    """Draw small UI icons once (no image plugins or resource files needed)."""
    d = Path(tempfile.gettempdir()) / "nexovarapose-ui-v2"
    d.mkdir(exist_ok=True)
    out = {}
    for name in ("check", "up", "down"):
        f = d / f"{name}.png"
        out[name] = f.as_posix()
        if f.is_file():
            continue
        pm = QPixmap(32, 32)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        if name == "check":
            p.setPen(QPen(QColor("#ffffff"), 4.5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            path = QPainterPath(QPointF(7, 17))
            path.lineTo(13.5, 23)
            path.lineTo(25, 9)
            p.drawPath(path)
        else:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(MUTED))
            ys = (22, 10) if name == "up" else (10, 22)
            path = QPainterPath(QPointF(6, ys[0]))
            path.lineTo(26, ys[0])
            path.lineTo(16, ys[1])
            path.closeSubpath()
            p.drawPath(path)
        p.end()
        pm.save(str(f), "PNG")
    return out
