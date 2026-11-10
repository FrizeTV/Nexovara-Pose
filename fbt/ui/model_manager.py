"""Download selected models before tracking and keep them in the project."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..model_catalog import EXTRA_MODELS, available_options
from ..onnx_pose import ensure_extra, find_extra
from ..pose import ModelManifest, ensure_model, find_model
from ..rtmpose import onnxruntime_available


class _Worker(QObject):
    progress = Signal(str, float)
    done = Signal(dict)

    def __init__(self, selected, models_dir, search_dirs, manifest):
        super().__init__()
        self.selected = selected
        self.models_dir = Path(models_dir)
        self.search_dirs = list(search_dirs)
        self.manifest = manifest

    def run(self):
        errors = {}
        for key in self.selected:
            try:
                self.progress.emit(key, 0.0)

                def report(fraction):
                    self.progress.emit(key, fraction)

                ensure = ensure_extra if key in EXTRA_MODELS else ensure_model
                ensure(key, self.models_dir, self.search_dirs, report, self.manifest)
                self.progress.emit(key, 1.0)
            except Exception as exc:
                errors[key] = str(exc)
        self.done.emit(errors)


class ModelDownloadDialog(QDialog):
    def __init__(self, models_dir, search_dirs, manifest: ModelManifest, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Модели нейросети — Nexovara Pose")
        self.resize(655, 500)
        self._models_dir, self._search_dirs, self._manifest = models_dir, search_dirs, manifest
        self._boxes = {}
        self._bars = {}
        self._worker = None
        self._busy = False
        lay = QVBoxLayout(self)
        intro = QLabel("Выберите модели для загрузки. Сохранённые файлы используются повторно. "
                       "Чтобы включить ансамбль, откройте «Настройки → Производительность».")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        if not onnxruntime_available():
            runtime = QLabel("Для запуска дополнительных моделей установите ONNX Runtime "
                             "через setup.bat или setup.sh. Скачать веса можно заранее.")
            runtime.setWordWrap(True)
            lay.addWidget(runtime)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        form = QFormLayout(content)
        dirs = [*search_dirs, models_dir]
        for option in available_options():
            key = option.id
            extra = key in EXTRA_MODELS
            ready = find_extra(key, dirs) if extra else find_model(key, dirs)
            box = QCheckBox(option.title)
            box.setChecked(not ready and not extra and key in ("lite", "full", "hand"))
            bar = QProgressBar()
            bar.setValue(100 if ready else 0)
            self._boxes[key], self._bars[key] = box, bar
            form.addRow(box, bar)
        scroll.setWidget(content)
        lay.addWidget(scroll, 1)
        self.status = QLabel("")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        self.button = QPushButton("Скачать выбранные")
        self.button.clicked.connect(self.start_download)
        self.skip = QPushButton("Закрыть")
        self.skip.clicked.connect(self.reject)
        row.addWidget(self.button)
        row.addWidget(self.skip)
        lay.addLayout(row)

    def start_download(self):
        if self._busy:
            return
        selected = [key for key, box in self._boxes.items() if box.isChecked()]
        if not selected:
            self.status.setText("Отметьте нужные модели.")
            return
        self._busy = True
        self.button.setEnabled(False)
        self.skip.setEnabled(False)
        for box in self._boxes.values():
            box.setEnabled(False)
        self._worker = _Worker(selected, self._models_dir, self._search_dirs, self._manifest)
        self._worker.progress.connect(self._progress, Qt.QueuedConnection)
        self._worker.done.connect(self._finished, Qt.QueuedConnection)
        threading.Thread(target=self._worker.run, name="model-download", daemon=True).start()

    def reject(self):
        if not self._busy:
            super().reject()

    @Slot(str, float)
    def _progress(self, key, fraction):
        self._bars[key].setValue(int(100 * max(0, min(1, fraction))))
        self.status.setText(f"{'Готово' if fraction >= 1 else 'Проверка и загрузка'} · {key}")

    @Slot(dict)
    def _finished(self, errors):
        self._busy = False
        self.button.setEnabled(True)
        self.skip.setEnabled(True)
        for key, box in self._boxes.items():
            box.setEnabled(True)
            if self._bars[key].value() == 100 and key not in errors:
                box.setChecked(False)
        if errors:
            for key in errors:
                self._bars[key].setValue(0)
            self.status.setText("Не все модели загружены. Можно повторить загрузку.")
            msg = QMessageBox(QMessageBox.Warning, "Загрузка моделей",
                              "\n".join(f"{key}: {err}" for key, err in errors.items()),
                              QMessageBox.Ok, self)
            msg.setTextFormat(Qt.PlainText)
            msg.exec()
        else:
            self.status.setText("Модели готовы. Теперь их можно выбрать в настройках.")
