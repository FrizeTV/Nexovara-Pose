"""Nexovara Pose entry point: `python -I main.py` (run.bat does this)."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# -I (isolated mode) does not put the script folder on sys.path -> add it explicitly
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
# Media Foundation: opening a webcam is much faster without HW transforms
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")


def _setup_logging(log_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fh = logging.handlers.RotatingFileHandler(log_dir / "nexovara_pose.log", maxBytes=1_000_000,
                                              backupCount=2, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stderr is not None:  # pythonw.exe has no console
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


_crash_file = None


def _crash_reports(data_dir: Path, log: logging.Logger) -> None:
    """If the program ever dies inside native code (a camera driver, a GPU
    driver), Python's faulthandler writes where it happened to crash.log in
    the data folder; errors in background threads go to nexovara_pose.log."""
    global _crash_file
    try:
        import faulthandler
        _crash_file = open(data_dir / "crash.log", "a", encoding="utf-8")
        _crash_file.write(f"\n--- Nexovara Pose started {__import__('time').strftime('%Y-%m-%d %H:%M:%S')}\n")
        _crash_file.flush()
        faulthandler.enable(_crash_file, all_threads=True)
    except Exception as e:
        log.warning("crash reports unavailable: %s", e)
    import threading

    def thread_hook(args):
        if args.exc_type is SystemExit:
            return
        log.error("Uncaught error in thread %s", getattr(args.thread, "name", "?"),
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    threading.excepthook = thread_hook


def _fatal(msg: str) -> None:
    if sys.stderr is not None:
        print(msg, file=sys.stderr)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "Nexovara Pose", 0x10)
        except Exception:
            pass
    sys.exit(1)


def main() -> None:
    if sys.version_info < (3, 10):
        _fatal("Нужен Python 3.10 или новее.")
    missing = []
    for mod, pkg in (("cv2", "opencv-contrib-python"), ("numpy", "numpy"),
                     ("mediapipe", "mediapipe"), ("PySide6", "PySide6-Essentials"),
                     ("psutil", "psutil")):
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        _fatal("Не установлены компоненты: " + ", ".join(missing) +
               "\n\nЗапустите setup.bat ещё раз.")

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    from fbt.config import app_dir, load_config
    from fbt.ui.main_window import MainWindow
    from fbt.ui.theme import apply_theme

    data_dir = app_dir()
    _setup_logging(data_dir)
    log = logging.getLogger("nexovara_pose")
    log.info("Starting Nexovara Pose")
    _crash_reports(data_dir, log)

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName("Nexovara Pose")
    apply_theme(app)
    cfg = load_config()
    win = MainWindow(cfg, models_dir=ROOT / "models", model_search_dirs=[ROOT / "models",
                                                                         data_dir / "models"])
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
