"""v4.4.1: camera hot-plug safety — the reader thread owns the capture, lost
cameras are reconnected by name, nothing touches cameras while Windows is
installing a device, one lock for all device operations."""
import ctypes
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import camera as C  # noqa: E402


class FakeCap:
    def __init__(self, frames=10**9, block=None):
        self.frames = frames
        self.block = block
        self.released = 0
        self.reads = 0
        self.lock = threading.Lock()

    def isOpened(self):  # noqa: N802
        return True

    def set(self, *a):
        return True

    def get(self, prop):
        return 0.0

    def read(self):
        if self.block is not None:
            self.block.wait()
        with self.lock:
            assert self.released == 0, "read() after release(): use-after-free"
            self.reads += 1
            if self.reads > self.frames:
                time.sleep(0.001)
                return False, None
        time.sleep(0.002)
        return True, np.zeros((48, 64, 3), np.uint8)

    def release(self):
        with self.lock:
            self.released += 1


def _wait(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_unplugged_camera_is_released_and_reopened_by_name(monkeypatch):
    caps, opened = [FakeCap(frames=20), FakeCap()], []

    def fake_open(index):
        opened.append(index)
        return caps[len(opened) - 1], 0
    monkeypatch.setattr(C, "open_capture", fake_open)
    plugged = {"on": False}
    monkeypatch.setattr(C, "enumerate_devices", lambda *a: (
        [C.CameraDevice(3, "Logitech C920")] if plugged["on"] else []))
    monkeypatch.setattr(C.CameraSource, "RETRY_S", 0.05)
    cam = C.CameraSource("0", 640, 480, 30, "MJPG", name="Logitech C920")
    cam.start()
    try:
        assert _wait(lambda: cam.lost)                      # unplugged
        assert caps[0].released == 1 and cam._cap is None   # let go at once
        time.sleep(0.3)
        assert len(opened) == 1                             # not plugged in: no opening
        plugged["on"] = True                                # plugged into another port
        assert _wait(lambda: not cam.lost and cam.reconnects == 1)
        assert opened == [0, 3]                             # same camera, new number
        seq = cam.get_latest(0, 1.0)[0]
        assert cam.get_latest(seq, 1.0) is not None         # frames flow again
    finally:
        cam.stop()
    assert caps[1].released == 1                            # released by the reader


def test_no_reopen_while_windows_installs_a_device(monkeypatch):
    caps, opened = [FakeCap(frames=5), FakeCap()], []
    monkeypatch.setattr(C, "open_capture",
                        lambda i: (opened.append(i), (caps[len(opened) - 1], 0))[1])
    monkeypatch.setattr(C, "enumerate_devices", lambda *a: None)
    monkeypatch.setattr(C.CameraSource, "RETRY_S", 0.05)
    monkeypatch.setattr(C, "SETTLE_S", 0.6)
    cam = C.CameraSource("0", 640, 480, 30)
    cam.start()
    try:
        assert _wait(lambda: cam.lost)
        C.notify_device_change()
        t0 = time.monotonic()
        assert _wait(lambda: not cam.lost, 5.0)
        assert time.monotonic() - t0 >= 0.5                 # waited for the device to settle
    finally:
        cam.stop()
        C._device_change_t = -1e9


def test_stop_never_releases_a_capture_the_reader_is_using(monkeypatch):
    gate = threading.Event()
    cap = FakeCap(block=gate)
    monkeypatch.setattr(C, "open_capture", lambda i: (cap, 0))
    monkeypatch.setattr(C.CameraSource, "STOP_JOIN_S", 0.2)
    cam = C.CameraSource("0", 640, 480, 30)
    cam._cap = cap
    cam.start()
    time.sleep(0.1)                              # the reader is stuck inside read()
    cam.stop()
    assert cap.released == 0                     # not freed under the driver's feet
    gate.set()                                   # read() returns
    assert _wait(lambda: cap.released == 1)      # the reader closes it itself


def test_names_come_from_cache_while_devices_settle(monkeypatch):
    calls = []
    monkeypatch.setattr(C, "_enumerate_os", lambda *a: calls.append(1) or
                        [C.CameraDevice(0, "Cam")])
    C._names_cache = (-1e9, None)
    try:
        assert C.enumerate_devices(0)[0].name == "Cam" and len(calls) == 1
        C.notify_device_change()
        assert C.devices_settling()
        assert C.enumerate_devices(0)[0].name == "Cam" and len(calls) == 1   # cached
    finally:
        C._device_change_t = -1e9
        C._names_cache = (-1e9, None)
    assert not C.devices_settling()


def test_device_operations_do_not_deadlock():
    errors = []

    def lister():
        for _ in range(30):
            C.enumerate_devices(0)

    def opener():
        for _ in range(30):
            try:
                C.open_capture(97)               # no such camera: fails quickly
            except RuntimeError:
                pass
            except Exception as e:
                errors.append(e)
    ts = [threading.Thread(target=f, daemon=True) for f in (lister, opener, lister, opener)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=60)
    assert not any(t.is_alive() for t in ts), "deadlock"
    assert not errors


def test_device_change_filter_reads_wm_devicechange(monkeypatch):
    pytest.importorskip("PySide6")
    from fbt.ui import devwatch as D

    class MSG(ctypes.Structure):
        _fields_ = [("hwnd", ctypes.c_void_p), ("message", ctypes.c_uint),
                    ("wParam", ctypes.c_size_t), ("lParam", ctypes.c_ssize_t),
                    ("time", ctypes.c_uint32), ("x", ctypes.c_long), ("y", ctypes.c_long)]
    monkeypatch.setattr(D, "_MSG", MSG)
    got = []
    f = D.DeviceChangeFilter(got.append)
    m = MSG(None, D.WM_DEVICECHANGE, D.DBT_DEVNODES_CHANGED, 0, 0, 0, 0)
    assert f.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(m)) == (False, 0)
    m2 = MSG(None, 0x0200, 0, 0, 0, 0, 0)                     # mouse move: ignored
    f.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(m2))
    f.nativeEventFilter(b"xcb_generic_event_t", ctypes.addressof(m))
    assert got == [D.DBT_DEVNODES_CHANGED]
    bad = D.DeviceChangeFilter(lambda k: 1 / 0)                # a failing callback ...
    assert bad.nativeEventFilter(b"windows_generic_MSG", ctypes.addressof(m)) == (False, 0)
    assert bad._failed                                         # ... disables itself quietly


def test_engine_keeps_running_when_the_camera_is_unplugged(monkeypatch, tmp_path):
    from fbt import engine as E
    from fbt.config import Config
    from tests.test_engine import FakeCamera, FakeEstimator

    class Unplugged(FakeCamera):
        n = 0

        def get_latest(self, last_seq, timeout=0.5):
            Unplugged.n += 1
            self.lost = 30 < Unplugged.n < 60
            if self.lost:
                time.sleep(0.02)
                return None
            return super().get_latest(last_seq, timeout)
    monkeypatch.setattr(E, "CameraSource", Unplugged)
    monkeypatch.setattr(E, "PoseEstimator", FakeEstimator)
    monkeypatch.setattr(E, "ensure_model", lambda *a, **k: tmp_path / "m.task")
    cfg = Config(output_vrchat=False, target="test", width=640, height=480, fps=30,
                 fourcc="MJPG").validate()
    cfg.mark_user_set(["output_vrchat", "width", "height", "fps", "fourcc"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    seen_lost = False
    try:
        end = time.time() + 4
        while time.time() < end and Unplugged.n < 90:
            s = eng.snapshot()
            seen_lost |= bool(s and s.get("camera_lost"))
            time.sleep(0.01)
        ev = eng.poll_events()
        assert eng.running and eng.snapshot()["tracking"]
    finally:
        eng.stop()
    msgs = [p for k, p in ev if k == "status"]
    assert seen_lost and any("отключена" in m for m in msgs) and any("снова" in m for m in msgs)
    assert not [p for k, p in ev if k == "error"]


def test_same_mode_prefers_mjpg():
    modes = [C.CameraMode(1280, 720, 30.2, 60, "YUY2"), C.CameraMode(1280, 720, 30.0, 60, "MJPG")]
    assert C.choose_best_mode(modes, prefer_hd=True).fourcc == "MJPG"
    faster = [C.CameraMode(1280, 720, 60.0, 60, "YUY2"), C.CameraMode(1280, 720, 30.0, 60, "MJPG")]
    assert C.choose_best_mode(faster, prefer_hd=True).fourcc == "YUY2"   # speed still wins
