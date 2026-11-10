"""v4.4.4: "camera opened but delivers no frames" — the camera of a previous run is
waited for, slow first frames are waited for, a busy camera no longer stops tracking."""
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import camera as C  # noqa: E402
from tests.test_v441 import FakeCap, _wait  # noqa: E402


def test_fourcc_garbage_is_not_shown():
    mjpg = sum(ord(c) << (8 * i) for i, c in enumerate("MJPG"))
    assert C._fourcc_str(mjpg) == "MJPG"
    garbage = sum(ord(c) << (8 * i) for i, c in enumerate("}\xeb6\xe4"))
    assert C._fourcc_str(garbage) == "?"
    assert C._fourcc_str(float("nan")) == "?"


class SlowStartCap(FakeCap):
    def __init__(self, bad_reads=5):
        super().__init__()
        self.bad = bad_reads

    def read(self):
        if self.bad > 0:
            self.bad -= 1
            return False, None
        return super().read()


def test_slow_first_frame_is_waited_for(monkeypatch):
    cap = SlowStartCap(5)
    monkeypatch.setattr(C, "open_capture", lambda i: (cap, 0))
    cam = C.CameraSource("0", 640, 480, 30, name="Cam A")
    cam.start(wait=True)
    try:
        assert not cam.lost and cam.get_latest(0, 1.0) is not None
    finally:
        cam.stop()


def test_new_run_waits_until_the_old_run_let_go_of_the_camera(monkeypatch):
    caps = []
    gate = threading.Event()

    class Blocking(FakeCap):
        def read(self):
            if gate.is_set():
                time.sleep(0.4)          # the old reader is finishing a slow frame
            return super().read()

    def fake_open(i):
        caps.append(Blocking())
        return caps[-1], 0
    monkeypatch.setattr(C, "open_capture", fake_open)
    old = C.CameraSource("0", 640, 480, 30, name="Cam B")
    old.start(wait=True)
    gate.set()
    monkeypatch.setattr(C.CameraSource, "STOP_JOIN_S", 0.05)
    old.stop()                            # returns before the reader released the camera
    new = C.CameraSource("0", 640, 480, 30, name="Cam B")
    t0 = time.monotonic()
    new.start(wait=True)
    try:
        assert caps[0].released == 1      # opened only after the old one was closed
        assert time.monotonic() - t0 > 0.1
    finally:
        gate.clear()
        new.stop()


def test_busy_camera_does_not_stop_tracking(monkeypatch):
    state = {"busy": True}
    good = FakeCap()

    def fake_open(i):
        if state["busy"]:
            return FakeCap(frames=0), 0   # opens, but no frames (another program has it)
        return good, 0
    monkeypatch.setattr(C, "open_capture", fake_open)
    monkeypatch.setattr(C, "enumerate_devices", lambda *a: None)
    monkeypatch.setattr(C.CameraSource, "FIRST_FRAME_S", 0.1)
    monkeypatch.setattr(C.CameraSource, "RETRY_S", 0.05)
    cam = C.CameraSource("0", 640, 480, 30, name="Cam C")
    cam.start(wait=True)                  # no exception any more
    try:
        assert cam.lost and "не отдаёт кадры" in cam.open_error
        state["busy"] = False             # the other program closed the camera
        assert _wait(lambda: not cam.lost)
        assert cam.get_latest(0, 1.0) is not None and cam.open_error is None
    finally:
        cam.stop()
    assert C._claimed.get("Cam C", 0) == 0


def test_without_wait_the_error_is_still_reported(monkeypatch):
    monkeypatch.setattr(C, "open_capture", lambda i: (FakeCap(frames=0), 0))
    monkeypatch.setattr(C.CameraSource, "FIRST_FRAME_S", 0.05)
    cam = C.CameraSource("0", 640, 480, 30)
    try:
        cam.start()
    except RuntimeError as e:
        assert "не отдаёт кадры" in str(e)
    else:
        raise AssertionError("expected an error")
    assert np.all(True)
