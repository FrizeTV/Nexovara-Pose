"""End-to-end pipeline tests with a fake camera and fake models.

Checks that the engine runs, calibrates, applies startup automation and sends
valid VRChat OSC + Nexovara Pose driver packets (including finger curls).
"""
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import engine as E  # noqa: E402
from fbt.config import Config  # noqa: E402
from fbt.hands import HandResult  # noqa: E402
from fbt.pose import PoseResult  # noqa: E402
from tests.test_core import standing_pose_mp  # noqa: E402

po = pytest.importorskip("pythonosc.osc_message")


class FakeCamera:
    def __init__(self, *a, **k):
        self.error = None
        self.measured_fps = 30.0
        self.actual_size = (640, 480)
        self.actual_fourcc = "MJPG"
        self._seq = 0

    def start(self, wait=False):
        pass

    def get_latest(self, last_seq, timeout=0.5):
        time.sleep(1 / 60)
        self._seq += 1
        return self._seq, np.zeros((480, 640, 3), np.uint8), time.monotonic()

    def stop(self):
        pass


class FakeEstimator:
    delegate = "FAKE"

    def __init__(self, *a, **k):
        self.base = standing_pose_mp(1.7)
        self.rng = np.random.default_rng(1)

    def process_rgb(self, rgb, ts):
        w = self.base + self.rng.normal(0, 0.005, self.base.shape)
        img = np.zeros((33, 3))
        img[:, :2] = 0.5 + w[:, :2] * 0.3
        return PoseResult(w, img, np.ones(33))

    process = process_rgb

    def detect(self, rgb, ts):
        return [self.process_rgb(rgb, ts)]

    def close(self):
        pass


class FakeHands:
    def __init__(self, *a, **k):
        pass

    def process(self, rgb, pose_image, vis, ts):
        img = np.full((21, 2), 0.5)
        return {"left": HandResult(img, np.full(5, 0.8)), "right": HandResult(img, np.zeros(5))}

    def close(self):
        pass


def _listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(0.05)
    return s


def _drain(sock, seconds=0.5):
    msgs = []
    t = time.time() + seconds
    while time.time() < t:
        try:
            msgs.append(po.OscMessage(sock.recv(4096)))
        except socket.timeout:
            pass
    return msgs


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.setattr(E, "CameraSource", FakeCamera)
    monkeypatch.setattr(E, "PoseEstimator", FakeEstimator)
    monkeypatch.setattr(E, "ensure_model", lambda *a, **k: tmp_path / "m.task")
    monkeypatch.setattr(E, "detect_target", lambda: ("vrchat_osc", "test"))
    probes = []

    def fake_probe(index, progress=None, stop=None):
        probes.append(index)
        return {"camera": str(index), "backend": 0, "time": 0, "modes": [
            {"width": 1280, "height": 720, "fps": 60.0, "fps_requested": 60, "fourcc": "MJPG"},
            {"width": 640, "height": 480, "fps": 30.0, "fps_requested": 60, "fourcc": "YUY2"}]}
    monkeypatch.setattr(E, "probe_camera", fake_probe)
    import fbt.hands
    monkeypatch.setattr(fbt.hands, "HandTracker", FakeHands)
    return probes


def _wait_calibrated(eng, timeout=5.0):
    deadline = time.time() + timeout
    events = []
    while time.time() < deadline:
        for kind, payload in eng.poll_events():
            assert kind != "error", payload
            events.append((kind, payload))
            if kind == "calibrated":
                return payload, events
        time.sleep(0.05)
    raise AssertionError("calibration did not finish")


def _drain_raw(sock, wait=0.3, limit=1.0):
    """Datagrams for `limit` seconds at most (the output never stops sending)."""
    sock.settimeout(wait)
    out, end = [], time.time() + limit
    while time.time() < end:
        try:
            out.append(sock.recv(65535))
        except socket.timeout:
            break
    return out


def _driver_records(pkt):
    import struct
    assert pkt[:4] == b"CFBT"
    n = pkt[12]
    recs = {}
    for i in range(n):
        r = pkt[16 + 52 * i:16 + 52 * (i + 1)]
        dev, flags, buttons = struct.unpack("<BBH", r[:4])
        vals = struct.unpack("<3f4f5f", r[4:])
        recs[dev] = (flags, vals)
    return recs


def test_engine_end_to_end(fakes, tmp_path):
    vrc, drv = _listen(), _listen()
    cfg = Config(vrchat_port=vrc.getsockname()[1], output_driver=True,
                 driver_port=drv.getsockname()[1], hmd_anchor=False, fingers=True,
                 finger_output="driver", hand_every=1).validate()
    # the user set these by hand -> the "auto" target preset must not override them
    cfg.mark_user_set(["vrchat_port", "output_driver", "driver_port", "hmd_anchor",
                       "finger_output"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    try:
        time.sleep(0.3)
        eng.request_calibration(10)
        cal, events = _wait_calibrated(eng)
        kinds = [k for k, _ in events]
        assert "target" in kinds and "camera_profile" in kinds
        applied = [p for k, p in events if k == "auto_applied"]
        merged = {k: v for p in applied for k, v in p["values"].items()}
        assert merged.get("width") == 1280 and merged.get("fps") == 60   # best camera mode
        assert "vrchat_port" not in merged and "output_driver" not in merged  # user choices
        assert cal["lengths"] and "thigh" in cal["lengths"]

        cfg.calibration = cal
        eng.update_config(cfg)
        time.sleep(0.5)

        vm = _drain(vrc)
        addrs = [m.address for m in vm]
        for a in ("/tracking/trackers/1/position", "/tracking/trackers/1/rotation",
                  "/tracking/trackers/3/position", "/tracking/trackers/head/position"):
            assert a in addrs, a
        # VRChat's heading is aligned ONCE after calibration, never continuously
        assert addrs.count("/tracking/trackers/head/rotation") <= 1
        hip = [m for m in vm if m.address == "/tracking/trackers/1/position"][-1].params
        assert abs(hip[0]) < 0.05 and 0.7 < hip[1] < 1.2  # hips centred, ~1 m high
        head = [m for m in vm if m.address == "/tracking/trackers/head/position"][-1].params
        assert 1.45 < head[1] < 1.75
        for m in vm:
            assert all(np.isfinite(p) for p in m.params)

        pk = _drain_raw(drv)
        assert pk
        recs = _driver_records(pk[-1])
        assert {0, 1, 2} <= set(recs)                               # fixed slots
        q = np.array(recs[0][1][3:7])
        assert abs(np.linalg.norm(q) - 1) < 1e-4
        assert 17 in recs and 18 in recs                            # Index-style hands
        assert recs[17][0] & 2 and max(recs[17][1][7:]) > 0.5       # curled fingers sent

        snap = eng.snapshot()
        assert snap["calibrated"] and snap["tracking"] and len(snap["trackers"]) == 3
        assert snap["hands"]
    finally:
        eng.stop()
        vrc.close()
        drv.close()
    assert not eng.running


def test_outputs_work_without_any_config_update(fakes, tmp_path):
    """Regression: the initial settings must be applied on start (outputs, fingers)."""
    vrc = _listen()
    cfg = Config(vrchat_port=vrc.getsockname()[1]).validate()
    cfg.mark_user_set(["vrchat_port"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    try:
        msgs = _drain(vrc, 1.0)
    finally:
        eng.stop()
        vrc.close()
    assert any(m.address == "/tracking/trackers/1/position" for m in msgs)


def test_camera_mode_user_choice_is_respected(fakes, tmp_path):
    cfg = Config(width=848, height=480, fps=30, output_vrchat=False).validate()
    cfg.mark_user_set(["width", "height", "fps", "fourcc", "output_vrchat"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    time.sleep(0.6)
    events = eng.poll_events()
    eng.stop()
    assert fakes == []  # no probing when the user fixed the camera mode
    for k, p in events:
        if k == "auto_applied":
            assert not {"width", "height", "fps", "fourcc"} & set(p["values"])


def test_engine_follows_one_person_among_two(fakes, tmp_path, monkeypatch):
    from tests.test_person_realism import BLUE, RED, frame_with, person
    me, other = person(0.45, 1.2), person(0.8, 0.8, legs=1.25)
    rgb = frame_with([(me, RED), (other, BLUE)])

    class TwoPeople(FakeEstimator):
        n = 0

        def detect(self, _rgb, ts):
            TwoPeople.n += 1
            return [other, me] if TwoPeople.n % 2 else [me, other]

    class Cam(FakeCamera):
        def get_latest(self, last_seq, timeout=0.5):
            time.sleep(1 / 60)
            self._seq += 1
            return self._seq, rgb[:, :, ::-1].copy(), time.monotonic()  # engine expects BGR

    monkeypatch.setattr(E, "PoseEstimator", TwoPeople)
    monkeypatch.setattr(E, "CameraSource", Cam)
    cfg = Config(output_vrchat=False).validate()
    cfg.mark_user_set(["output_vrchat"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    try:
        time.sleep(0.8)
        snap = eng.snapshot()
        assert snap["tracking"] and len(snap["others"]) == 1
        assert np.allclose(snap["image_lm"], me.image)          # always the same person
        eng.pick_person(0.8, 0.5)                                # user clicks the other one
        time.sleep(0.4)
        assert np.allclose(eng.snapshot()["image_lm"], other.image)
    finally:
        eng.stop()
