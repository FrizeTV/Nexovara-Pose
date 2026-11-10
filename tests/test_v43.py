"""v4.3: smoother and straighter tracking, headset anchoring through the driver,
VRChat one-shot yaw alignment, VMT removed."""
import json
import math
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbt import filters as F  # noqa: E402
from fbt.config import Config, load_config  # noqa: E402
from fbt.hmd import BodyAnchor, HmdState, parse_feedback  # noqa: E402
from fbt.mathutil import mat_to_unity_euler, rot_x, rot_y  # noqa: E402
from fbt.outputs import DriverOutput, VRChatOutput  # noqa: E402
from fbt.skeleton import (L_ANK, L_HEEL, L_KNEE, L_TOE, R_ANK, R_HEEL, R_KNEE, R_TOE, Tracker,  # noqa: E402
                          compute_trackers)
from tests.test_core import standing_pose_mp  # noqa: E402


def _unity_pose():
    from fbt.skeleton import mp_to_unity, uncalibrated_transform
    return uncalibrated_transform(mp_to_unity(standing_pose_mp()))


# ------------------------------------------------------------------ the twitch fix
class _Est:
    """Fake landmarker: the 'pose' it returns says which instance produced it."""
    made = []

    def __init__(self, model_path, *a, num_poses=1, **k):
        self.num_poses, self.delegate = num_poses, "CPU"
        _Est.made.append(self)

    def detect(self, rgb, ts):
        if self.num_poses == 1:
            return [] if _Est.lost else ["single"]
        return ["multi"] * _Est.people

    def close(self):
        pass


def test_pose_always_from_the_same_landmarker_when_alone():
    """v4.2 bug: twice a second the pose came from the other landmarker -> twitch."""
    from fbt.pose import make_estimator
    _Est.made, _Est.people, _Est.lost = [], 1, False
    est = make_estimator("m.task", num_poses=3, cls=_Est)
    got = [est.detect(None, i * 33)[0] for i in range(120)]
    assert set(got) == {"single"}                     # 4 s alone: never the other model
    assert _Est.made[1].num_poses == 3                # ... which still looked around
    _Est.people = 2                                   # someone walks in
    got = [est.detect(None, (120 + i) * 33) for i in range(40)]
    assert got[-1] == ["multi", "multi"]              # now everyone, from one source
    _Est.people, _Est.lost = 1, True
    assert est.detect(None, 9_000_000)                # lost -> filled by the full search


# ------------------------------------------------------------------ filters
def _noisy_static(n=300, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    base = _unity_pose()
    return [base + rng.standard_t(3, size=base.shape) * sigma for _ in range(n)], base


def test_pose_filter_less_jitter_than_old_filter():
    seq, _ = _noisy_static()
    old, new = F.OneEuroFilter(1.2, 1.5), F.PoseFilter(1.2, 1.5)
    o = np.array([old(x, i / 30) for i, x in enumerate(seq)])
    n = np.array([new(x, i / 30, np.ones(33)) for i, x in enumerate(seq)])
    jit = lambda y, ids: float(np.sqrt(np.mean(np.sum(np.diff(y[50:, ids], axis=0) ** 2, axis=2))))  # noqa: E731
    for ids in ([0, 11, 12, 23, 24], [25, 26], [27, 28, 29, 30, 31, 32]):
        assert jit(n, ids) < jit(o, ids)


def test_pose_filter_no_overshoot_and_not_slower():
    for speed in (0.5, 1.0, 2.0):
        X = np.zeros((90, 33, 3))
        for i in range(90):
            t = i / 30 - 0.5
            X[i, :, 0] = 0.0 if t < 0 else min(t * speed / 0.35, 1.0) * 0.35
        old, new = F.OneEuroFilter(1.2, 1.5), F.PoseFilter(1.2, 1.5)
        yo = np.array([old(X[i], i / 30) for i in range(90)])[:, 27, 0]
        yn = np.array([new(X[i], i / 30, np.ones(33)) for i in range(90)])[:, 27, 0]
        assert yn.max() <= 0.35 + 1e-6                          # never flies past
        assert np.argmax(yn > 0.315) <= np.argmax(yo > 0.315)   # reaches 90 % no later


def test_leg_swap_guard_swaps_back():
    g = F.LegSwapGuard()
    P = _unity_pose()
    P[[L_KNEE, L_ANK, L_HEEL, L_TOE], 0] -= 0.15        # legs apart
    P[[R_KNEE, R_ANK, R_HEEL, R_TOE], 0] += 0.15
    vis = np.linspace(0, 1, 33)
    g(P, vis)
    bad = F.swap_legs(P)
    bad_vis = vis.copy()
    bad_vis[[L_ANK, R_ANK]] = vis[[R_ANK, L_ANK]]       # the model swapped them together
    fixed, v = g(bad, bad_vis)
    assert np.allclose(fixed, P) and g.swaps == 1
    assert v[L_ANK] == vis[L_ANK] and v[R_ANK] == vis[R_ANK]
    small = P.copy()
    small[[L_ANK], 0] += 0.01                           # normal motion: untouched
    out, _ = g(small, vis)
    assert np.allclose(out, small)


def test_stabilizer_holds_noise_passes_motion():
    st = F.Stabilizer(0.004, 0.8)
    rng = np.random.default_rng(1)
    p0 = np.array([0.1, 1.0, 0.2])
    outs = [st.position("hip", p0 + rng.normal(0, 0.0012, 3)) for _ in range(60)]
    steps = np.linalg.norm(np.diff(outs, axis=0), axis=1)
    assert steps.mean() < 0.0006                         # mostly still
    moved = st.position("hip", p0 + np.array([0.05, 0.0, 0.0]))
    assert abs(moved[0] - (p0[0] + 0.05)) < 1e-9         # a real move passes at once
    R = rot_y(30.0)
    assert np.allclose(st.rotation("hip", R), R)
    assert np.allclose(st.rotation("hip", rot_y(30.2)), R, atol=1e-3)   # tiny wobble held
    assert np.allclose(st.rotation("hip", rot_y(40.0)), rot_y(40.0))


# ------------------------------------------------------------------ trackers
def test_foot_stays_flat_when_knees_bend():
    P = _unity_pose()
    flat = compute_trackers(P, ["left_foot"])["left_foot"].rot
    Q = P.copy()
    Q[[L_KNEE, R_KNEE], 2] += 0.15                       # knees forward (squat a bit)
    bent = compute_trackers(Q, ["left_foot"])["left_foot"].rot
    up_angle = math.degrees(math.acos(float(np.clip(bent[:, 1] @ flat[:, 1], -1, 1))))
    assert up_angle < 6.0                                # was ~20 deg with shin-up
    Q2 = P.copy()
    Q2[[L_TOE], 1] += 0.08                               # toes up: pitch is real
    tilted = compute_trackers(Q2, ["left_foot"])["left_foot"].rot
    assert tilted[1, 2] > flat[1, 2] + 0.15


def test_unity_euler_matches_vrchat_order():
    """VRChat: euler degrees applied Z, X, Y (Unity) — R = Ry @ Rx @ Rz."""
    def rz(a):
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for x, y, z in ((10, 20, 30), (-35, 170, 5), (60, -80, -20)):
        R = rot_y(y) @ rot_x(x) @ rz(z)
        ex, ey, ez = mat_to_unity_euler(R)
        R2 = rot_y(ey) @ rot_x(ex) @ rz(ez)
        assert np.allclose(R, R2, atol=1e-9)


# ------------------------------------------------------------------ headset
def _feedback(pos, quat, flags=1):
    return struct.pack("<4sHHII3f4f", b"CFBH", 1, flags, 7, 0, *pos, *quat)


def test_parse_feedback_strict():
    pos, rot, fl = parse_feedback(_feedback((0.2, 1.7, -0.4), (0, 0, 0, 1)))
    assert np.allclose(pos, [0.2, 1.7, 0.4]) and np.allclose(rot, np.eye(3)) and fl == 1
    for bad in (b"", _feedback((0, 1, 0), (0, 0, 0, 1))[:-1], b"XXXX" + _feedback((0, 1, 0), (0, 0, 0, 1))[4:],
                _feedback((float("nan"), 1, 0), (0, 0, 0, 1)), _feedback((0, 1, 0), (0, 0, 0, 0)),
                _feedback((500, 1, 0), (0, 0, 0, 1))):
        assert parse_feedback(bad) is None


def test_driver_feedback_reaches_hmd_state():
    drv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    drv.bind(("127.0.0.1", 0))
    drv.settimeout(2)
    out = DriverOutput(drv.getsockname()[1])
    state = HmdState()
    out.send([Tracker("hip", np.zeros(3), np.eye(3))], None)
    _data, app = drv.recvfrom(4096)
    drv.sendto(_feedback((0.0, 1.75, 0.0), (0, 0, 0, 1)), app)
    stranger = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stranger.sendto(_feedback((9.0, 9.0, 9.0), (0, 0, 0, 1)), app)   # not from the driver
    out.poll_feedback(state)            # starts the receiving thread
    time.sleep(0.2)
    p = state.latest()
    assert p is not None and p.source == "driver" and abs(p.pos[1] - 1.75) < 1e-6
    drv.sendto(_feedback((0.0, 1.6, 0.0), (0, 0, 0, 1), flags=3), app)  # our virtual HMD
    time.sleep(0.2)
    assert state.latest().virtual
    out.shutdown()
    drv.close()
    stranger.close()


def test_hmd_state_prefers_driver_over_openvr():
    st = HmdState()
    st.update(np.array([0, 1.7, 0]), np.eye(3), "driver", t=10.0)
    st.update(np.array([5, 5, 5]), np.eye(3), "openvr", t=10.1)
    assert st.latest(now=10.2).source == "driver"
    st.update(np.array([5, 5, 5]), np.eye(3), "openvr", t=11.0)
    assert st.latest(now=11.0).source == "openvr"     # driver silent -> fallback used


def test_body_anchor_glues_head_to_headset_and_damps_noise():
    from fbt.hmd import HmdPose
    anchor = BodyAnchor()
    hmd_R = rot_y(90.0)
    hmd = HmdPose(np.array([1.0, 1.7, 2.0]), hmd_R, 0.0, "driver")
    rng = np.random.default_rng(3)
    heads, hips = [], []
    for i in range(120):
        head = Tracker("head", np.array([0.0, 1.62, 0.0]) + rng.normal(0, 0.004, 3), np.eye(3))
        hip = Tracker("hip", np.array([0.0, 1.0, 0.0]), np.eye(3))
        tr, h, _ = anchor.apply([hip], head, {}, hmd, 90.0, i / 90)
        heads.append(h.pos)
        hips.append(tr[0].pos)
    target = BodyAnchor.head_target(hmd)
    assert np.linalg.norm(np.mean(heads[60:], axis=0) - target) < 0.004
    # the hip hangs below the headset, turned with the calibration heading
    assert abs(tr[0].pos[1] - (target[1] - 0.62)) < 0.02
    assert np.allclose(tr[0].rot, rot_y(90.0))
    # the camera's head noise no longer shakes the whole body (offset is filtered;
    # before, the body was re-glued every frame and the hip moved with every
    # head wobble: steps ~= the head's, 4-8 mm)
    head_steps = np.linalg.norm(np.diff(heads[60:], axis=0), axis=1).mean()
    hip_steps = np.linalg.norm(np.diff(hips[60:], axis=0), axis=1).mean()
    assert hip_steps < 0.35 * head_steps
    # walking: follows the headset closely
    for i in range(120, 180):
        hmd = HmdPose(np.array([1.0 + (i - 120) * 0.01, 1.7, 2.0]), hmd_R, i / 90, "driver")
        _tr, h, _ = anchor.apply([], Tracker("head", np.array([0.0, 1.62, 0.0]), np.eye(3)), {},
                                 hmd, 90.0, i / 90)
    lag = BodyAnchor.head_target(hmd)[0] - h.pos[0]
    assert lag < 0.08                                   # at 0.9 m/s


def test_sender_anchors_driver_output_only_to_a_real_headset(monkeypatch):
    from fbt import sender as SM
    got = []
    monkeypatch.setattr(DriverOutput, "send", lambda self, tr, head, hands=None, controls=None,
                        **_k: got.append((tr[0].pos.copy(), head.pos.copy())))
    monkeypatch.setattr(DriverOutput, "poll_feedback", lambda self, state: 0)
    cfg = Config(output_vrchat=False, output_driver=True, driver_hmd=False, hmd_anchor=True,
                 output_smoothing=False).validate()
    st = HmdState()
    snd = SM.OutputSender(st)
    snd.configure(cfg)
    snd.start()
    try:
        head = Tracker("head", np.array([0.0, 1.6, 0.0]), np.eye(3))
        hip = Tracker("hip", np.array([0.0, 1.0, 0.0]), np.eye(3))
        snd.push(SM.Keyframe(time.monotonic(), [hip], head, {}, 0.0))
        time.sleep(0.15)
        assert got and np.allclose(got[-1][0], [0, 1.0, 0])     # no headset: unchanged
        st.update(np.array([3.0, 1.8, 0.0]), np.eye(3), "driver")
        for _ in range(3):
            snd.push(SM.Keyframe(time.monotonic(), [hip], head, {}, 0.0))
            time.sleep(0.05)
            st.update(np.array([3.0, 1.8, 0.0]), np.eye(3), "driver")
        assert got[-1][0][0] > 2.5 and snd.anchored          # glued to the headset
        st.update(np.array([3.0, 1.8, 0.0]), np.eye(3), "driver", virtual=True)
        snd.push(SM.Keyframe(time.monotonic(), [hip], head, {}, 0.0))
        time.sleep(0.1)
        assert np.allclose(got[-1][0], [0, 1.0, 0])          # our own virtual HMD: never
    finally:
        snd.stop()


# ------------------------------------------------------------------ VRChat OSC
def test_vrchat_head_rotation_only_once_on_request():
    po = pytest.importorskip("pythonosc.osc_message")
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.3)
    out = VRChatOutput(rx.getsockname()[1])
    rng = np.random.default_rng(0)
    for i in range(30):
        head = Tracker("head", np.array([0.0, 1.6, 0.0]) + rng.normal(0, 0.003, 3), rot_y(5.0))
        out.send([], head, i / 90)
        if i == 10:
            out.request_yaw_align(12.0)
    msgs = []
    try:
        while True:
            msgs.append(po.OscMessage(rx.recv(4096)))
    except socket.timeout:
        pass
    rot = [m for m in msgs if m.address == "/tracking/trackers/head/rotation"]
    assert len(rot) == 1 and abs(rot[0].params[1] - 12.0) < 1e-4 and rot[0].params[0] == 0
    pos = np.array([m.params for m in msgs if m.address == "/tracking/trackers/head/position"])
    assert len(pos) == 30
    assert np.std(pos[10:, 0]) < 0.003 * 0.6           # head noise damped before VRChat
    out.shutdown()
    rx.close()


# ------------------------------------------------------------------ VMT removed
def test_old_vmt_config_is_migrated(tmp_path):
    f = tmp_path / "config.json"
    f.write_text(json.dumps({"schema_version": 3, "target": "steamvr", "output_vmt": True,
                             "vmt_port": 39570, "finger_output": "vmt",
                             "user_set": ["output_vmt", "vmt_port", "camera"]}))
    cfg = load_config(f)
    assert cfg.target == "steamvr_driver" and cfg.output_driver and cfg.finger_output == "driver"
    assert not hasattr(cfg, "output_vmt") and "output_vmt" not in cfg.user_set
    assert "camera" in cfg.user_set
    from fbt import presets
    assert "steamvr" not in presets.PRESETS


def test_no_vmt_left_in_the_program():
    for p in list((ROOT / "fbt").rglob("*.py")) + list((ROOT / "tools").glob("*.py")):
        text = p.read_text(encoding="utf-8")
        if p.name == "config.py":
            text = text.split("def _migrate_vmt")[0] + text.split("def load_config")[1]
        assert "/VMT/" not in text and "VMTOutput" not in text and "vmt_port" not in text, p
