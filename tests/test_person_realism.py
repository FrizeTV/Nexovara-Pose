"""Tests for v3: person lock, realism (Standable-like), output interpolation,
loopback-only output."""
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import skeleton as S  # noqa: E402
from fbt.config import Config  # noqa: E402
from fbt.mathutil import rot_y  # noqa: E402
from fbt.osc import OscClient  # noqa: E402
from fbt.person import PersonLock  # noqa: E402
from fbt.pose import PoseResult  # noqa: E402
from fbt.realism import Realism, estimate_legs, fix_knees  # noqa: E402
from fbt.sender import Keyframe, OutputSender, interpolate  # noqa: E402
from fbt.skeleton import Tracker  # noqa: E402
from tests.test_core import standing_pose_mp  # noqa: E402

W, H = 640, 480


# ---------------------------------------------------------------- helpers
def person(cx: float, scale: float = 1.0, height: float = 1.7, legs: float = 1.0) -> PoseResult:
    """A person whose hips are at image x = cx; `legs` scales leg length (proportions)."""
    world = standing_pose_mp(height)
    leg_ids = [S.L_KNEE, S.R_KNEE, S.L_ANK, S.R_ANK, S.L_HEEL, S.R_HEEL, S.L_TOE, S.R_TOE]
    world[leg_ids, 1] *= legs
    img = np.zeros((33, 3))
    img[:, 0] = cx + world[:, 0] * 0.25 * scale
    img[:, 1] = 0.45 + world[:, 1] * 0.33 * scale
    return PoseResult(world, img, np.ones(33))


def frame_with(people_colors):
    """RGB frame where each person's torso is filled with a colour."""
    rgb = np.full((H, W, 3), 40, np.uint8)
    for p, col in people_colors:
        pts = p.image[[S.L_SH, S.R_SH, S.L_HIP, S.R_HIP], :2] * [W, H]
        x0, y0 = pts.min(axis=0).astype(int)
        x1, y1 = pts.max(axis=0).astype(int)
        rgb[y0:y1, x0:x1] = col
    return rgb


RED, BLUE, GREEN = (200, 30, 30), (30, 40, 200), (30, 180, 60)


# ---------------------------------------------------------------- person lock
def test_lock_picks_main_person_and_ignores_others():
    lock = PersonLock()
    me, other = person(0.5, 1.2), person(0.85, 0.8, legs=1.25)
    t = 0.0
    chosen, info = lock.select([other, me], frame_with([(me, RED), (other, BLUE)]), t)
    assert chosen is me and info["people"] == 2 and len(info["others"]) == 1
    # the detector's order changes and the other person walks past — still me
    for i in range(1, 30):
        t = i / 30
        other = person(0.85 - i * 0.01, 0.8, legs=1.25)
        rgb = frame_with([(me, RED), (other, BLUE)])
        chosen, _ = lock.select([me, other] if i % 2 else [other, me], rgb, t)
        assert chosen is me, i


def test_lock_does_not_jump_to_someone_else():
    lock = PersonLock()
    me, other = person(0.4), person(0.75, legs=1.25)
    lock.select([me, other], frame_with([(me, RED), (other, BLUE)]), 0.0)
    # I leave the frame; the other person stays -> nothing is tracked
    for i in range(1, 60):
        chosen, info = lock.select([other], frame_with([(other, BLUE)]), i / 30 * 0.05 + 0.1,
                                   relock_single_s=0)
        assert chosen is None and "ищу" in info["status"]
    # I come back somewhere else in the frame -> recognised by clothes + proportions
    me2 = person(0.2)
    chosen, _ = lock.select([other, me2], frame_with([(me2, RED), (other, BLUE)]), 5.0)
    assert chosen is me2


def test_relock_when_only_one_person_for_a_while():
    lock = PersonLock()
    me = person(0.5)
    lock.select([me], frame_with([(me, RED)]), 0.0)
    stranger = person(0.5, legs=1.3)
    rgb = frame_with([(stranger, GREEN)])
    chosen, _ = lock.select([stranger], rgb, 2.0, relock_single_s=3.0)
    assert chosen is None
    chosen, info = lock.select([stranger], rgb, 5.5, relock_single_s=3.0)
    assert chosen is stranger and "перезахват" in info["status"]


def test_click_to_pick_person():
    lock = PersonLock()
    a, b = person(0.3), person(0.75, legs=1.25)
    rgb = frame_with([(a, RED), (b, BLUE)])
    lock.select([a, b], rgb, 0.0)
    lock.pick(0.75, 0.5)
    chosen, info = lock.select([a, b], rgb, 0.03)
    assert chosen is b and "вручную" in info["status"]
    chosen, _ = lock.select([b, a], rgb, 0.06)
    assert chosen is b


def test_no_people():
    chosen, info = PersonLock().select([], frame_with([]), 0.0)
    assert chosen is None and info["people"] == 0


# ---------------------------------------------------------------- realism
def _cal_pose():
    raw = S.mp_to_unity(standing_pose_mp(1.7))
    cal = S.compute_calibration([raw], 1.75)
    return cal, cal.apply(raw)


def test_knee_bent_backwards_is_mirrored():
    _, P = _cal_pose()
    bad = P.copy()
    bad[S.L_KNEE, 2] -= 0.12           # knee behind the leg line (user faces +Z)
    fixed = fix_knees(bad)
    assert fixed[S.L_KNEE, 2] > P[S.L_KNEE, 2] + 0.05
    good = P.copy()
    good[S.R_KNEE, 2] += 0.12          # normal forward bend is untouched
    assert np.allclose(fix_knees(good), good)


def test_leg_estimation_fills_invisible_legs():
    cal, P = _cal_pose()
    garbage = P.copy()
    legs = [S.L_KNEE, S.L_ANK, S.L_HEEL, S.L_TOE, S.R_KNEE, S.R_ANK, S.R_HEEL, S.R_TOE]
    garbage[legs] += np.array([0.3, 0.4, -0.5])     # nonsense where the camera can't see
    vis = np.ones(33)
    vis[legs] = 0.05
    r = Realism()
    out, info = None, None
    for _ in range(40):                               # ~1.3 s at 30 fps
        out, info = r.process(garbage, vis, 1 / 30, cal.lengths, auto_floor=False)
    assert min(info.legs_estimated) > 0.95
    assert info.statuses["left_foot"] == "estimated"
    for hip, ank in ((S.L_HIP, S.L_ANK), (S.R_HIP, S.R_ANK)):
        assert abs(out[ank, 0] - out[hip, 0]) < 0.05     # feet under the hips
        assert out[ank, 1] < 0.2                         # near the floor
    # legs visible again -> back to the measurement, smoothly
    vis[:] = 1.0
    out1, info1 = r.process(P, vis, 1 / 30, cal.lengths, auto_floor=False)
    assert 0.0 < max(info1.legs_estimated) < 1.0         # blending, not snapping
    for _ in range(40):
        out1, info1 = r.process(P, vis, 1 / 30, cal.lengths, auto_floor=False)
    assert max(info1.legs_estimated) < 0.01 and np.abs(out1 - P).max() < 1e-3


def test_sitting_state_and_seated_leg_estimate():
    cal, P = _cal_pose()
    seated = P.copy()
    seated[:, 1] -= 0.45                               # hips much lower
    vis = np.ones(33)
    r = Realism()
    for _ in range(40):
        _, info = r.process(seated, vis, 1 / 30, cal.lengths, auto_floor=False)
    assert info.state == "сидит"
    E = estimate_legs(seated, cal.lengths, sit=1.0, lie=0.0)
    thigh = E[S.L_KNEE] - E[S.L_HIP]
    assert thigh[2] > 0.3 and abs(thigh[1]) < 0.05     # thighs forward, horizontal


def test_auto_floor_fixes_floating_feet():
    cal, P = _cal_pose()
    floating = P.copy()
    floating[:, 1] += 0.06
    r = Realism()
    out = None
    for _ in range(300):                               # 10 s standing still
        out, info = r.process(floating, np.ones(33), 1 / 30, cal.lengths)
    feet = [S.L_ANK, S.L_HEEL, S.L_TOE, S.R_ANK, S.R_HEEL, S.R_TOE]
    assert abs(out[feet, 1].min()) < 0.01


# ---------------------------------------------------------------- interpolation
def _kf(t, x, yaw):
    return Keyframe(t, [Tracker("hip", np.array([x, 1.0, 0.0]), rot_y(yaw))],
                    Tracker("head", np.array([x, 1.6, 0.0]), rot_y(yaw)))


def test_interpolation_midpoint():
    k = interpolate(_kf(0.0, 0.0, 0), _kf(0.1, 1.0, 90), 0.05)
    assert np.allclose(k.trackers[0].pos, [0.5, 1.0, 0.0])
    assert np.allclose(k.trackers[0].rot, rot_y(45), atol=1e-9)
    e = interpolate(_kf(0.0, 0.0, 0), _kf(0.1, 1.0, 0), 10.0)   # far future: clamped
    assert e.trackers[0].pos[0] <= 1.0 + 0.05 / 0.1 + 1e-9


def _listen():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(0.05)
    return s


def test_sender_upsamples_to_fixed_rate():
    po = pytest.importorskip("pythonosc.osc_message")
    sock = _listen()
    cfg = Config(vrchat_port=sock.getsockname()[1], send_rate=90, hold_time=1.0).validate()
    snd = OutputSender()
    snd.configure(cfg)
    snd.start()
    import threading
    t0 = time.monotonic()

    def camera():                                          # camera at 20 fps
        while time.monotonic() - t0 < 1.2:
            now = time.monotonic()
            snd.push(_kf(now, now - t0, 0))
            time.sleep(0.05)
    th = threading.Thread(target=camera)
    th.start()
    xs = []
    try:
        time.sleep(0.3)
        sock.setblocking(False)                            # drop start-up packets
        while True:
            try:
                sock.recv(4096)
            except BlockingIOError:
                break
        sock.settimeout(0.05)
        end = time.time() + 0.6
        while time.time() < end:
            try:
                m = po.OscMessage(sock.recv(4096))
                if m.address == "/tracking/trackers/1/position":
                    xs.append(m.params[0])
            except socket.timeout:
                pass
    finally:
        th.join()
        snd.stop()
        sock.close()
    assert snd.rate > 60                                   # ~90 packets/s from 20 fps input
    steps = np.diff(xs)
    assert len(xs) > 10 and np.all(steps > -1e-6)          # smooth, monotonic motion
    assert steps.max() < 0.03                              # no 50 ms jumps (would be 0.05)


def test_sender_stops_when_tracking_lost_and_when_paused():
    sock = _listen()
    cfg = Config(vrchat_port=sock.getsockname()[1], hold_time=0.2).validate()
    snd = OutputSender()
    snd.configure(cfg)
    snd.start()
    try:
        snd.push(_kf(time.monotonic(), 0.0, 0))
        time.sleep(0.5)                                    # older than hold_time
        while True:
            try:
                sock.recv(4096)
            except socket.timeout:
                break
        time.sleep(0.2)
        with pytest.raises(socket.timeout):
            sock.recv(4096)
        snd.paused = True
        snd.push(_kf(time.monotonic(), 0.0, 0))
        time.sleep(0.15)
        with pytest.raises(socket.timeout):
            sock.recv(4096)
    finally:
        snd.stop()
        sock.close()


# ---------------------------------------------------------------- PC only
def test_output_is_loopback_only():
    OscClient(9000).close()
    for bad in ("192.168.1.20", "8.8.8.8", "10.0.0.5"):
        with pytest.raises(ValueError):
            OscClient(9000, bad)
