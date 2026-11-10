"""v4.4.3: no shaking from velocity guesses, held pose while out of view, robust head
when it leaves the picture, VRChat head = eyes, the lock keeps you when you come close."""
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import skeleton as S  # noqa: E402
from fbt.config import Config  # noqa: E402
from fbt.outputs import (FLAG_VELOCITY, VRChatOutput, VelocityTracker,  # noqa: E402
                         encode_driver_packet, eyes_of)
from fbt.person import PersonLock  # noqa: E402
from fbt.pose import PoseResult  # noqa: E402
from fbt.sender import Keyframe, OutputSender  # noqa: E402
from fbt.skeleton import HeadModel, Tracker  # noqa: E402
from tests.test_core import standing_pose_mp  # noqa: E402
from tests.test_person_realism import RED, frame_with, person  # noqa: E402


def test_velocity_is_smooth_and_zero_when_standing():
    v = VelocityTracker()
    rng = np.random.default_rng(1)
    out = []
    for i in range(90):                       # walking 1 m/s, rendered times jitter-free
        t = i / 90
        out.append(v("hip", np.array([t, 1.0, 0.0]), t))
    assert abs(out[-1][0] - 1.0) < 0.02
    v.reset()
    for i in range(90):                       # standing, sub-millimetre wobble
        p = np.array([0.0, 1.0, 0.0]) + rng.normal(0, 0.0003, 3)
        z = v("hip", p, i / 90)
    assert np.linalg.norm(z) < 0.01
    big = VelocityTracker()
    big("hip", np.zeros(3), 0.0)
    assert np.linalg.norm(big("hip", np.array([5.0, 0, 0]), 0.011)) <= 6.0 + 1e-9


def test_velocity_goes_to_the_driver_only_for_trackers():
    hip = Tracker("hip", np.array([0.1, 1.0, 0.2]), np.eye(3))
    pkt = encode_driver_packet(1, [(0, hip, None, None, np.array([0.5, 0.0, 1.0]))])
    flags = pkt[16 + 1]
    assert flags & FLAG_VELOCITY
    import struct
    vals = struct.unpack("<3f4f5f", pkt[16 + 4:16 + 52])
    assert vals[7] == pytest.approx(0.5) and vals[9] == pytest.approx(-1.0)   # Unity -> OpenVR
    plain = encode_driver_packet(1, [(0, hip, None)])
    assert not plain[17] & FLAG_VELOCITY


def test_head_rebuilt_from_shoulders_when_cut_off():
    P = -standing_pose_mp(1.7)
    P[:, 1] -= P[:, 1].min()
    vis, img = np.ones(33), np.full((33, 3), 0.5)
    hm = HeadModel()
    good = S.compute_head(P)
    for _ in range(60):
        hm(good, P, vis, img, 1 / 30)                # learn the shoulders-to-head distance
    wild = Tracker("head", good.pos + np.array([0.0, 0.6, -0.8]), good.rot)   # guessed head
    img_cut = img.copy()
    img_cut[[0, 7, 8], 1] = -0.1                     # head above the picture
    for _ in range(30):
        h = hm(wild, P, vis, img_cut, 1 / 30)
    assert np.linalg.norm(h.pos - good.pos) < 0.05
    for _ in range(60):
        h = hm(good, P, vis, img, 1 / 30)            # seen again: back to the camera's head
    assert np.allclose(h.pos, good.pos)


def test_vrchat_gets_the_eyes_not_the_middle_of_the_head():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.5)
    out = VRChatOutput(rx.getsockname()[1])
    head = Tracker("head", np.array([0.0, 1.6, 0.0]), np.eye(3))
    out.send([], head)
    po = pytest.importorskip("pythonosc.osc_message")
    m = po.OscMessage(rx.recv(4096))
    rx.close()
    out.shutdown()
    assert m.address == "/tracking/trackers/head/position"
    assert np.allclose(m.params, [0.0, 1.63, 0.09], atol=1e-6)     # eyes: forward + up
    assert np.allclose(eyes_of(head), [0.0, 1.63, 0.09])


def test_lock_keeps_you_while_you_walk_up_to_the_camera():
    lock = PersonLock()
    lock.select([person(0.5)], frame_with([(person(0.5), RED)]), 0.0)
    for i in range(1, 50):
        scale = 1.0 + i * 0.035                       # coming closer: bigger every frame
        p = person(0.5 + 0.002 * i, scale=scale, legs=1.0)
        vis = p.visibility.copy()
        vis[25:] = 0.1                                # legs out of the picture
        p = PoseResult(p.world * (1 + 0.1 * (i > 25)), p.image, vis)   # proportions look off
        col = (200 - 3 * i, 30 + 2 * i, 30)           # light changes the colours
        chosen, info = lock.select([p], frame_with([(p, col)]), i / 30, relock_single_s=3.0)
        assert chosen is not None, (i, info)


def test_incoherent_body_is_dropped():
    from fbt.engine import body_is_coherent
    P = -standing_pose_mp(1.7)
    tr = S.compute_trackers(P, ["hip", "left_foot", "right_foot"])
    head = S.compute_head(P)
    assert body_is_coherent(P, tr, head)
    far = dict(tr)
    far["left_foot"] = Tracker("left_foot", tr["left_foot"].pos + np.array([0, 0, 40.0]), None)
    assert not body_is_coherent(P, far, head)
    assert not body_is_coherent(P, tr, Tracker("head", head.pos + np.array([0, 5.0, 0]), None))


def test_sender_holds_the_last_pose_while_you_are_out_of_view():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.3)
    cfg = Config(output_vrchat=False, output_driver=True, driver_port=rx.getsockname()[1],
                 hmd_anchor=False, hold_time=0.1, lost_hold_s=1.0).validate()
    s = OutputSender()
    s.configure(cfg)
    s.start()
    try:
        hip = Tracker("hip", np.array([0.0, 1.0, 0.0]), np.eye(3))
        s.push(Keyframe(time.monotonic(), [hip], None))
        time.sleep(0.5)                               # well past hold_time
        got = 0
        end = time.time() + 0.3
        while time.time() < end:
            try:
                rx.recv(4096)
                got += 1
            except socket.timeout:
                break
        assert got > 5 and s.holding                  # still sent: trackers stay in the game
        time.sleep(1.0)                               # past lost_hold_s
        while True:
            try:
                rx.recv(4096)
            except socket.timeout:
                break
        assert not s.holding
    finally:
        s.stop()
        rx.close()


def test_floor_survives_a_moment_out_of_view():
    from fbt.realism import Realism
    r = Realism()
    r.floor = 0.12
    r.reset(keep_floor=True)
    assert r.floor == 0.12
    r.reset()
    assert r.floor == 0.0
