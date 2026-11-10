"""v4.4.2: the body is glued to where the headset was when the camera took the
picture (no sliding when you move your head), steadier calibration heading."""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt.hmd import BodyAnchor, HmdState  # noqa: E402
from fbt.mathutil import rot_y, yaw_of  # noqa: E402
from fbt.sender import Keyframe, interpolate  # noqa: E402
from fbt.skeleton import Tracker  # noqa: E402

CAMERA_DELAY = 0.10          # picture -> keyframe (camera + network + filters)


def _lean(t):                 # head forward/back relative to the hips (leaning, nodding)
    return np.array([0.0, 0.0, 0.20 * math.sin(2 * math.pi * 0.7 * t)])


def _simulate(aligned: bool) -> float:
    state = HmdState()
    anchor = BodyAnchor()
    hips_world = np.array([1.0, 0.95, -0.5])
    feet = []
    for i in range(int(3.0 * 90)):
        t = i / 90.0
        head_world = hips_world + np.array([0.0, 0.7, 0.0]) + _lean(t)
        # the headset is in front of the head centre by 8 cm (looking +Z here)
        state.update(head_world + np.array([0, 0, 0.08]), np.eye(3), "driver", t=t)
        t_cap = math.floor((t - CAMERA_DELAY) * 30) / 30          # newest picture we have
        if t_cap < 0:
            continue
        head_cam = Tracker("head", np.array([0.0, 1.65, 0.0]) + _lean(t_cap), np.eye(3))
        foot = Tracker("left_foot", np.array([-0.1, 0.05, 0.0]), np.eye(3))
        hmd = state.at(t_cap, 0.5, now=t) if aligned else state.latest(0.5, now=t)
        trs, _, _ = anchor.apply([foot], head_cam, {}, hmd, 0.0, t)
        if t > 1.0:
            feet.append(trs[0].pos)
    return float(np.std(np.array(feet)[:, 2]))


def test_feet_stay_put_when_you_lean():
    old, new = _simulate(False), _simulate(True)
    print(f"foot sway while leaning: {old * 100:.1f} cm -> {new * 100:.1f} cm")
    assert new < 0.01 and new < old / 4


def test_history_interpolates_headset_pose():
    s = HmdState()
    for k in range(5):
        s.update(np.array([k * 0.1, 1.7, 0]), rot_y(10.0 * k), "driver", t=10 + k * 0.01)
    p = s.at(10.025, now=10.04)
    assert p.pos[0] == pytest.approx(0.25) and yaw_of(p.rot[:, 2]) == pytest.approx(25.0)
    assert s.at(9.0, now=10.04).pos[0] == pytest.approx(0.4)      # older: where it is now
    assert s.at(11.0, now=10.04).pos[0] == pytest.approx(0.4)     # newer: the newest
    assert s.at(10.0, now=12.0) is None                           # stale
    s.update(np.array([5.0, 1.7, 0]), np.eye(3), "openvr", t=10.6)   # another source
    assert s.at(10.02, now=10.6).pos[0] == pytest.approx(5.0)     # histories not mixed


def test_keyframes_carry_capture_time():
    a = Keyframe(1.0, [], None, capture_t=0.90)
    b = Keyframe(1.1, [], None, capture_t=0.95)
    assert interpolate(a, b, 1.05).capture_t == pytest.approx(0.925)
    assert interpolate(Keyframe(1.0, []), b, 1.05).capture_t is None


def test_calibration_heading_is_averaged(tmp_path):
    from fbt import engine as E
    from fbt.config import Config
    from tests.test_core import standing_pose_mp
    eng = E.Engine(Config().validate(), tmp_path, data_dir=tmp_path)
    cfg = Config(output_driver=True, hmd_anchor=True, output_vrchat=False).validate()
    rng = np.random.default_rng(0)
    eng.request_calibration(20)
    cal = None
    t0 = time.monotonic()
    for k in range(20):
        t = t0 + k / 300
        yaw = 30.0 + rng.normal(0, 4.0)          # the head wobbles a little
        eng.hmd_state.update(np.array([0, 1.7, 0]), rot_y(yaw), "driver", t=t)
        cal = eng._calibration_step(-standing_pose_mp(1.7) + rng.normal(0, 0.002, (33, 3)),
                                    cfg, None, t) or cal
    assert cal is not None and abs(cal.hmd_yaw - 30.0) < 2.5
    assert E.circular_mean_deg([179.0, -179.0]) == pytest.approx(180.0, abs=1e-6) or \
        E.circular_mean_deg([179.0, -179.0]) == pytest.approx(-180.0, abs=1e-6)
