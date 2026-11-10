import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import mathutil as M  # noqa: E402
from fbt.config import load_config, save_config  # noqa: E402
from fbt.filters import OneEuroFilter  # noqa: E402
from fbt.osc import encode_message  # noqa: E402
from fbt import skeleton as S  # noqa: E402


# ------------------------------------------------------------------ math
@pytest.mark.parametrize("seed", range(20))
def test_unity_euler_roundtrip(seed):
    rng = np.random.default_rng(seed)
    ex, ey, ez = rng.uniform(-170, 170, 3)
    ex = float(np.clip(ex, -85, 85))
    R = M.unity_euler_to_mat(ex, ey, ez)
    R2 = M.unity_euler_to_mat(*M.mat_to_unity_euler(R))
    assert np.allclose(R, R2, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
def test_quat_roundtrip(seed):
    rng = np.random.default_rng(100 + seed)
    R = M.axis_angle(rng.normal(size=3), rng.uniform(-math.pi, math.pi))
    assert np.allclose(M.quat_to_mat(M.mat_to_quat(R)), R, atol=1e-9)


def test_unity_quaternion_euler_convention():
    # Unity: Quaternion.Euler(0, 90, 0) * Vector3.forward == Vector3.right
    R = M.unity_euler_to_mat(0, 90, 0)
    assert np.allclose(R @ M.Z_AXIS, M.X_AXIS, atol=1e-9)
    # Quaternion.Euler(90, 0, 0) * Vector3.forward == Vector3.down
    assert np.allclose(M.unity_euler_to_mat(90, 0, 0) @ M.Z_AXIS, -M.Y_AXIS, atol=1e-9)


def test_rotation_between_including_opposite():
    for a, b in [((1, 0, 0), (0, 1, 0)), ((0, 0, 1), (0, 0, -1)), ((1, 0, 0), (-1, 0, 0))]:
        R = M.rotation_between(a, b)
        assert np.allclose(R @ np.array(a, float), b, atol=1e-9)
        assert abs(np.linalg.det(R) - 1) < 1e-9
    # 180 deg on a horizontal vector must keep "up" up
    R = M.rotation_between((1, 0, 0), (-1, 0, 0))
    assert np.allclose(R @ M.Y_AXIS, M.Y_AXIS, atol=1e-9)


def test_bases_are_proper_rotations():
    for B in (M.basis_from_right_up((1, 0.1, 0), (0, 1, 0.2)),
              M.basis_from_up_forward((0, 1, 0.3), (0.1, 0, 1)),
              M.basis_from_forward_up((0.2, 0, 1), (0, 1, 0))):
        assert np.allclose(B.T @ B, np.eye(3), atol=1e-9)
        assert abs(np.linalg.det(B) - 1) < 1e-9
    assert M.basis_from_right_up((0, 0, 0), (0, 1, 0)) is None


# --------------------------------------------------------------- filter
def test_one_euro_smooths_noise_and_tracks_steps():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    rng = np.random.default_rng(0)
    out = [f(np.array([0.0]) + rng.normal(0, 0.01, 1), i / 30) for i in range(60)]
    assert np.std(out[10:]) < 0.01
    for i in range(60, 120):
        y = f(np.array([1.0]), i / 30)
    assert abs(y[0] - 1.0) < 0.02


# ------------------------------------------------------------------ osc
def test_osc_encoding():
    msg = encode_message("/tracking/trackers/1/position", [1.0, 2.0, 3.0])
    assert len(msg) % 4 == 0
    assert msg.startswith(b"/tracking/trackers/1/position\x00")
    assert b",fff\x00" in msg
    assert msg[-12:] == struct.pack(">fff", 1.0, 2.0, 3.0)
    m2 = encode_message("/VMT/Room/Unity", [3, 7, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    assert b",iiffffffff\x00" in m2
    with pytest.raises(ValueError):
        encode_message("/x", [float("nan")])
    with pytest.raises(ValueError):
        encode_message("noslash", [])


def test_osc_decodes_with_python_osc_if_available():
    po = pytest.importorskip("pythonosc.osc_message")
    raw = encode_message("/a/b", [1, 2.5, "hi", True])
    m = po.OscMessage(raw)
    assert m.address == "/a/b"
    assert list(m.params) == [1, 2.5, "hi", True]


# --------------------------------------------------------------- config
def test_config_validation_and_roundtrip(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"vrchat_ip": "evil.example.com", "vrchat_port": 999999,
                             "fps": "abc", "rotate": "45", "model": "../../x",
                             "trackers": {"hip": True, "bogus": True},
                             "user_height": float("nan") if False else 99,
                             "unknown_key": 1}))
    c = load_config(p)
    assert not hasattr(c, "vrchat_ip")  # PC-only: no network address in the config at all
    assert c.vrchat_port == 65535
    assert c.fps == 30 and c.rotate == 0 and c.model == "full"
    assert "bogus" not in c.trackers and c.trackers["hip"]
    assert c.user_height == 2.5
    save_config(c, p)
    assert load_config(p).to_dict() == c.to_dict()


# ------------------------------------------------------------- skeleton
def standing_pose_mp(height=1.75):
    """Synthetic MediaPipe world landmarks: person standing, facing the camera.

    MediaPipe axes: x = image right (= person's LEFT), y = down, z = away from camera.
    """
    P = np.zeros((33, 3))
    hip_y = 0.0
    def put(i, left, x, y, z=0.0):
        # left side of the body appears on image right (+x)
        P[i] = [x if left else -x, y, z]
    H = height
    put(S.L_HIP, True, 0.10, hip_y); put(S.R_HIP, False, 0.10, hip_y)
    put(S.L_KNEE, True, 0.10, 0.25 * H); put(S.R_KNEE, False, 0.10, 0.25 * H)
    put(S.L_ANK, True, 0.10, 0.47 * H); put(S.R_ANK, False, 0.10, 0.47 * H)
    put(S.L_HEEL, True, 0.10, 0.50 * H, 0.05); put(S.R_HEEL, False, 0.10, 0.50 * H, 0.05)
    put(S.L_TOE, True, 0.10, 0.52 * H, -0.15); put(S.R_TOE, False, 0.10, 0.52 * H, -0.15)
    put(S.L_SH, True, 0.20, -0.30 * H); put(S.R_SH, False, 0.20, -0.30 * H)
    put(S.L_EL, True, 0.25, -0.12 * H); put(S.R_EL, False, 0.25, -0.12 * H)
    put(S.L_WR, True, 0.25, 0.02 * H); put(S.R_WR, False, 0.25, 0.02 * H)
    floor_y = 0.52 * H
    ear_y = floor_y - S.EAR_HEIGHT_RATIO * H
    put(S.L_EAR, True, 0.08, ear_y, 0.02); put(S.R_EAR, False, 0.08, ear_y, 0.02)
    P[S.NOSE] = [0, ear_y + 0.02, -0.10]
    return P


def test_axes_conversion_person_right_is_plus_x_and_forward_is_plus_z():
    U = S.mp_to_unity(standing_pose_mp())
    assert U[S.R_HIP, 0] > 0 and U[S.L_HIP, 0] < 0       # right side -> +X
    assert U[S.L_SH, 1] > U[S.L_HIP, 1]                    # shoulders above hips
    assert U[S.L_TOE, 2] > U[S.L_HEEL, 2]                  # toes point forward (+Z)
    assert U[S.NOSE, 2] > U[S.L_EAR, 2]


@pytest.mark.parametrize("pitch,yaw", [(0, 0), (15, 0), (-10, 30), (20, -60)])
def test_calibration_levels_and_scales_the_body(pitch, yaw):
    H = 1.80
    U = S.mp_to_unity(standing_pose_mp(1.60))  # model sees a "smaller" person
    tilt = M.rot_y(yaw) @ M.rot_x(pitch)       # camera tilted / turned
    raw = U @ tilt.T + np.array([0.3, 1.1, -2.0])
    cal = S.compute_calibration([raw], user_height=H)
    P = cal.apply(raw)
    assert abs(np.min(P[S.FOOT_IDS, 1])) < 1e-6                     # feet on floor
    ear_h = (P[S.L_EAR, 1] + P[S.R_EAR, 1]) / 2
    assert abs(ear_h / S.EAR_HEIGHT_RATIO - H) < 1e-3               # scaled to user height
    tr = S.compute_trackers(P, ["hip", "left_foot", "right_foot", "left_knee", "chest",
                                "left_elbow"])
    assert np.allclose(tr["hip"].rot, np.eye(3), atol=1e-6)        # upright, facing +Z
    assert np.allclose(tr["hip"].pos[[0, 2]], 0, atol=1e-6)
    for n, t in tr.items():
        assert t.rot is not None and abs(np.linalg.det(t.rot) - 1) < 1e-6, n
    assert tr["left_foot"].rot[2, 2] > 0.9                          # foot points forward
    # serialization round-trip
    cal2 = S.Calibration.from_dict(json.loads(json.dumps(cal.to_dict())))
    assert np.allclose(cal2.apply(raw), P)


def test_calibration_from_dict_rejects_garbage():
    assert S.Calibration.from_dict({"R": [1] * 9, "scale": 1, "offset": [0, 0, 0]}) is None
    assert S.Calibration.from_dict({"nope": 1}) is None
    assert S.Calibration.from_dict({"R": np.eye(3).ravel().tolist(), "scale": 1e9,
                                    "offset": [0, 0, 0]}) is None


def test_estimate_root_depth():
    world = standing_pose_mp(1.75)
    # project with a known camera: person 2.5 m away, centred
    vfov = 50.0
    f = 0.5 / math.tan(math.radians(vfov) / 2)
    aspect = 4 / 3
    depth = 2.5
    cam = world + np.array([0.0, 0.0, depth])
    img = np.zeros((33, 3))
    img[:, 0] = 0.5 + cam[:, 0] * f / cam[:, 2] / aspect
    img[:, 1] = 0.5 + cam[:, 1] * f / cam[:, 2]
    root = S.estimate_root(img, world, aspect, vfov)
    assert root is not None
    assert abs(-root[2] - depth) < 0.15
    assert abs(root[0]) < 0.05
