"""v4: performance helpers, multi-camera fusion + rig calibration, own SteamVR
driver protocol (cross-checked against the C++ parser), driver management."""
import json
import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbt import steamvr, system  # noqa: E402
from fbt import skeleton as S  # noqa: E402
from fbt.config import Config  # noqa: E402
from fbt.mathutil import axis_angle, cross3, normalize, quat_to_mat, rot_y  # noqa: E402
from fbt.multicam import RigCalibrator, agreement, fuse, kabsch, rig_rotation  # noqa: E402
from fbt.outputs import (DEV_HMD, DEV_LEFT_HAND, DriverOutput, encode_driver_packet,  # noqa: E402
                         unity_to_openvr)
from fbt.pose import PoseResult, prepare_rgb  # noqa: E402
from fbt.skeleton import Tracker  # noqa: E402
from tests.test_core import standing_pose_mp  # noqa: E402

CPP_TEST = ROOT / "driver" / "build" / ("Release" if os.name == "nt" else "") / (
    "nexovarapose_protocol_test" + (".exe" if os.name == "nt" else ""))


# ---------------------------------------------------------------- performance
def test_fast_vector_math_matches_numpy():
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = rng.normal(size=3), rng.normal(size=3)
        assert np.allclose(cross3(a, b), np.cross(a, b))
        assert np.allclose(normalize(a), a / np.linalg.norm(a))
    assert normalize(np.zeros(3)) is None and normalize([np.nan, 0, 0]) is None


def test_prepare_rgb_downscales_and_converts():
    img = np.zeros((720, 1280, 3), np.uint8)
    img[..., 0] = 255                       # blue in BGR
    rgb = prepare_rgb(img)
    assert rgb.shape == (540, 960, 3) and rgb[0, 0, 2] == 255 and rgb[0, 0, 0] == 0
    small = np.zeros((480, 640, 3), np.uint8)
    assert prepare_rgb(small).shape == (480, 640, 3)


def test_precise_sleep_is_accurate():
    errs = []
    for _ in range(20):
        t0 = time.perf_counter()
        system.precise_sleep(0.005)
        errs.append(time.perf_counter() - t0 - 0.005)
    assert np.mean(errs) < 0.003 and min(errs) >= -1e-4


# ---------------------------------------------------------------- multi-camera
def _view(world, R_cam, rng, depth_sigma=0.08, lat_sigma=0.01, vis=1.0):
    """What a camera with orientation R_cam (main <- this camera) measures."""
    local = world @ R_cam                    # main frame -> this camera's frame
    noise = rng.normal(size=local.shape) * [lat_sigma, lat_sigma, depth_sigma]
    return PoseResult(local + noise, np.zeros((33, 3)), np.full(33, vis))


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(1)
    R = axis_angle([0.2, 1.0, 0.1], np.radians(75))
    src = rng.normal(size=(200, 3))
    dst = src @ R.T + rng.normal(size=src.shape) * 0.01
    assert np.allclose(kabsch(src, dst), R, atol=0.02)


def test_two_cameras_fix_depth_error():
    rng = np.random.default_rng(2)
    truth = standing_pose_mp(1.75)
    truth[S.L_WR, 2] -= 0.3                  # arm forward: depth matters
    R_aux = rot_y(90)                        # second camera from the side
    err_single, err_fused = [], []
    for _ in range(100):
        main = _view(truth, np.eye(3), rng)
        aux = _view(truth, R_aux, rng)
        err_single.append(np.linalg.norm(main.world - truth, axis=1).mean())
        world, vis = fuse([(np.eye(3), main, 1.0), (R_aux, aux, 1.0)])
        err_fused.append(np.linalg.norm(world - truth, axis=1).mean())
    assert np.mean(err_fused) < 0.4 * np.mean(err_single)   # >2.5x more accurate


def test_fusion_trusts_visible_camera_for_occluded_joint():
    rng = np.random.default_rng(3)
    truth = standing_pose_mp(1.7)
    main = _view(truth, np.eye(3), rng, depth_sigma=0.001, lat_sigma=0.001)
    main.world[S.L_ANK] += [0.4, 0.4, 0.4]   # garbage where main can't see
    main.visibility[S.L_ANK] = 0.05
    R = rot_y(-60)
    aux = _view(truth, R, rng, depth_sigma=0.001, lat_sigma=0.001)
    world, vis = fuse([(np.eye(3), main, 1.0), (R, aux, 1.0)])
    assert np.linalg.norm(world[S.L_ANK] - truth[S.L_ANK]) < 0.06
    assert vis[S.L_ANK] == 1.0


def test_agreement_rejects_wrong_person():
    rng = np.random.default_rng(4)
    me = standing_pose_mp(1.75)
    R = rot_y(45)
    main = _view(me, np.eye(3), rng)
    aux_me = _view(me, R, rng)
    other = standing_pose_mp(1.6)            # someone else, doing something else
    for j in (S.L_EL, S.L_WR, S.R_EL, S.R_WR):
        other[j, 1] -= 0.6                    # both arms raised
    for j in (S.L_KNEE, S.L_ANK, S.L_HEEL, S.L_TOE):
        other[j] += [0.0, -0.35, -0.35]       # one leg lifted forward
    aux_other = _view(other, R, rng)
    from fbt.multicam import AGREE_LIMIT_M
    assert agreement(R, aux_me, main) < AGREE_LIMIT_M < agreement(R, aux_other, main)


def test_rig_calibration_end_to_end():
    rng = np.random.default_rng(5)
    R_true = axis_angle([0.05, 1.0, -0.03], np.radians(-70))
    rig = RigCalibrator(["1"], seconds=0.01)
    base = standing_pose_mp(1.75)
    for i in range(80):                       # user turns and moves arms
        world = base @ rot_y(i * 3.0).T
        world[S.L_WR] += [0, -0.3 * np.sin(i / 7), 0.2]
        main = _view(world, np.eye(3), rng, depth_sigma=0.04)
        aux = _view(world, R_true, rng, depth_sigma=0.04)
        rig.add(main, {"1": aux})
    time.sleep(0.02)
    assert rig.done()
    res = rig.solve()["1"]
    R = rig_rotation(res)
    assert R is not None
    angle = np.degrees(np.arccos(np.clip((np.trace(R.T @ R_true) - 1) / 2, -1, 1)))
    assert angle < 3.0 and res["rms_cm"] < 15
    assert "error" in RigCalibrator(["2"], 0.0).solve()["2"]          # no data -> error
    assert rig_rotation({"R": [1, 0, 0, 0, 1, 0, 0, 0, 2]}) is None    # not a rotation


def test_engine_fuses_extra_camera(tmp_path):
    from fbt import engine as E

    class FakeWorker:
        def __init__(self):
            self.source = "1"
            self.state = type("St", (), {"fps": 30.0, "found": True, "people": 1, "error": None})()

    class FakeAux:
        def __init__(self, item):
            self.workers = [FakeWorker()]
            self.item = item

        def fresh(self, now):
            return {"1": self.item}

    rng = np.random.default_rng(6)
    truth = standing_pose_mp(1.75)
    R = rot_y(90)
    main = _view(truth, np.eye(3), rng)
    aux = _view(truth, R, rng)
    eng = E.Engine(Config().validate(), tmp_path)
    cfg = Config().validate()
    cfg.camera_rig = {"1": {"R": R.reshape(-1).tolist(), "rms_cm": 3.0}}
    res, info = eng._multicam(cfg, main, FakeAux((1.0, aux)), 1.02)
    assert info["fused"] == 2 and info["cameras"][0]["used"]
    assert np.linalg.norm(res.world - truth, axis=1).mean() < \
        np.linalg.norm(main.world - truth, axis=1).mean()
    cfg.camera_rig = {}                        # not calibrated -> not used
    res2, info2 = eng._multicam(cfg, main, FakeAux((1.0, aux)), 1.02)
    assert res2 is main and not info2["cameras"][0]["used"]


def test_extra_cameras_config_is_sanitised():
    c = Config(camera="0", extra_cameras=[{"source": "0"}, {"source": "1", "rotate": 45},
                                          {"source": "1"}, "junk", {"source": "2", "mirror": 1},
                                          {"source": "3"}, {"source": "4"}]).validate()
    assert [x["source"] for x in c.extra_cameras] == ["1", "2", "3"]
    assert c.extra_cameras[0]["rotate"] == 0 and c.extra_cameras[1]["mirror"] is True


# ---------------------------------------------------------------- own driver protocol
def test_unity_to_openvr_conversion():
    # a device turned to face the user's right (+X) must face +X in OpenVR too,
    # where a device's "forward" is its local -Z
    p, q = unity_to_openvr(np.array([0.1, 1.0, 0.5]), rot_y(90))
    assert p == (0.1, 1.0, -0.5)
    assert np.allclose(quat_to_mat(q) @ [0, 0, -1], [1, 0, 0], atol=1e-9)
    _, q0 = unity_to_openvr(np.zeros(3), np.eye(3))
    assert np.allclose(q0, [0, 0, 0, 1])


def _decode(packet: bytes):
    magic, ver, _f, seq, n = struct.unpack_from("<4sHHIB", packet)
    recs = [struct.unpack_from("<BBH3f4f5f", packet, 16 + i * 52) for i in range(n)]
    return magic, ver, seq, recs


def test_driver_packet_layout():
    t = Tracker("left_foot", np.array([0.1, 0.08, 0.2]), np.eye(3))
    h = Tracker("left_hand", np.array([-0.3, 1.0, 0.2]), np.eye(3))
    bad = Tracker("hip", np.array([np.nan, 0, 0]), np.eye(3))
    pkt = encode_driver_packet(42, [(1, t, None), (DEV_LEFT_HAND, h, np.array([0, .9, .5, .5, 2.0])),
                                    (0, bad, None)])
    magic, ver, seq, recs = _decode(pkt)
    assert magic == b"CFBT" and ver == 1 and seq == 42 and len(pkt) == 16 + 2 * 52
    assert len(recs) == 2                                   # NaN record dropped
    assert recs[0][0] == 1 and recs[0][1] == 1 and recs[0][5] == pytest.approx(-0.2)
    assert recs[1][1] == 3 and recs[1][-1] == pytest.approx(1.0)   # curls flag, clamped


@pytest.mark.skipif(not CPP_TEST.is_file(), reason="C++ driver tests not built")
def test_cpp_parser_reads_python_packets(tmp_path):
    t = Tracker("hip", np.array([0.05, 0.95, 0.1]), rot_y(30))
    h = Tracker("right_hand", np.array([0.3, 1.1, 0.25]), rot_y(-20))
    pkt = encode_driver_packet(7, [(0, t, None), (18, h, np.array([.1, .2, .3, .4, .5]))])
    f = tmp_path / "p.bin"
    f.write_bytes(pkt)
    out = subprocess.run([str(CPP_TEST), "--parse", str(f)], capture_output=True, text=True,
                         timeout=20).stdout.split("\n")
    assert out[0] == "seq 7"
    a = [float(x) for x in out[1].split()]
    pos, q = unity_to_openvr(t.pos, t.rot)
    assert a[0] == 0 and np.allclose(a[2:5], pos, atol=1e-5) and np.allclose(a[5:9], q, atol=1e-5)
    b = [float(x) for x in out[2].split()]
    assert b[0] == 18 and np.allclose(b[9:14], [.1, .2, .3, .4, .5], atol=1e-4)
    f.write_bytes(pkt[:-1])                                  # truncated -> rejected
    assert "INVALID" in subprocess.run([str(CPP_TEST), "--parse", str(f)],
                                       capture_output=True, text=True, timeout=20).stdout


def test_driver_output_sends_hmd_and_hands():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(1.0)
    out = DriverOutput(s.getsockname()[1], send_hmd=True, head_rotation=False)
    hip = Tracker("hip", np.array([0, 0.95, 0]), rot_y(40))
    head = Tracker("head", np.array([0, 1.6, 0]), rot_y(10) @ axis_angle([1, 0, 0], 0.3))
    hand = Tracker("left_hand", np.array([-0.3, 1.0, 0.2]), np.eye(3))
    try:
        out.send([hip], head, {"left": (hand, None)})
        _m, _v, _s, recs = _decode(s.recv(4096))
    finally:
        out.shutdown()
        s.close()
    ids = {r[0]: r for r in recs}
    assert set(ids) == {0, DEV_HMD, DEV_LEFT_HAND}
    hmd = ids[DEV_HMD]                     # (id, flags, reserved, x, y, z, qx, qy, qz, qw, curls…)
    q = hmd[6:10]
    R = quat_to_mat(q)
    fwd = R @ [0, 0, -1]                                     # OpenVR forward
    assert abs(fwd[1]) < 1e-6                                # level view (no head pitch)
    assert np.degrees(np.arctan2(fwd[0], -fwd[2])) == pytest.approx(40, abs=1e-4)  # body yaw
    assert hmd[4] > 1.6                                      # eyes above the ear midpoint


# ---------------------------------------------------------------- driver management
def _fake_steamvr(tmp_path, monkeypatch):
    rt = tmp_path / "SteamVR"
    (rt / "bin").mkdir(parents=True)
    log = tmp_path / "vrpathreg.log"
    script = rt / "bin" / ("vrpathreg.exe" if os.name == "nt" else "vrpathreg.sh")
    script.write_text(f"#!/bin/sh\necho \"$@\" >> '{log}'\n")
    script.chmod(0o755)
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    settings = cfgdir / "steamvr.vrsettings"
    settings.write_text(json.dumps({"steamvr": {"forcedDriver": "other"}}))
    paths = steamvr.OpenVRPaths([rt], [cfgdir], [])
    monkeypatch.setattr(steamvr, "find_paths", lambda: paths)
    monkeypatch.setattr(steamvr, "is_running", lambda: False)
    return log, settings


def test_no_hmd_mode_settings(tmp_path, monkeypatch):
    _, settings = _fake_steamvr(tmp_path, monkeypatch)
    steamvr.set_no_hmd_mode(True, settings)
    d = json.loads(settings.read_text())
    assert d["driver_nexovarapose"]["enableHmd"] is True
    assert d["steamvr"]["forcedDriver"] == "nexovarapose" and d["steamvr"]["activateMultipleDrivers"]
    assert steamvr.no_hmd_mode_enabled(settings)
    assert d["steamvr"]["requireHmd"] is False      # SteamVR must not wait for a real headset
    steamvr.set_no_hmd_mode(False, settings)
    d = json.loads(settings.read_text())
    # the previous choice of forced driver comes back
    assert d["driver_nexovarapose"]["enableHmd"] is False and d["steamvr"]["forcedDriver"] == "other"
    assert "requireHmd" not in d["steamvr"] and "nexovarapose_restore" not in d["driver_nexovarapose"]
    assert len(list(settings.parent.glob("*nexovarapose-backup*"))) >= 1


def test_driver_tracker_roles(tmp_path, monkeypatch):
    _, settings = _fake_steamvr(tmp_path, monkeypatch)
    steamvr.assign_tracker_roles(["hip", "left_foot"], settings)
    t = json.loads(settings.read_text())["trackers"]
    assert t == {"/devices/nexovarapose/Nexovara Pose-hip": "TrackerRole_Waist",
                 "/devices/nexovarapose/Nexovara Pose-left_foot": "TrackerRole_LeftFoot"}


def test_presets_prefer_own_driver(monkeypatch):
    from fbt import presets
    monkeypatch.setattr(presets, "running_apps", lambda: {"steamvr": True})
    monkeypatch.setattr(steamvr, "driver_registered", lambda: True)
    monkeypatch.setattr(steamvr, "no_hmd_mode_enabled", lambda *a: False)
    assert presets.detect_target()[0] == "steamvr_driver"
    monkeypatch.setattr(steamvr, "no_hmd_mode_enabled", lambda *a: True)
    assert presets.detect_target()[0] == "steamvr_nohmd"
    monkeypatch.setattr(steamvr, "driver_registered", lambda: False)
    assert presets.detect_target()[0] == "vrchat_osc"     # nothing for SteamVR -> OSC
