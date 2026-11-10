"""Tests for v2 features: auto-config, camera mode choice, accuracy helpers,
finger curls, SteamVR role writer, recommendations."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import autotune, steamvr  # noqa: E402
from fbt import skeleton as S  # noqa: E402
from fbt.camera import CameraMode, choose_best_mode  # noqa: E402
from fbt.config import Config, apply_auto, load_config, save_config  # noqa: E402
from fbt.hands import finger_curls, hand_crop  # noqa: E402
from fbt.pose import ModelManifest  # noqa: E402
from tests.test_core import standing_pose_mp  # noqa: E402


# ---------------------------------------------------------------- config
def test_apply_auto_respects_user_choices():
    c = Config().validate()
    c.mark_user_set(["model"])
    changed = apply_auto(c, {"model": "heavy", "fps": 60, "bogus": 1})
    assert changed == ["fps"] and c.model == "full" and c.fps == 60
    changed = apply_auto(c, {"model": "heavy"}, force=True)  # user accepted a recommendation
    assert c.model == "heavy" and "model" not in c.user_set


def test_v1_config_migrates(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"vrchat_port": 9001, "model": "lite", "first_run_done": True}))
    c = load_config(p)
    assert c.vrchat_port == 9001 and c.model == "lite"
    assert c.first_run_done is False and c.schema_version >= 2  # auto-setup offered once
    c.user_set = ["model", "nonexistent", 5]
    c.camera_profiles = {"0": {"modes": [{"width": 1, "x": float("nan")}]}, "bad": 3}
    save_config(c, p)
    c2 = load_config(p)
    assert c2.user_set == ["model"]
    assert "bad" not in c2.camera_profiles
    assert c2.camera_profiles["0"]["modes"][0]["x"] is None


# ---------------------------------------------------------------- camera
def test_choose_best_mode_prefers_fps_then_reasonable_size():
    modes = [CameraMode(1920, 1080, 30, 60, "MJPG"), CameraMode(1280, 720, 60, 60, "MJPG"),
             CameraMode(640, 480, 61, 60, "MJPG"), CameraMode(640, 480, 30, 60, "YUY2")]
    assert (choose_best_mode(modes, prefer_hd=True).width) == 1280
    best = choose_best_mode(modes, prefer_hd=False)
    assert best.fps >= 60
    assert choose_best_mode([CameraMode(1920, 1080, 5, 60, "YUY2"),
                             CameraMode(640, 480, 30, 60, "MJPG")]).width == 640
    assert choose_best_mode([]) is None


# --------------------------------------------------------------- accuracy
def _calibrated():
    raw = S.mp_to_unity(standing_pose_mp(1.7))
    cal = S.compute_calibration([raw], 1.75)
    return cal, cal.apply(raw)


def test_bone_lengths_are_enforced():
    cal, P = _calibrated()
    noisy = P.copy()
    noisy[S.L_ANK] += np.array([0.0, -0.08, 0.15])   # depth error stretches the shin
    fixed = S.apply_bone_lengths(noisy, cal.lengths)
    shin = np.linalg.norm(fixed[S.L_ANK] - fixed[S.L_KNEE])
    assert abs(shin - cal.lengths["shin"]) < 1e-9
    # untouched pose is (almost) unchanged
    assert np.abs(S.apply_bone_lengths(P, cal.lengths) - P).max() < 1e-6


def test_foot_lock_and_floor_clamp():
    _, P = _calibrated()
    fs = S.FootStabilizer()
    out = None
    for i in range(10):  # standing still -> foot gets planted
        Q = P.copy()
        Q[[S.L_ANK, S.L_HEEL, S.L_TOE], 0] += 0.003 * i  # slow slide (0.09 m/s)
        out = fs.apply(Q, 1 / 30, lock=True, clamp=True)
    assert fs.locked[0]
    assert abs(out[S.L_ANK, 0] - P[S.L_ANK, 0]) < 0.01      # slide suppressed
    Q = P.copy()
    Q[[S.R_ANK, S.R_HEEL, S.R_TOE], 1] -= 0.2               # below the floor
    out = S.FootStabilizer().apply(Q, 1 / 30, lock=False, clamp=True)
    assert out[[S.R_ANK, S.R_HEEL, S.R_TOE], 1].min() >= -1e-9
    # a real step releases the lock
    Q = P.copy()
    Q[[S.L_ANK, S.L_HEEL, S.L_TOE], 2] += 0.05
    fs.apply(Q, 1 / 30, lock=True, clamp=True)
    assert not fs.locked[0]


def test_calibration_rejects_partial_body():
    P = S.mp_to_unity(standing_pose_mp(1.7))
    P[:, 1] *= 0.05   # everything squashed -> head/feet not really visible
    with pytest.raises(ValueError):
        S.compute_calibration([P], 1.7)


def test_old_calibration_without_lengths_still_loads():
    cal, _ = _calibrated()
    d = cal.to_dict()
    d.pop("lengths")
    d.pop("root_motion")
    c2 = S.Calibration.from_dict(d)
    assert c2 is not None and c2.lengths == {} and c2.root_motion is False


def test_hand_pose_orientation():
    _, P = _calibrated()
    for side in ("left", "right"):
        t = S.compute_hand(P, side)
        assert t.rot is not None and abs(np.linalg.det(t.rot) - 1) < 1e-9


# ---------------------------------------------------------------- fingers
def _hand(curl_deg: float) -> np.ndarray:
    """Synthetic flat hand in the XY plane, fingers bent by curl_deg per joint."""
    pts = np.zeros((21, 3))
    bases = {1: -0.03, 5: -0.02, 9: 0.0, 13: 0.02, 17: 0.04}
    for b, x in bases.items():
        pos = np.array([x, 0.08 if b != 1 else 0.03, 0.0])
        pts[b] = pos
        ang = 0.0
        for j in range(1, 4):
            ang += np.radians(curl_deg)
            pos = pos + 0.025 * np.array([0, np.cos(ang), -np.sin(ang)])
            pts[b + j] = pos
    return pts


def test_finger_curls():
    assert finger_curls(_hand(0)).max() < 0.1
    c = finger_curls(_hand(80))
    assert c[1:].min() > 0.9
    mid = finger_curls(_hand(40))[1:]
    assert 0.35 < mid.mean() < 0.65


def test_hand_crop_inside_image():
    img = np.zeros((33, 3))
    img[:, :2] = 0.5
    img[S.L_WR, :2] = [0.98, 0.5]
    img[S.L_INDEX, :2] = [1.02, 0.46]
    img[S.L_PINKY, :2] = [1.02, 0.54]
    img[S.L_EL, :2] = [0.85, 0.6]
    crop = hand_crop(img, np.ones(33), "left", 640, 480)
    assert crop is not None
    x0, y0, s = crop
    assert x0 >= 0 and y0 >= 0 and x0 + s <= 640 and y0 + s <= 480
    vis = np.ones(33)
    vis[S.L_WR] = 0.0
    assert hand_crop(img, vis, "left", 640, 480) is None


# ---------------------------------------------------------------- models
def test_model_manifest_detects_tampering(tmp_path):
    m = ModelManifest(tmp_path / "models.json")
    f = tmp_path / "pose_landmarker_full.task"
    f.write_bytes(b"A" * 100)
    assert m.verify_or_record(f)        # first sight -> recorded
    assert m.verify_or_record(f)
    f.write_bytes(b"B" * 100)
    assert not m.verify_or_record(f)    # changed -> refused


# ---------------------------------------------------------------- steamvr
def test_assign_tracker_roles(tmp_path, monkeypatch):
    monkeypatch.setattr(steamvr, "is_running", lambda: False)
    f = tmp_path / "steamvr.vrsettings"
    f.write_text(json.dumps({"steamvr": {"x": 1}, "trackers": {"/devices/other/A": "TrackerRole_Chest"}}))
    backup = steamvr.assign_tracker_roles(["hip", "left_foot", "right_foot", "chest"], f)
    data = json.loads(f.read_text())
    assert data["steamvr"] == {"x": 1}
    t = data["trackers"]
    assert t["/devices/other/A"] == "TrackerRole_Chest"        # foreign entries untouched
    assert t["/devices/nexovarapose/Nexovara Pose-hip"] == "TrackerRole_Waist"
    assert t["/devices/nexovarapose/Nexovara Pose-left_foot"] == "TrackerRole_LeftFoot"
    assert t["/devices/nexovarapose/Nexovara Pose-right_foot"] == "TrackerRole_RightFoot"
    assert t["/devices/nexovarapose/Nexovara Pose-chest"] == "TrackerRole_Chest"
    assert json.loads(backup.read_text())["trackers"] == {"/devices/other/A": "TrackerRole_Chest"}


def test_assign_roles_refuses_when_running_or_broken(tmp_path, monkeypatch):
    f = tmp_path / "steamvr.vrsettings"
    f.write_text("{broken")
    monkeypatch.setattr(steamvr, "is_running", lambda: True)
    with pytest.raises(RuntimeError):
        steamvr.assign_tracker_roles(["hip"], f)
    monkeypatch.setattr(steamvr, "is_running", lambda: False)
    with pytest.raises(ValueError):
        steamvr.assign_tracker_roles(["hip"], f)
    assert f.read_text() == "{broken"
    assert not list(tmp_path.glob("*backup*"))


# ---------------------------------------------------------------- tuning
def test_recommend_picks_model_by_budget():
    cfg = Config().validate()
    prof = {"modes": [{"width": 1280, "height": 720, "fps": 60.0, "fps_requested": 60,
                       "fourcc": "MJPG"}]}
    bench = {"results": [
        {"variant": "lite", "delegate": "CPU", "ms": 6.0, "detect_rate": 1.0},
        {"variant": "full", "delegate": "CPU", "ms": 9.0, "detect_rate": 1.0},
        {"variant": "heavy", "delegate": "CPU", "ms": 35.0, "detect_rate": 1.0}],
        "hand_ms": 8.0}
    recs = {r.key: r.value for r in autotune.recommend(cfg, {"threads": 8}, prof, bench,
                                                       "steamvr_driver", False)}
    assert recs["model"] == "full"          # heavy is too slow for 60 fps with headroom
    assert recs["fps"] == 60 and recs["width"] == 1280
    assert recs["output_driver"] is True and recs["target"] == "steamvr_driver"
    slow = dict(bench, results=[dict(r, ms=r["ms"] * 6) for r in bench["results"]])
    recs = {r.key: r.value for r in autotune.recommend(cfg, {"threads": 4}, prof, slow,
                                                       "vrchat_osc", True)}
    assert recs["model"] == "lite" and recs["fingers"] is True and recs["hand_every"] >= 2
    # nobody in front of the camera -> heuristic by CPU threads
    nobody = dict(bench, results=[dict(r, detect_rate=0.0) for r in bench["results"]])
    recs = {r.key: r.value for r in autotune.recommend(cfg, {"threads": 4}, None, nobody,
                                                       "test", False)}
    assert recs["model"] == "lite"


def test_pc_profile_contains_no_names():
    p = autotune.pc_profile_from({"cpu": "Secret CPU Name", "threads": 8, "ram_gb": 16,
                                  "os": "X"}, {"results": [], "hand_ms": None})
    assert "Secret" not in json.dumps(p) and p["threads"] == 8
