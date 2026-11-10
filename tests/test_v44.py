"""v4.4: confidence + advice, posture smoothing, profiles, recording/replay/tuning,
automatic camera rig calibration, VRChat headset pose (OSCQuery), RTMPose add-on."""
import json
import socket
import struct
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fbt import guidance as G  # noqa: E402
from fbt import multicam as MC  # noqa: E402
from fbt import oscquery as OQ  # noqa: E402
from fbt import profiles as PR  # noqa: E402
from fbt import recording as REC  # noqa: E402
from fbt import rtmpose as RT  # noqa: E402
from fbt.config import Config, load_config, save_config  # noqa: E402
from fbt.filters import PoseFilter  # noqa: E402
from fbt.hmd import HmdState  # noqa: E402
from fbt.mathutil import rot_x, rot_y, unity_euler_to_mat, yaw_of  # noqa: E402
from fbt.pose import PoseResult  # noqa: E402
from tests.test_core import standing_pose_mp  # noqa: E402


def _person(rng=None, noise=0.0, move=0.0, t=0.0):
    w = standing_pose_mp(1.7).copy()
    if move:
        # arms and legs swing (hip-centred like MediaPipe)
        for i in (13, 15, 17, 19, 21):
            w[i, 2] += move * np.sin(2.0 * t)
        for i in (14, 16, 18, 20, 22):
            w[i, 2] -= move * np.sin(2.0 * t)
        for i in (25, 27, 29, 31):
            w[i, 2] += 0.5 * move * np.sin(1.5 * t)
    if rng is not None and noise:
        w = w + rng.normal(0, noise, w.shape)
    img = np.zeros((33, 3))
    img[:, 0] = 0.5 + w[:, 0] * 0.35
    img[:, 1] = 0.5 + w[:, 1] * 0.35
    return PoseResult(w, img, np.ones(33))


# ---------------------------------------------------------------- confidence / advice
def test_confidence_reflects_visibility_frame_and_status():
    r = _person()
    assert G.raw_confidence("hip", r.visibility, r.image) == pytest.approx(1.0)
    vis = r.visibility.copy()
    vis[[27, 29, 31]] = 0.2
    assert G.raw_confidence("left_foot", vis, r.image) == pytest.approx(0.2)
    img = r.image.copy()
    img[[28, 30, 32], 1] = 1.05                       # right foot below the picture
    assert G.raw_confidence("right_foot", r.visibility, img) < 0.05
    assert G.raw_confidence("left_knee", r.visibility, r.image, "estimated") <= 0.35
    assert G.raw_confidence("left_knee", r.visibility, r.image, "held") <= 0.2
    c = G.TrackerConfidence()
    a = c.update(["hip"], r.visibility, r.image, {}, 1, 0.0)["hip"]
    b = c.update(["hip"], vis * 0 + 0.0, r.image, {}, 1, 0.033)["hip"]
    assert a == pytest.approx(1.0) and 0.8 < b < 1.0      # smoothed, no jump


def test_advice_and_debounce():
    r = _person()
    assert G.advice(r.image, r.visibility, 4 / 3, 120, 30, True, True,
                    {"hip": 1.0}) == []
    img = r.image.copy()
    img[[27, 28, 29, 30, 31, 32], 1] = 1.02
    keys = [k for k, _ in G.advice(img, r.visibility, 4 / 3, 120, 30, True, True, {})]
    assert keys[0] == "feet"
    small = r.image.copy()
    small[:, :2] = 0.5 + (small[:, :2] - 0.5) * 0.4
    keys = [k for k, _ in G.advice(small, r.visibility, 4 / 3, 120, 30, True, True, {})]
    assert "far" in keys
    keys = [k for k, _ in G.advice(None, None, 4 / 3, 20, 12, True, True, {})]
    assert keys == ["dark", "fps"]
    g = G.Guide()
    items = [("dark", "темно")]
    assert g.update(items, 0.0) == []
    assert g.update(items, 0.5) == []
    assert g.update(items, 1.0) == ["темно"]             # after SHOW_AFTER
    assert g.update([], 1.5) == ["темно"]                # kept a little longer
    assert g.update([], 5.0) == []


# ---------------------------------------------------------------- posture smoothing
def test_posture_smoothing_calms_legs_only_when_sitting():
    rng = np.random.default_rng(3)
    base = -standing_pose_mp(1.7)

    def jitter(sit):
        f = PoseFilter(1.2, 1.5)
        f.set_posture(sit, 0.0)
        out = [f(base + rng.normal(0, 0.01, base.shape), i / 30, np.ones(33))
               for i in range(240)]
        o = np.array(out[60:])
        return np.sqrt(np.mean(np.sum(np.diff(o[:, 25:33], axis=0) ** 2, axis=2))), \
            np.sqrt(np.mean(np.sum(np.diff(o[:, 15:23], axis=0) ** 2, axis=2)))
    legs_stand, hands_stand = jitter(0.0)
    legs_sit, hands_sit = jitter(1.0)
    assert legs_sit < legs_stand * 0.85
    assert hands_sit == pytest.approx(hands_stand, rel=0.25)
    f = PoseFilter()
    f.set_posture(0.0, 0.0)
    assert np.allclose(f.softness, 1.0)                  # standing: exactly as before


# ---------------------------------------------------------------- profiles
def test_profiles_save_apply_and_validate(tmp_path):
    c = Config().validate()
    c.user_height, c.target = 1.85, "vrchat_osc"
    c.mark_user_set(["user_height"])
    c.calibration = {"R": np.eye(3).reshape(-1).tolist(), "scale": 1.0,
                     "offset": [0, 0, 0], "lengths": {}}
    PR.save_profile(c, "  Аватар 1  ")
    assert "Аватар 1" in c.profiles and c.active_profile == "Аватар 1"
    c.user_height, c.target, c.calibration = 1.6, "steamvr_driver", None
    c.user_set = ["target"]
    assert PR.differs(c)
    changed = PR.apply_profile(c, "Аватар 1")
    assert set(changed) >= {"user_height", "target", "calibration"}
    assert c.user_height == 1.85 and c.target == "vrchat_osc" and c.calibration
    assert c.user_set == ["user_height"]
    PR.rename_profile(c, "Аватар 1", "Brother")
    assert c.active_profile == "Brother"
    p = tmp_path / "config.json"
    save_config(c, p)
    c2 = load_config(p)
    assert c2.profiles["Brother"]["user_height"] == 1.85
    # hostile profile data is cleaned like the rest of the config
    data = json.loads(p.read_text(encoding="utf-8"))
    data["profiles"] = {"x" * 99: {"user_height": 99, "evil": "rm -rf", "target": "nope"},
                        "ok": "not a dict"}
    data["active_profile"] = "missing"
    p.write_text(json.dumps(data), encoding="utf-8")
    c3 = load_config(p)
    name = next(iter(c3.profiles))
    assert len(name) <= 40 and c3.profiles[name]["user_height"] == 2.5
    assert "evil" not in c3.profiles[name] and c3.profiles[name]["target"] == "auto"
    assert c3.active_profile == ""
    PR.delete_profile(c, "Brother")
    assert c.profiles == {} and c.active_profile == ""


# ---------------------------------------------------------------- recording
def _make_recording(tmp_path, seconds=40.0, fps=30.0, still_until=15.0, seed=0):
    rng = np.random.default_rng(seed)
    rec = REC.Recorder(tmp_path / "recordings" / "r.npz", {"aspect": 16 / 9})
    for i in range(int(seconds * fps)):
        t = i / fps
        rec.add(100.0 + t, _person(rng, 0.006, 0.0 if t < still_until else 0.25, t))
    return rec.save()


def test_recording_roundtrip_and_strict_loading(tmp_path):
    path = _make_recording(tmp_path, seconds=3.0)
    r = REC.load_recording(path)
    assert r.frames == 90 and 2.9 < r.seconds < 3.0 and abs(r.fps - 30) < 0.5
    assert r.result(5).world.shape == (33, 3) and r.meta["aspect"] == pytest.approx(16 / 9)
    assert REC.list_recordings(tmp_path) == [path]
    bad = tmp_path / "recordings" / "bad.npz"
    np.savez(bad, t=np.arange(3.0), world=np.zeros((3, 33, 3)), image=np.zeros((3, 33, 3)),
             vis=np.zeros((3, 33)), meta=np.array([{"a": 1}], dtype=object))
    with pytest.raises(ValueError):
        REC.load_recording(bad)                         # object arrays are refused
    np.savez(bad, t=np.arange(3.0), world=np.full((3, 33, 3), np.nan),
             image=np.zeros((3, 33, 3)), vis=np.zeros((3, 33)), meta=np.array("{}"))
    with pytest.raises(ValueError):
        REC.load_recording(bad)
    np.savez(bad, x=np.zeros(3))
    with pytest.raises(ValueError):
        REC.load_recording(bad)


def test_replay_source_and_estimator(tmp_path):
    r = REC.load_recording(_make_recording(tmp_path, seconds=1.0))
    src = REC.ReplaySource(r)
    est = REC.ReplayEstimator(src)
    src.start()
    try:
        got, seq = [], 0
        t_end = time.time() + 2.0
        while len(got) < 10 and time.time() < t_end:
            item = src.get_latest(seq, 0.2)
            if item is None:
                continue
            seq = item[0]
            est.seq = seq
            got.append(est.detect(item[1], 0))
        assert len(got) >= 10 and all(len(g) == 1 for g in got)
        assert item[1].shape[1] == 640 and abs(item[1].shape[0] - 360) <= 1
    finally:
        src.stop()


def test_tune_filter_suggests_only_real_improvements(tmp_path):
    r = REC.load_recording(_make_recording(tmp_path, seconds=40.0))
    res = REC.tune_filter(r, 1.2, 1.5, True)
    assert res.ok
    if res.better:
        assert res.jitter_new <= res.jitter_old * 1.10 and res.error_new <= res.error_old * 1.10
        assert res.jitter_new / res.jitter_old + res.error_new / res.error_old < 1.9
    # a terrible starting point is always improved
    bad = REC.tune_filter(r, 20.0, 50.0, True)
    assert bad.ok and bad.better and bad.jitter_new < bad.jitter_old
    short = REC.load_recording(_make_recording(tmp_path, seconds=2.0))
    assert not REC.tune_filter(short, 1.2, 1.5).ok


def test_engine_replays_a_recording(tmp_path):
    from fbt import engine as E
    path = _make_recording(tmp_path, seconds=3.0)
    drv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    drv.bind(("127.0.0.1", 0))
    drv.settimeout(0.3)
    cfg = Config(camera=str(path), output_vrchat=False, output_driver=True,
                 driver_port=drv.getsockname()[1], hmd_anchor=False,
                 target="test").validate()
    cfg.mark_user_set(["output_vrchat", "output_driver", "driver_port", "hmd_anchor"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    try:
        pk = []
        end = time.time() + 3.0
        while time.time() < end and len(pk) < 20:
            try:
                pk.append(drv.recv(65535))
            except socket.timeout:
                pass
        errors = [p for k, p in eng.poll_events() if k == "error"]
        assert not errors, errors
        assert len(pk) >= 20 and pk[-1][:4] == b"CFBT"
        snap = eng.snapshot()
        assert snap["tracking"] and snap["stats"]["replay"]
        assert snap["confidence"].get("hip", 0) > 0.5
        assert not any("света" in h for h in snap["hints"])     # the backdrop isn't "dark"
        eng.start_recording()          # not while replaying: refused, no crash
        time.sleep(0.2)
    finally:
        eng.stop()
        drv.close()


def test_engine_records_pose_numbers(tmp_path, monkeypatch):
    from fbt import engine as E
    from tests.test_engine import FakeCamera, FakeEstimator
    monkeypatch.setattr(E, "CameraSource", FakeCamera)
    monkeypatch.setattr(E, "PoseEstimator", FakeEstimator)
    monkeypatch.setattr(E, "ensure_model", lambda *a, **k: tmp_path / "m.task")
    cfg = Config(output_vrchat=False, target="test", width=640, height=480, fps=30,
                 fourcc="MJPG").validate()
    cfg.mark_user_set(["output_vrchat", "width", "height", "fps", "fourcc"])
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng.start()
    try:
        time.sleep(0.3)
        eng.start_recording()
        time.sleep(0.8)
        assert eng.snapshot()["stats"]["recording"] is not None
        eng.stop_recording()
        time.sleep(0.3)
        ev = [p for k, p in eng.poll_events() if k == "recording"]
    finally:
        eng.stop()
    assert ev[0]["state"] == "started" and ev[-1]["state"] == "saved"
    r = REC.load_recording(Path(ev[-1]["path"]))
    assert r.frames > 20 and Path(ev[-1]["path"]).parent == tmp_path / "recordings"


# ---------------------------------------------------------------- automatic rig calibration
def _aux_view(res, R_aux_to_main, rng, noise=0.004):
    # aux camera sees the same hip-centred body rotated: world_aux = world_main @ R
    w = res.world @ R_aux_to_main + rng.normal(0, noise, res.world.shape)
    return PoseResult(w, res.image.copy(), np.ones(33))


def test_auto_rig_solves_rotation_from_ordinary_moves():
    rng = np.random.default_rng(5)
    R_true = rot_y(70.0) @ rot_x(8.0)
    ar = MC.AutoRig()
    entry = None
    for i in range(120):
        main = _person(rng, 0.003, 0.3, i / 30)
        ar.add("1", main, _aux_view(main, R_true, rng), crowded=False)
        entry = entry or ar.step("1", None, i / 30)
    assert entry is not None and entry["auto"] and entry["rms_cm"] < 6
    R = np.array(entry["R"]).reshape(3, 3)
    assert MC.rotation_angle_deg(R, R_true) < 2.0


def test_auto_rig_refuses_bad_data():
    rng = np.random.default_rng(6)
    R_true = rot_y(40.0)
    ar = MC.AutoRig()
    for i in range(120):                                  # standing perfectly still
        main = _person(rng, 0.002)
        ar.add("1", main, _aux_view(main, R_true, rng), crowded=False)
        assert ar.step("1", None, i / 30) is None
    ar = MC.AutoRig()
    for i in range(120):                                  # two different people
        main = _person(rng, 0.003, 0.3, i / 30)
        other = _person(rng, 0.003, 0.3, i / 30 + 1.3)
        ar.add("1", main, _aux_view(other, R_true, rng, 0.03), crowded=False)
        assert ar.step("1", None, i / 30) is None
    ar = MC.AutoRig()
    for i in range(120):                                  # several people in view
        main = _person(rng, 0.003, 0.3, i / 30)
        ar.add("1", main, _aux_view(main, R_true, rng), crowded=True)
        assert ar.step("1", None, i / 30) is None


def test_auto_rig_resolves_a_moved_camera():
    rng = np.random.default_rng(7)
    R_old, R_new = rot_y(60.0), rot_y(95.0)
    ar = MC.AutoRig()
    entry = None
    for i in range(400):
        t = i / 30
        main = _person(rng, 0.003, 0.3, t)
        aux = _aux_view(main, R_new, rng)
        ar.add("1", main, aux, crowded=False)
        ar.note_agreement("1", MC.agreement(R_old, aux, main), t)
        e = ar.step("1", R_old, t)
        if e is not None:
            entry = e
            break
    assert entry is not None
    assert MC.rotation_angle_deg(np.array(entry["R"]).reshape(3, 3), R_new) < 2.0
    # a calibrated camera that agrees is never touched
    ar = MC.AutoRig()
    for i in range(200):
        main = _person(rng, 0.003, 0.3, i / 30)
        aux = _aux_view(main, R_new, rng)
        ar.add("1", main, aux, crowded=False)
        ar.note_agreement("1", MC.agreement(R_new, aux, main), i / 30)
        assert ar.step("1", R_new, i / 30) is None


# ---------------------------------------------------------------- VRChat OSCQuery
def _osc(addr, *floats):
    def s(x):
        b = x.encode() + b"\x00"
        return b + b"\x00" * ((-len(b)) % 4)
    return s(addr) + s("," + "f" * len(floats)) + b"".join(struct.pack(">f", f)
                                                            for f in floats)


def test_osc_parsing_is_strict():
    m = _osc(OQ.HEAD_POSE, 1, 2, 3, 10, 20, 30)
    (addr, args), = OQ.parse_packet(m)
    assert addr == OQ.HEAD_POSE and args == pytest.approx([1, 2, 3, 10, 20, 30])
    bundle = b"#bundle\x00" + b"\x00" * 8 + struct.pack(">i", len(m)) + m \
        + struct.pack(">i", len(m)) + m
    assert len(OQ.parse_packet(bundle)) == 2
    for bad in (b"", b"abcd", m[:-3], b"#bundle\x00" + b"\x00" * 8 + struct.pack(">i", 999),
                _osc("/x").replace(b",", b"?"), b"x" * 5000):
        with pytest.raises(Exception):
            OQ.parse_packet(bad)
    pos, R = OQ.pose_from_args([1.0, 1.6, 0.5, 0.0, 90.0, 0.0])
    assert np.allclose(pos, [1, 1.6, 0.5]) and abs(yaw_of(R[:, 2]) - 90) < 1e-6
    assert np.allclose(R, unity_euler_to_mat(0, 90, 0))
    assert OQ.pose_from_args([1.0] * 5) is None
    assert OQ.pose_from_args([float("nan")] + [0.0] * 5) is None


def test_oscquery_service_http_and_udp():
    st = HmdState()
    svc = OQ.OscQueryService(st, name="Nexovara Pose-TEST", advertise=False)
    svc.start()
    try:
        base = f"http://127.0.0.1:{svc.http_port}"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        info = json.loads(opener.open(base + "/?HOST_INFO", timeout=3).read())
        assert info["NAME"] == "Nexovara Pose-TEST" and info["OSC_PORT"] == svc.osc_port
        assert info["OSC_IP"] == "127.0.0.1" and info["OSC_TRANSPORT"] == "UDP"
        root = json.loads(opener.open(base + "/", timeout=3).read())
        node = root["CONTENTS"]["tracking"]["CONTENTS"]["vrsystem"]["CONTENTS"]["head"][
            "CONTENTS"]["pose"]
        assert node["FULL_PATH"] == OQ.HEAD_POSE and node["TYPE"] == "ffffff"
        assert node["ACCESS"] == 2
        sub = json.loads(opener.open(base + "/tracking/vrsystem", timeout=3).read())
        assert sub["FULL_PATH"] == "/tracking/vrsystem"
        with pytest.raises(urllib.error.HTTPError):
            opener.open(base + "/nope", timeout=3)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.sendto(b"garbage", ("127.0.0.1", svc.osc_port))
        s.sendto(_osc(OQ.HEAD_POSE, 0.2, 1.6, -0.3, 0.0, 45.0, 0.0), ("127.0.0.1", svc.osc_port))
        s.close()
        end = time.time() + 2
        while st.latest(1.0) is None and time.time() < end:
            time.sleep(0.02)
        p = st.latest(1.0)
        assert p is not None and p.source == "vrchat" and np.allclose(p.pos, [0.2, 1.6, -0.3],
                                                                       atol=1e-6)
        assert abs(yaw_of(p.rot[:, 2]) - 45) < 1e-4 and svc.poses == 1
    finally:
        svc.stop()


def test_oscquery_mdns_announcement():
    pytest.importorskip("zeroconf")
    svc = OQ.OscQueryService(HmdState(), name="Nexovara Pose-T" + str(int(time.time()) % 1000))
    svc.start()
    try:
        end = time.time() + 15
        while svc._zc is None and time.time() < end and "не удалось" not in svc.status:
            time.sleep(0.1)
        if svc._zc is None:
            pytest.skip("no multicast here: " + svc.status)
        info = svc._zc.get_service_info("_oscjson._tcp.local.",
                                        f"{svc.name}._oscjson._tcp.local.", timeout=3000)
        assert info is not None and info.port == svc.http_port
        assert socket.inet_aton("127.0.0.1") in info.addresses
    finally:
        svc.stop()


def test_vrchat_output_glued_to_vrchat_headset(tmp_path):
    po = pytest.importorskip("pythonosc.osc_message")
    from fbt.sender import Keyframe, OutputSender
    from fbt.skeleton import Tracker
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(0.2)
    cfg = Config(output_vrchat=True, vrchat_port=rx.getsockname()[1], vrchat_hmd=True,
                 output_smoothing=False, output_driver=False).validate()
    s = OutputSender()
    s.configure(cfg)
    s.start()
    try:
        end = time.time() + 2
        while s.oscquery is None and time.time() < end:
            s.push(Keyframe(time.monotonic(), [], None))
            time.sleep(0.02)
        assert s.oscquery is not None
        R_h = unity_euler_to_mat(0, 30, 0)
        s.vrc_hmd_state.update(np.array([2.0, 1.7, 1.0]), R_h, "vrchat")
        hip = Tracker("hip", np.array([0.0, 1.0, 0.0]), np.eye(3))
        head = Tracker("head", np.array([0.05, 1.6, 0.02]), np.eye(3))
        msgs = []
        for i in range(10):
            s.vrc_hmd_state.update(np.array([2.0, 1.7, 1.0]), R_h, "vrchat")
            s.push(Keyframe(time.monotonic(), [hip], head, vrc_yaw=30.0))
            time.sleep(0.03)
            try:
                while True:
                    msgs.append(po.OscMessage(rx.recv(4096)))
            except socket.timeout:
                pass
        heads = [m.params for m in msgs if m.address == "/tracking/trackers/head/position"]
        rots = [m.params for m in msgs if m.address == "/tracking/trackers/head/rotation"]
        target = np.array([2.0, 1.7, 1.0]) - R_h[:, 2] * 0.08       # head centre
        assert heads and np.allclose(heads[-1], [2.0, 1.7, 1.0], atol=1e-4)  # = VRChat's HMD
        assert len(rots) >= 5 and abs(rots[-1][1] - 30.0) < 1e-3       # continuous heading
        hips = [m.params for m in msgs if m.address == "/tracking/trackers/1/position"]
        # body glued under the headset: hip ~0.6 m below the head
        assert abs(hips[-1][1] - (target[1] - 0.6)) < 0.1
        assert s.vrc_anchored
    finally:
        s.stop()
        rx.close()
    assert s.oscquery is None


# ---------------------------------------------------------------- RTMPose add-on
def _gauss(L, mu, sigma=4.0, peak=0.9):
    x = np.arange(L)
    return peak * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def test_simcc_decoding_is_sub_bin_precise():
    mus_x = np.linspace(20.3, 350.7, 26)
    mus_y = np.linspace(30.6, 480.2, 26)
    sx = np.stack([_gauss(384, m) for m in mus_x])
    sy = np.stack([_gauss(512, m) for m in mus_y])
    locs, scores = RT.decode_simcc(sx, sy)
    assert np.abs(locs[:, 0] - mus_x).max() < 0.05 and np.abs(locs[:, 1] - mus_y).max() < 0.05
    assert np.allclose(scores, 0.9, atol=0.01)
    # the old argmax decoding is off by up to half a bin
    assert np.abs(np.argmax(sx, axis=1) - mus_x).max() > 0.25


def test_preprocess_and_back_to_image_coordinates():
    rgb = np.zeros((540, 960, 3), np.uint8)
    rgb[300:306, 400:406] = 255                        # a bright dot at (403, 303)
    bbox = np.array([300.0, 100.0, 520.0, 500.0])
    x, center, scale = RT.preprocess(rgb, bbox, (192, 256))
    assert x.shape == (1, 3, 256, 192) and x.dtype == np.float32
    img = x[0].transpose(1, 2, 0) * RT.STD + RT.MEAN
    wts = np.clip(img[..., 0], 0, None)
    yy, xx = np.indices(wts.shape)
    cx, cy = (wts * xx).sum() / wts.sum(), (wts * yy).sum() / wts.sum()   # dot centre
    back = RT.to_image(np.array([[cx * 2.0, cy * 2.0]]), center, scale, (192, 256))[0]
    assert np.abs(back - [402.5, 302.5]).max() < 1.5


def test_refine_moves_confident_agreeing_points_only():
    res = _person()
    pts = RT.mapped_points(np.zeros((26, 2)), np.zeros(26), 640, 480)
    assert set(pts) == set(RT.HALPE_TO_MP.values()) | {31, 32}
    aspect = 4 / 3
    k = RT.world_scale(res, aspect)
    assert k is not None and 2.0 < k < 4.0              # ~1/0.35 m per image height
    shifted = {27: (res.image[27, :2] + [0.01, 0.0], 0.95),   # confident, small shift
               28: (res.image[28, :2] + [0.01, 0.0], 0.10),   # unsure -> ignored
               25: (res.image[25, :2] + [0.4, 0.0], 0.95)}    # far away -> gated
    out, info = RT.refine(res, shifted, aspect, strength=1.0)
    assert info["used"] == 1 and info["gated"] == 1
    assert out.image[27, 0] == pytest.approx(res.image[27, 0] + 0.01)
    assert out.world[27, 0] == pytest.approx(res.world[27, 0] + 0.01 * aspect * k)
    assert np.allclose(out.world[[25, 28]], res.world[[25, 28]])
    assert np.allclose(out.world[:, 2], res.world[:, 2])   # depth stays MediaPipe's
    out0, _ = RT.refine(res, shifted, aspect, strength=0.0)
    assert np.allclose(out0.world, res.world)


def test_extract_onnx_takes_only_end2end(tmp_path):
    z = tmp_path / "m.zip"
    blob = b"\x08\x07" + b"\x00" * (RT.MIN_ONNX_BYTES + 10)
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("../../evil.onnx", b"x")
        f.writestr("20230605/rtmpose-m/end2end.onnx", blob)
        f.writestr("20230605/rtmpose-m/pipeline.json", "{}")
    dest = tmp_path / "out" / "model.onnx"
    dest.parent.mkdir()
    RT.extract_onnx(z, dest)
    assert dest.read_bytes() == blob and not (tmp_path / "evil.onnx").exists()
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a/end2end.onnx", b"not onnx")
    with pytest.raises(RuntimeError):
        RT.extract_onnx(z, tmp_path / "out" / "m2.onnx")
    with pytest.raises(RuntimeError):
        RT._check_url("https://evil.example/model.zip")
    RT._check_url(RT.REFINERS["rtmpose_m"].url)


def _fake_rtmpose_onnx(path, mus_x, mus_y, w=192, h=256):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper
    cx = np.stack([_gauss(w * 2, m) for m in mus_x]).astype(np.float32)[None]
    cy = np.stack([_gauss(h * 2, m) for m in mus_y]).astype(np.float32)[None]
    nodes = [
        helper.make_node("ReduceSum", ["input"], ["s"], keepdims=0),
        helper.make_node("Mul", ["s", "zero"], ["z"]),
        helper.make_node("Add", ["cx", "z"], ["simcc_x"]),
        helper.make_node("Add", ["cy", "z"], ["simcc_y"]),
    ]
    g = helper.make_graph(
        nodes, "fake_rtmpose",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, h, w])],
        [helper.make_tensor_value_info("simcc_x", TensorProto.FLOAT, [1, 26, w * 2]),
         helper.make_tensor_value_info("simcc_y", TensorProto.FLOAT, [1, 26, h * 2])],
        [numpy_helper.from_array(cx, "cx"), numpy_helper.from_array(cy, "cy"),
         numpy_helper.from_array(np.zeros((), np.float32), "zero")])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_refiner_runs_an_onnx_model_end_to_end(tmp_path):
    pytest.importorskip("onnxruntime")
    res = _person()
    w, h = 640, 480
    bbox = RT.bbox_from_landmarks(res.image, res.visibility, w, h)
    _x, center, scale = RT.preprocess(np.zeros((h, w, 3), np.uint8), bbox, (192, 256))
    # the fake network answers with MediaPipe's own points, shifted 2 px right
    target = res.image[:, :2] * [w, h] + [2.0, 0.0]
    halpe = np.zeros((26, 2))
    for k, m in RT.HALPE_TO_MP.items():
        halpe[k] = target[m]
    for m, (big, small) in RT.TOES.items():
        halpe[big] = halpe[small] = target[m]
    for k in (17, 18, 19):
        halpe[k] = target[0]
    locs = ((halpe - center + scale / 2) / scale * [192, 256]) * RT.SIMCC_SPLIT
    path = tmp_path / "fake.onnx"
    _fake_rtmpose_onnx(path, locs[:, 0], locs[:, 1])
    ref = RT.Refiner("rtmpose_m", path, use_gpu=False)
    assert ref.device == "CPU"
    out = ref(res, np.zeros((h, w, 3), np.uint8), 1.0)
    assert ref.last["used"] >= 18 and ref.last["gated"] == 0
    assert np.allclose(out.image[27, 0] * w, target[27, 0], atol=0.2)
    assert "RTMPose" in ref.describe()
    ref.close()


def test_engine_runs_with_the_refiner(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    from fbt import engine as E
    from tests.test_engine import FakeCamera, FakeEstimator
    monkeypatch.setattr(E, "CameraSource", FakeCamera)
    monkeypatch.setattr(E, "PoseEstimator", FakeEstimator)
    monkeypatch.setattr(E, "ensure_model", lambda *a, **k: tmp_path / "m.task")
    monkeypatch.setattr(RT, "MIN_ONNX_BYTES", 1000)
    models = tmp_path / "models"
    models.mkdir()
    _fake_rtmpose_onnx(models / RT.REFINERS["rtmpose_m"].file, np.full(26, 190.0),
                       np.full(26, 250.0))
    cfg = Config(output_vrchat=False, target="test", refiner="rtmpose_m", width=640,
                 height=480, fps=30, fourcc="MJPG").validate()
    cfg.mark_user_set(["output_vrchat", "width", "height", "fps", "fourcc"])
    eng = E.Engine(cfg, models, data_dir=tmp_path)
    eng.start()
    try:
        time.sleep(1.2)
        snap = eng.snapshot()
        ev = eng.poll_events()
    finally:
        eng.stop()
    assert not [p for k, p in ev if k == "error"]
    assert any("RTMPose включён" in str(p) for k, p in ev if k == "status")
    assert snap["tracking"] and snap["stats"]["refiner"].startswith("RTMPose")


def test_engine_falls_back_when_refiner_is_unavailable(tmp_path, monkeypatch):
    from fbt import engine as E
    monkeypatch.setattr(RT, "onnxruntime_available", lambda: False)
    eng = E.Engine(Config(refiner="rtmpose_x").validate(), tmp_path, data_dir=tmp_path)
    assert eng._make_refiner(Config(refiner="rtmpose_x").validate()) is None
    msgs = [p for k, p in eng.poll_events() if k == "status"]
    assert any("недоступно" in m for m in msgs)
    assert eng._make_refiner(Config().validate()) is None   # off: nothing happens


def test_new_settings_are_validated():
    c = Config(refiner="mystery", refiner_strength=7, vrchat_hmd=1, rig_auto=0,
               posture_smoothing="yes").validate()
    assert c.refiner == "off" and c.refiner_strength == 1.0
    assert c.vrchat_hmd is True and c.rig_auto is False and c.posture_smoothing is True
    d = Config().validate()
    assert d.refiner == "off" and d.vrchat_hmd is False and d.rig_auto and d.posture_smoothing


def test_calibration_keeps_vrchat_yaw():
    from fbt.skeleton import Calibration
    c = Calibration(np.eye(3), 1.0, np.zeros(3), 10.0, {}, False, 33.0)
    c2 = Calibration.from_dict(c.to_dict())
    assert c2.vrc_yaw == 33.0 and c2.hmd_yaw == 10.0
    d = c.to_dict()
    d.pop("vrc_yaw")
    assert Calibration.from_dict(d).vrc_yaw is None        # old calibrations still load
