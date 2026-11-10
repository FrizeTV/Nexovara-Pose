"""v4.2: SteamVR without a headset — driver installed to an ASCII path and
registered without vrpathreg, SteamVR settings, diagnostics from SteamVR's log,
controller buttons from keyboard/mouse, target detection, VRChat desktop."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbt import controls as K  # noqa: E402
from fbt import steamvr  # noqa: E402
from fbt.config import Config  # noqa: E402
from fbt.outputs import DEV_HMD, DEV_LEFT_HAND, DEV_RIGHT_HAND, DriverOutput, encode_driver_packet  # noqa: E402
from fbt.skeleton import Tracker  # noqa: E402

GARBLED = "C:\\Users\\\udccf\udcee\udceb\\Desktop\\FBT\\Nexovara Pose\\driver\\dist\\nexovarapose"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A SteamVR install: openvrpaths.vrpath (with an entry garbled by vrpathreg),
    steamvr.vrsettings (driver blocked by safe mode), vrserver.txt, a built driver."""
    rt = tmp_path / "SteamVR"
    rt.mkdir()
    cfgdir, logdir = tmp_path / "config", tmp_path / "logs"
    cfgdir.mkdir()
    logdir.mkdir()
    settings = cfgdir / "steamvr.vrsettings"
    settings.write_text(json.dumps({"steamvr": {"requireHmd": True},
                                    "driver_nexovarapose": {"enable": False,
                                                      "blocked_by_safe_mode": True}}))
    vrpath = tmp_path / "openvrpaths.vrpath"
    other = str(tmp_path / "drivers" / "vmt")
    doc = json.dumps({"config": [str(cfgdir)], "external_drivers": [other, "@@G@@"],
                      "jsonid": "vrpathreg", "log": [str(logdir)], "runtime": [str(rt)],
                      "version": 1}, indent="\t")
    raw = doc.encode().replace(b"@@G@@", GARBLED.replace("\\", "\\\\").encode(
        "utf-8", "surrogateescape"))
    vrpath.write_bytes(raw)                                   # not valid UTF-8 (cp1251 bytes)
    dist = tmp_path / "proj" / "driver" / "dist" / "nexovarapose"
    binary = dist / "bin" / ("win64" if os.name == "nt" else "linux64") / (
        "driver_nexovarapose.dll" if os.name == "nt" else "driver_nexovarapose.so")
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"driver v1")
    (dist / "driver.vrdrivermanifest").write_text("{}")
    inst = tmp_path / "ProgramData" / "Nexovara Pose" / "SteamVR" / "nexovarapose"
    monkeypatch.setattr(steamvr, "_vrpath_file", lambda: vrpath)
    monkeypatch.setattr(steamvr, "driver_dist_dir", lambda: dist)
    monkeypatch.setattr(steamvr, "driver_install_dir", lambda: inst)
    monkeypatch.setattr(steamvr, "is_running", lambda: False)
    return dict(vrpath=vrpath, settings=settings, dist=dist, inst=inst, binary=binary,
                logdir=logdir, other=other)


def _vrpaths(f):
    return json.loads(f["vrpath"].read_bytes().decode("utf-8", "surrogateescape"))


def test_install_copies_to_ascii_folder_and_repairs_registration(fake):
    assert steamvr.registered_driver_paths() == [GARBLED]
    assert not steamvr.driver_registered()
    assert steamvr.driver_blocked(fake["settings"])
    inst = steamvr.register_driver()
    # The fixture intentionally replaces the install directory with a temp
    # path that may contain the localized Windows user name.  ASCII safety is
    # tested separately by test_install_dir_is_ascii_on_windows; here verify
    # that registration uses the configured directory and completes correctly.
    assert inst == fake["inst"]
    assert (inst / "driver.vrdrivermanifest").is_file()
    d = _vrpaths(fake)
    # garbled entry gone, other drivers kept, ours added once
    assert d["external_drivers"] == [fake["other"], str(inst)]
    assert d["jsonid"] == "vrpathreg" and d["version"] == 1
    assert steamvr.driver_registered() and steamvr.installed_copy_current()
    s = json.loads(fake["settings"].read_text())
    assert s["driver_nexovarapose"]["enable"] is True and "blocked_by_safe_mode" not in s["driver_nexovarapose"]
    assert list(fake["vrpath"].parent.glob("openvrpaths.vrpath.nexovarapose-backup-*"))
    # rebuilt driver -> the installed copy is outdated until installed again
    fake["binary"].write_bytes(b"driver v2")
    assert not steamvr.installed_copy_current()
    steamvr.register_driver()
    assert steamvr.installed_copy_current()
    assert _vrpaths(fake)["external_drivers"].count(str(inst)) == 1
    steamvr.unregister_driver()
    assert _vrpaths(fake)["external_drivers"] == [fake["other"]]
    assert not fake["inst"].exists()


def test_install_refuses_while_steamvr_runs_or_unbuilt(fake, monkeypatch):
    monkeypatch.setattr(steamvr, "is_running", lambda: True)
    with pytest.raises(RuntimeError, match="Закройте SteamVR"):
        steamvr.register_driver()
    monkeypatch.setattr(steamvr, "is_running", lambda: False)
    fake["binary"].unlink()
    with pytest.raises(RuntimeError, match="build_driver"):
        steamvr.register_driver()


def test_install_dir_is_ascii_on_windows(monkeypatch, tmp_path):
    monkeypatch.undo()
    monkeypatch.setattr(steamvr.sys, "platform", "win32")
    monkeypatch.setenv("ProgramData", "C:\\ProgramData")
    assert str(steamvr.driver_install_dir()).replace("/", "\\").endswith(
        "ProgramData\\Nexovara Pose\\SteamVR\\nexovarapose")
    assert str(steamvr.driver_install_dir()).isascii()


def test_no_hmd_mode_window_and_restore(fake):
    steamvr.set_no_hmd_mode(True, fake["settings"], window=(1920, 1080), vertical_fov=75)
    s = json.loads(fake["settings"].read_text())
    drv, sv = s["driver_nexovarapose"], s["steamvr"]
    assert drv["enableHmd"] and drv["window_width"] == 1920 and drv["vertical_fov"] == 75
    assert sv == {"requireHmd": False, "forcedDriver": "nexovarapose", "activateMultipleDrivers": True}
    steamvr.set_no_hmd_mode(True, fake["settings"])          # twice: original values kept
    steamvr.set_no_hmd_mode(False, fake["settings"])
    sv = json.loads(fake["settings"].read_text())["steamvr"]
    assert sv["requireHmd"] is True and "forcedDriver" not in sv


def test_diagnose_walks_the_chain(fake):
    names = lambda checks: {c.title: c.ok for c in checks}  # noqa: E731
    d = names(steamvr.diagnose(True))
    assert d["Драйвер собран"] and d["Драйвер установлен в SteamVR"] is False
    assert d["Драйвер включён в SteamVR"] is False and d["Режим без шлема в SteamVR"] is False
    done = steamvr.fix_all(True, (1600, 900), 90)
    assert len(done) == 2
    (fake["logdir"] / "vrserver.txt").write_text(
        "Fri Oct 26 2026 20:00:01.001 - Loading driver nexovarapose\n"
        "Fri Oct 26 2026 20:00:01.200 - driver_nexovarapose: nexovarapose: driver initialised (virtual HMD on, "
        "protocol 1)\n"
        "Fri Oct 26 2026 20:00:09.000 - driver_nexovarapose: nexovarapose: receiving data from the Nexovara Pose app\n",
        encoding="utf-8")
    d = names(steamvr.diagnose(True))
    assert all(d[k] for k in ("Драйвер установлен в SteamVR", "Драйвер включён в SteamVR",
                              "Режим без шлема в SteamVR", "SteamVR загрузил драйвер",
                              "Данные от Nexovara Pose доходят до драйвера"))
    assert len(steamvr.driver_log_lines()) == 3


def test_diagnose_reports_load_failure(fake):
    steamvr.register_driver()
    (fake["logdir"] / "vrserver.txt").write_text(
        "Unable to load driver nexovarapose because of error VRInitError_Init_FileNotFound(103)\n")
    d = {c.title: (c.ok, c.text) for c in steamvr.diagnose(False)}
    assert d["SteamVR загрузил драйвер"][0] is False and "103" in d["SteamVR загрузил драйвер"][1]


def test_detect_target_without_driver_is_osc(monkeypatch):
    from fbt import presets
    monkeypatch.setattr(presets, "running_apps", lambda: {"steamvr": True})
    monkeypatch.setattr(steamvr, "driver_registered", lambda: False)
    assert presets.detect_target()[0] == "vrchat_osc"
    monkeypatch.setattr(presets, "running_apps", lambda: {"steamvr": False})
    monkeypatch.setattr(steamvr, "driver_registered", lambda: True)
    monkeypatch.setattr(steamvr, "no_hmd_mode_enabled", lambda *a: True)
    assert presets.detect_target()[0] == "steamvr_nohmd"


def test_vrchat_desktop_detection(monkeypatch):
    psutil = pytest.importorskip("psutil")

    class P:
        def __init__(self, cmd):
            self.info, self._cmd = {"name": "VRChat.exe"}, cmd

        def cmdline(self):
            return self._cmd
    monkeypatch.setattr(steamvr, "is_running", lambda: True)
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [P(["VRChat.exe", "--no-vr"])])
    assert steamvr.vrchat_desktop_running()
    monkeypatch.setattr(psutil, "process_iter", lambda attrs=None: [P(["VRChat.exe"])])
    assert not steamvr.vrchat_desktop_running()
    monkeypatch.setattr(steamvr, "is_running", lambda: False)   # no SteamVR -> desktop
    assert steamvr.vrchat_desktop_running()


# ------------------------------------------------------------------ keyboard / mouse
class FakeKeys:
    def __init__(self):
        self.pressed, self.cur = set(), (0.0, 0.0)

    def down(self, vk):
        return vk in self.pressed

    def cursor(self):
        return self.cur


def test_keyboard_controls_map_to_buttons():
    fk = FakeKeys()
    kc = K.KeyboardControls("F8", "F9", "F10", backend=fk)
    assert kc.poll().active is False
    fk.pressed = {0x77}                                    # F8
    st = kc.poll()
    assert st.left == K.BTN_B and st.right == 0 and "меню" in st.describe()
    fk.pressed = {0x78}                                    # F9 -> both triggers
    st = kc.poll()
    assert st.left & K.BTN_TRIGGER and st.right & K.BTN_TRIGGER
    fk.pressed = {0x79}                                    # F10 toggles pointer (edge)
    assert kc.poll().pointer is not None
    assert kc.poll().pointer is not None                   # held: still on, no flicker
    fk.pressed = set()
    fk.cur = (1.0, -1.0)                                   # top-right corner
    st = kc.poll()
    assert st.pointer == (K.POINTER_YAW_DEG, -K.POINTER_PITCH_DEG)
    fk.pressed = {K.VK_LBUTTON}
    assert kc.poll().right & K.BTN_TRIGGER
    fk.pressed = {0x79}
    kc.poll()
    fk.pressed = set()
    assert kc.poll().pointer is None                       # toggled off
    off = K.KeyboardControls("off", "off", "off", backend=fk)
    fk.pressed = {0x77, 0x78, 0x79}
    assert not off.poll().active


def test_pointer_pose_points_where_the_mouse_is():
    head = np.array([0.0, 1.6, 0.0])
    R = np.eye(3)
    pos, rot = K.pointer_pose(head, R, 0.0, 0.0)
    assert np.allclose(rot[:, 2], [0, 0, 1]) and pos[0] > 0 and pos[1] < 1.6
    _p, rot = K.pointer_pose(head, R, 30.0, 0.0)
    assert rot[:, 2][0] > 0.4                              # cursor right -> points right
    _p, rot = K.pointer_pose(head, R, 0.0, 20.0)
    assert rot[:, 2][1] < -0.3                             # cursor down -> points down


def test_driver_packet_carries_buttons_and_pointer():
    import struct
    out = DriverOutput(39999, send_hmd=True)
    sent = []
    out.client.send_raw = sent.append
    st = K.ControlState(left=K.BTN_B, right=K.BTN_TRIGGER, pointer=(0.0, 0.0))
    head = Tracker("head", np.array([0.0, 1.6, 0.0]), np.eye(3))
    hip = Tracker("hip", np.array([0.0, 1.0, 0.0]), np.eye(3))
    hands = {"left": (Tracker("left_hand", np.array([-0.3, 1.0, 0.1]), np.eye(3)), None)}
    out.send([hip], head, hands, st)
    pkt = sent[-1]
    n = pkt[12]
    recs = {pkt[16 + 52 * i]: pkt[16 + 52 * i:16 + 52 * (i + 1)] for i in range(n)}
    assert set(recs) == {0, DEV_HMD, DEV_LEFT_HAND, DEV_RIGHT_HAND}  # right hand from the mouse
    flags, buttons = recs[DEV_LEFT_HAND][1], struct.unpack("<H", recs[DEV_LEFT_HAND][2:4])[0]
    assert flags & 4 and buttons == K.BTN_B
    assert struct.unpack("<H", recs[DEV_RIGHT_HAND][2:4])[0] == K.BTN_TRIGGER
    assert recs[0][1] & 4 == 0                              # trackers carry no buttons
    # old-style call (no controls) is byte-compatible with protocol v1 apps
    assert encode_driver_packet(1, [(0, hip, None)])[16 + 1] == 1
    out.shutdown()


def test_cpp_parser_reads_buttons(tmp_path):
    exe = ROOT / "driver" / "build" / ("Release" if os.name == "nt" else "") / (
        "nexovarapose_protocol_test" + (".exe" if os.name == "nt" else ""))
    if not exe.is_file():
        pytest.skip("driver not built")
    t = Tracker("right_hand", np.array([0.1, 1.2, 0.3]), np.eye(3))
    f = tmp_path / "p.bin"
    f.write_bytes(encode_driver_packet(5, [(DEV_RIGHT_HAND, t, None, K.BTN_B | K.BTN_TRIGGER)]))
    out = subprocess.run([str(exe), "--parse", str(f)], capture_output=True, text=True).stdout
    last = out.strip().splitlines()[-1].split()
    assert int(last[0]) == DEV_RIGHT_HAND and int(last[-1]) == K.BTN_B | K.BTN_TRIGGER


def test_config_validates_keys_and_window():
    c = Config(key_menu="F9", key_triggers="F9", key_pointer="Escape", nohmd_window="99x9",
               nohmd_fov=500).validate()
    assert c.key_menu == "F9" and c.key_triggers != "F9" and c.key_pointer == "F10"
    assert c.nohmd_window == "1600x900" and c.nohmd_fov == 120


def test_sender_sends_buttons_even_without_body(monkeypatch):
    """Tracking lost (nobody in view) but F8 pressed: the driver still gets the button."""
    import time
    from fbt import sender as SM
    fk = FakeKeys()
    monkeypatch.setattr(SM, "KeyboardControls",
                        lambda *a: K.KeyboardControls(*a, backend=fk))
    got = []
    monkeypatch.setattr(DriverOutput, "send", lambda self, tr, head, hands=None, controls=None,
                        **_k: got.append((len(tr), controls.left if controls else None)))
    cfg = Config(output_vrchat=False, output_driver=True, driver_hmd=True, hold_time=0.1,
                 lost_hold_s=0.0, send_rate=90).validate()     # (holding is tested in v4.4.3)
    snd = SM.OutputSender()
    snd.configure(cfg)
    snd.start()
    try:
        head = Tracker("head", np.array([0.0, 1.6, 0.0]), np.eye(3))
        hands = {"left": (Tracker("left_hand", np.array([-0.3, 1.0, 0.1]), np.eye(3)), None)}
        snd.push(SM.Keyframe(time.monotonic() - 1.0, [Tracker("hip", np.zeros(3), np.eye(3))],
                             head, hands))
        time.sleep(0.15)
        assert not got                                      # stale body, no keys -> silent
        fk.pressed = {0x77}
        time.sleep(0.15)
        assert got and all(n == 0 for n, _ in got) and got[-1][1] == K.BTN_B
    finally:
        snd.stop()


def test_engine_anchors_driver_only_with_a_real_headset():
    from fbt.engine import _wants_hmd
    assert _wants_hmd(Config(output_driver=True, driver_hmd=False, hmd_anchor=True).validate())
    assert not _wants_hmd(Config(output_driver=True, driver_hmd=True, hmd_anchor=True).validate())
    assert not _wants_hmd(Config(output_vrchat=True, output_driver=False,
                                 hmd_anchor=True).validate())


# ------------------------------------------------------------------ tracking / fingers
class _FakeEst:
    """Stands in for PoseEstimator: returns `scene` people (shared list)."""
    instances = []

    def __init__(self, model_path, *a, num_poses=1, **k):
        self.num_poses, self.calls, self.delegate = num_poses, 0, "CPU"
        _FakeEst.instances.append(self)

    def detect(self, rgb, ts):
        self.calls += 1
        n = _FakeEst.scene[0]
        return ["p"] * (min(n, 1) if self.num_poses == 1 and not _FakeEst.single_lost[0] else
                        0 if self.num_poses == 1 else n)

    def close(self):
        pass


def test_adaptive_estimator_runs_multi_person_search_only_when_needed():
    from fbt.pose import AdaptivePoseEstimator, make_estimator
    _FakeEst.instances, _FakeEst.scene, _FakeEst.single_lost = [], [1], [False]
    est = make_estimator("m.task", num_poses=3, cls=_FakeEst)
    assert isinstance(est, AdaptivePoseEstimator)
    single, multi = _FakeEst.instances
    ts = 0
    for _ in range(60):                         # alone for 2 s at 30 fps
        assert len(est.detect(None, ts)) == 1
        ts += 33
    assert single.calls > 50 and multi.calls <= 5   # one look around every 0.5 s
    _FakeEst.scene[0] = 2                       # someone walks in
    for _ in range(20):
        est.detect(None, ts)
        ts += 33
    assert est.crowded and len(est.detect(None, ts)) == 2   # every frame: all people
    _FakeEst.scene[0] = 1                       # they leave; after HOLD_S back to cheap mode
    for _ in range(90):
        est.detect(None, ts)
        ts += 33
    before = multi.calls
    for _ in range(10):
        est.detect(None, ts)
        ts += 33
    assert multi.calls - before <= 1 and not est.crowded
    _FakeEst.single_lost[0] = True              # the 1-person tracker lost you:
    assert len(est.detect(None, ts)) == 1       # same frame is filled by the full search
    assert type(make_estimator("m.task", num_poses=1, cls=_FakeEst)) is _FakeEst


def test_finger_range_learns_open_hand():
    from fbt.hands import FingerRange
    fr = FingerRange()
    relaxed = np.array([0.15, 0.12, 0.18, 0.2, 0.22])
    for i in range(30):
        out = fr(relaxed, i * 0.05)
    assert np.allclose(out, 0.0)                # relaxed open hand -> straight fingers
    fist = fr(np.ones(5), 2.0)
    assert np.allclose(fist, 1.0)               # a fist is still a fist
    half = fr(np.full(5, 0.6), 2.1)
    assert np.all(half < 0.6) and np.all(half > 0.4)
    fr2 = FingerRange()
    for i in range(30):
        out = fr2(np.full(5, 0.7), i * 0.05)    # never opened: base capped at MAX_BASE
    assert np.all(out > 0.5)


def test_hand_in_crop_must_belong_to_that_wrist():
    from fbt.hands import _is_this_hand
    from fbt.skeleton import L_WR, R_WR
    img = np.zeros((33, 3))
    img[L_WR, :2] = [0.40, 0.5]
    img[R_WR, :2] = [0.46, 0.5]                 # hands close together
    vis = np.ones(33)
    assert _is_this_hand(np.array([0.405, 0.5]), img, vis, "left", 1280, 720, 200)
    assert not _is_this_hand(np.array([0.458, 0.5]), img, vis, "left", 1280, 720, 200)
    assert not _is_this_hand(np.array([0.9, 0.5]), img, vis, "left", 1280, 720, 200)
