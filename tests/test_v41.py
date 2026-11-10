"""v4.1: models from non-ASCII (Cyrillic) paths, camera names everywhere,
finding a camera again after its number changed, robust auto-setup."""
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbt import camera as C  # noqa: E402
from fbt.camera import CameraDevice, camera_label, find_device, profile_for  # noqa: E402
from fbt.config import Config, remap_camera_sources  # noqa: E402
from fbt.pose import read_model_bytes  # noqa: E402

DEVS = [CameraDevice(0, "Integrated Webcam"), CameraDevice(1, "Logitech C920"),
        CameraDevice(2, "DroidCam Source 3")]


# ------------------------------------------------------------------ models
def test_models_are_given_to_mediapipe_as_bytes():
    """MediaPipe's native code on Windows can't open C:\\Users\\Пользователь\\...:
    the model must be passed as a buffer, never as a path."""
    mp = pytest.importorskip("mediapipe")
    from fbt.pose import _base_options
    opts = _base_options(mp, b"model-bytes", False)
    assert opts.model_asset_buffer == b"model-bytes"
    assert not opts.model_asset_path
    src = (ROOT / "fbt" / "pose.py").read_text("utf-8") + (ROOT / "fbt" / "hands.py").read_text(
        "utf-8")
    assert "model_asset_path=" not in src


def test_read_model_bytes_checks_file(tmp_path):
    d = tmp_path / "Пользователь" / "models"
    d.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="не найден"):
        read_model_bytes(d / "pose_landmarker_full.task")
    bad = d / "pose_landmarker_full.task"
    bad.write_bytes(b"<html>error</html>" * 100_000)
    with pytest.raises(RuntimeError, match="повреждён"):
        read_model_bytes(bad)
    import zipfile
    good = d / "ok.task"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("a.tflite", os.urandom(1_200_000))
    assert read_model_bytes(good) == good.read_bytes()


def _real_model():
    for p in [ROOT / "models" / "pose_landmarker_lite.task",
              ROOT / "models" / "pose_landmarker_full.task",
              Path(os.environ.get("NEXOVARAPOSE_TEST_MODEL",
                                  os.environ.get("CAMFBT_TEST_MODEL", "/nonexistent")))]:
        if p.is_file():
            return p
    return None


@pytest.mark.skipif(_real_model() is None, reason="no model file available offline")
def test_real_model_loads_from_cyrillic_folder(tmp_path):
    pytest.importorskip("mediapipe")
    from fbt.pose import PoseEstimator
    d = tmp_path / "Пользователь" / "Рабочий стол" / "models"
    d.mkdir(parents=True)
    p = d / _real_model().name
    shutil.copy(_real_model(), p)
    est = PoseEstimator(p, num_poses=2)
    try:
        for i in range(3):
            assert est.detect(np.zeros((480, 640, 3), np.uint8), i * 33) == []
    finally:
        est.close()


# ------------------------------------------------------------------ camera names
def test_camera_labels():
    assert camera_label("1", "Logitech C920") == "Logitech C920 (#1)"
    assert camera_label("3") == "Камера #3"
    assert camera_label("3", "Камера #3") == "Камера #3"
    assert camera_label(str(Path("/x/Видео/test.mp4"))) == "Видео: test.mp4"
    assert len(C.short_label("0", "A" * 80)) <= 34


def test_find_device_by_name_when_numbers_change():
    assert find_device(DEVS, "1", "Logitech C920") == (DEVS[1], "ok")
    # C920 was #3 before another camera was unplugged -> found by name
    assert find_device(DEVS, "3", "Logitech C920") == (DEVS[1], "moved")
    # same number, but another device is there now -> follow the name
    assert find_device(DEVS, "0", "Logitech C920") == (DEVS[1], "moved")
    assert find_device(DEVS, "1", "Unplugged Cam") == (None, "missing")
    # config from before names were stored: number only
    assert find_device(DEVS, "2", "") == (DEVS[2], "ok")
    assert find_device(DEVS, "3", "") == (None, "missing")       # the user's log: camera #3
    assert find_device(None, "3", "x") == (None, "unknown")
    assert find_device(DEVS, "/v.mp4", "") == (None, "unknown")


def test_linux_names_from_sysfs(tmp_path, monkeypatch):
    base = tmp_path / "v4l"
    for n, name, idx in [("video0", "HD Webcam", "0"), ("video1", "HD Webcam", "1"),
                         ("video2", "Phone\x07Cam", "0")]:
        (base / n).mkdir(parents=True)
        (base / n / "name").write_text(name + "\n")
        (base / n / "index").write_text(idx + "\n")
    real = os.path.isdir
    monkeypatch.setattr(C.os.path, "isdir", lambda p: True if p == "/sys/class/video4linux"
                        else real(p))
    real_listdir, real_join = os.listdir, os.path.join
    monkeypatch.setattr(C.os, "listdir", lambda p: real_listdir(base) if p ==
                        "/sys/class/video4linux" else real_listdir(p))
    monkeypatch.setattr(C.os.path, "join", lambda a, *r: real_join(
        str(base) if a == "/sys/class/video4linux" else a, *r))
    devs = C._enumerate_linux()
    assert devs == [CameraDevice(0, "HD Webcam"), CameraDevice(2, "PhoneCam")]


def test_enumeration_is_cached_and_never_opens_cameras(monkeypatch):
    calls = []
    monkeypatch.setattr(C, "_enumerate_os", lambda backend=None: calls.append(1) or DEVS)
    monkeypatch.setattr(C.cv2, "VideoCapture", lambda *a: pytest.fail("camera opened"))
    monkeypatch.setattr(C, "_names_cache", (-1e9, None))
    assert C.enumerate_devices() == DEVS
    assert C.device_name("1") == "Logitech C920" and C.device_name("7") == ""
    assert len(calls) == 1
    assert C.list_cameras() == [0, 1, 2]
    assert C.list_camera_devices(allow_open=False) == DEVS


def test_without_names_cameras_are_probed_only_when_allowed(monkeypatch):
    monkeypatch.setattr(C, "_enumerate_os", lambda backend=None: None)
    monkeypatch.setattr(C, "_names_cache", (-1e9, None))
    monkeypatch.setattr(C, "_probe_indices", lambda n: [0, 2])
    assert C.list_camera_devices(allow_open=False) is None
    assert C.list_camera_devices() == [CameraDevice(0, "Камера #0"), CameraDevice(2, "Камера #2")]
    # the OS list came back empty -> double-check by opening (camera may still work)
    monkeypatch.setattr(C, "_enumerate_os", lambda backend=None: [])
    monkeypatch.setattr(C, "_names_cache", (-1e9, None))
    assert C.list_camera_devices(allow_open=False) == []
    assert C.list_camera_devices() == [CameraDevice(0, "Камера #0"), CameraDevice(2, "Камера #2")]


def test_open_error_names_the_camera(monkeypatch):
    class Dead:
        def isOpened(self):
            return False

        def release(self):
            pass
    monkeypatch.setattr(C, "_enumerate_os", lambda backend=None: DEVS)
    monkeypatch.setattr(C, "_names_cache", (-1e9, None))
    monkeypatch.setattr(C.cv2, "VideoCapture", lambda *a: Dead())
    with pytest.raises(RuntimeError) as e:
        C.open_capture(1)
    assert "Logitech C920 (#1)" in str(e.value) and "занята" in str(e.value)


def test_profiles_belong_to_the_named_camera():
    profiles = {"0": {"name": "Integrated Webcam", "modes": []}, "1": {"modes": []}}
    assert profile_for(profiles, "0", "Integrated Webcam") is profiles["0"]
    assert profile_for(profiles, "0", "Logitech C920") is None   # index now another device
    assert profile_for(profiles, "1", "Logitech C920") is profiles["1"]  # old, unnamed
    assert profile_for(profiles, "5", "") is None


def test_remap_swapped_cameras():
    cfg = Config().validate()
    cfg.camera_profiles = {"0": {"name": "A"}, "1": {"name": "B"}, "4": {"name": "X"}}
    cfg.camera_rig = {"1": {"rms_cm": 2}}
    remap_camera_sources(cfg, {"0": "1", "1": "0"})
    assert cfg.camera_profiles == {"1": {"name": "A"}, "0": {"name": "B"}, "4": {"name": "X"}}
    assert cfg.camera_rig == {"0": {"rms_cm": 2}}


def test_config_keeps_names():
    cfg = Config(camera="1", camera_name=" Logitech\x00 C920 ",
                 extra_cameras=[{"source": "2", "name": "DroidCam Source 3"},
                                {"source": "/a.mp4", "name": "junk"}]).validate()
    assert cfg.camera_name == "Logitech C920"
    assert cfg.extra_cameras[0]["name"] == "DroidCam Source 3"
    assert cfg.extra_cameras[1]["name"] == ""
    assert Config(camera="/a.mp4", camera_name="x").validate().camera_name == ""


# ------------------------------------------------------------------ engine
def _engine(tmp_path, cfg, monkeypatch, devs):
    from fbt import engine as E
    monkeypatch.setattr(E, "enumerate_devices", lambda *a: devs)
    eng = E.Engine(cfg, tmp_path, data_dir=tmp_path)
    eng._resolve_cameras(cfg)
    return dict(eng.poll_events())


def test_engine_replaces_missing_camera_from_log(tmp_path, monkeypatch):
    """The user's log: camera #3 could not be opened -> tracking must start with a
    connected camera instead of failing."""
    cfg = Config(camera="3").validate()
    ev = _engine(tmp_path, cfg, monkeypatch, DEVS)
    assert cfg.camera == "0" and cfg.camera_name == "Integrated Webcam"
    assert ev["cameras_resolved"]["persist"] is True
    assert "не найдена" in ev["status"] and "Integrated Webcam (#0)" in ev["status"]


def test_engine_keeps_remembered_camera_choice_when_unplugged(tmp_path, monkeypatch):
    cfg = Config(camera="4", camera_name="Razer Kiyo",
                 extra_cameras=[{"source": "0", "name": "Integrated Webcam"}]).validate()
    ev = _engine(tmp_path, cfg, monkeypatch, DEVS)
    assert cfg.camera == "1"                      # first camera not used as an extra one
    assert ev["cameras_resolved"]["persist"] is False  # settings keep "Razer Kiyo"
    assert "Razer Kiyo (#4)" in ev["status"]


def test_engine_follows_moved_cameras(tmp_path, monkeypatch):
    cfg = Config(camera="2", camera_name="Logitech C920",
                 extra_cameras=[{"source": "1", "name": "DroidCam Source 3"},
                                {"source": "5", "name": "Gone"}]).validate()
    cfg.camera_profiles = {"2": {"name": "Logitech C920", "modes": []}}
    cfg.camera_rig = {"1": {"rms_cm": 3.0}}
    ev = _engine(tmp_path, cfg, monkeypatch, DEVS)
    upd = ev["cameras_resolved"]
    assert (cfg.camera, upd["persist"]) == ("1", True)
    assert upd["mapping"] == {"2": "1", "1": "2"}
    assert cfg.camera_profiles == {"1": {"name": "Logitech C920", "modes": []}}
    assert cfg.camera_rig == {"2": {"rms_cm": 3.0}}
    assert cfg.extra_cameras[0]["source"] == "2" and cfg.extra_cameras[0]["enabled"]
    assert cfg.extra_cameras[1]["enabled"] is False            # unplugged: skipped this run
    assert upd["extras"][1] is None
    assert "Gone (#5)" in ev["status"]


def test_engine_learns_names(tmp_path, monkeypatch):
    cfg = Config(camera="1").validate()
    ev = _engine(tmp_path, cfg, monkeypatch, DEVS)
    assert cfg.camera_name == "Logitech C920" and ev["cameras_resolved"]["persist"]
    assert "status" not in ev
    # empty OS list: don't guess, opening the camera reports a clear error itself
    cfg = Config(camera="0").validate()
    assert _engine(tmp_path, cfg, monkeypatch, []) == {} and cfg.camera == "0"


def test_engine_without_names_changes_nothing(tmp_path, monkeypatch):
    cfg = Config(camera="3").validate()
    assert _engine(tmp_path, cfg, monkeypatch, None) == {}
    assert cfg.camera == "3"


# ------------------------------------------------------------------ UI
@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from fbt.ui.theme import apply_theme
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


def test_autosetup_lists_cameras_by_name(app, tmp_path, monkeypatch):
    from fbt.pose import ModelManifest
    from fbt.ui import autosetup as A
    monkeypatch.setattr(A, "list_camera_devices", lambda **k: DEVS)
    dlg = A.AutoSetupDialog(Config(camera="3").validate(), tmp_path, [], ModelManifest(None))
    dlg._on_cameras(DEVS)
    items = [dlg.cam_combo.itemText(i) for i in range(dlg.cam_combo.count())]
    assert items == ["Integrated Webcam (#0)", "Logitech C920 (#1)", "DroidCam Source 3 (#2)"]
    assert dlg.cam_combo.currentIndex() == 0 and "не найдена" in dlg.cam_note.text()
    assert dlg.go_btn.isEnabled()
    dlg2 = A.AutoSetupDialog(Config(camera="0", camera_name="Logitech C920").validate(),
                             tmp_path, [], ModelManifest(None))
    dlg2._on_cameras(DEVS)
    assert dlg2.cam_combo.currentText() == "Logitech C920 (#1)"
    dlg2._on_cameras([])
    assert not dlg2.go_btn.isEnabled() and "не найдена" in dlg2.cam_note.text()
    dlg.close()
    dlg2.close()


def test_autosetup_camera_failure_is_recoverable(app, tmp_path, monkeypatch):
    """Auto-setup used to crash with "Не удалось открыть камеру #3"."""
    from fbt.pose import ModelManifest
    from fbt.ui import autosetup as A
    monkeypatch.setattr(A, "list_camera_devices", lambda **k: DEVS)
    monkeypatch.setattr(A, "ensure_model", lambda v, *a, **k: tmp_path / f"{v}.task")

    def busy(index, *a, **k):
        raise RuntimeError(C.open_capture.__doc__ or f"Не удалось открыть камеру «{index}»")
    monkeypatch.setattr(A, "probe_camera", busy)
    dlg = A.AutoSetupDialog(Config().validate(), tmp_path, [], ModelManifest(None))
    dlg._on_cameras(DEVS)
    dlg.cam_combo.setCurrentIndex(1)
    msgs = []
    dlg._bridge.failed.disconnect()
    dlg._bridge.failed.connect(msgs.append)
    dlg._bridge.progress.disconnect()
    texts = []
    dlg._bridge.progress.connect(lambda p, t: texts.append(t))
    dlg._start()
    dlg._thread.join(5)
    app.processEvents()
    assert msgs and msgs[0].startswith("camera:")
    dlg._on_failed(msgs[0])
    assert not dlg.back_btn.isHidden() and "Logitech C920" in dlg.step_labels[1].text()
    dlg._back()
    assert dlg.stack.currentIndex() == 0 and dlg.back_btn.isHidden()
    dlg.close()


def test_autosetup_result_shows_and_returns_camera(app, tmp_path):
    from fbt import autotune
    from fbt.pose import ModelManifest
    from fbt.ui.autosetup import AutoSetupDialog
    cfg = Config().validate()
    dlg = AutoSetupDialog(cfg, tmp_path, [], ModelManifest(None))
    prof = {"modes": [{"width": 1280, "height": 720, "fps": 30.0, "fps_requested": 60,
                       "fourcc": "MJPG"}]}
    bench = {"results": [{"variant": "full", "delegate": "CPU", "ms": 9.0, "detect_rate": 1.0}],
             "hand_ms": None}
    recs = autotune.recommend(cfg, {"threads": 8}, prof, bench, "vrchat_osc", False)
    dlg._on_done({"sysinfo": {"cpu": "x", "threads": 8}, "profile": prof, "bench": bench,
                  "recs": recs, "camera": "1", "camera_name": "Logitech C920"})
    assert "Камера: Logitech C920 (#1)" in dlg.sys_label.text()
    dlg._apply()
    assert dlg.result_extra["camera"] == "1"
    assert dlg.result_extra["camera_name"] == "Logitech C920"


def test_settings_camera_lists_show_names(app):
    from fbt.ui.settings_dialog import SettingsDialog
    cfg = Config(camera="1", camera_name="Logitech C920",
                 extra_cameras=[{"source": "4", "name": "Old Phone"}]).validate()
    dlg = SettingsDialog(cfg)
    got = []
    dlg.changed.connect(got.append)
    dlg.set_cameras(DEVS)
    assert dlg.cam_combo.currentText() == "Logitech C920 (#1)"
    dlg.cam_combo.setCurrentIndex(2)
    dlg._camera_edited()
    assert (cfg.camera, cfg.camera_name) == ("2", "DroidCam Source 3")
    assert "camera" in cfg.user_set and got[-1] == ["camera", "camera_name"]
    assert "DroidCam Source 3" in dlg.cam_note.text()
    # extra camera row: not connected now -> shown by name with a note
    from PySide6.QtWidgets import QComboBox
    combos = dlg._extra_box.itemAt(0).widget().findChildren(QComboBox)
    assert combos[0].currentText() == "Old Phone (#4) — не подключена"
    assert "Old Phone (#4)" in dlg.rig_state.text()
    combos[0].setCurrentIndex(0)
    dlg._extra_set_source(0, combos[0])
    assert cfg.extra_cameras[0]["source"] == "0"
    assert cfg.extra_cameras[0]["name"] == "Integrated Webcam"
    # typing a number still works
    dlg.cam_combo.setEditText("1")
    dlg._camera_edited()
    assert (cfg.camera, cfg.camera_name) == ("1", "Logitech C920")
    dlg.close()


def test_main_window_shows_camera_name_and_saves_resolution(app, tmp_path, monkeypatch):
    from fbt.ui import main_window as M
    monkeypatch.setattr(M, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(M, "save_config", lambda cfg: None)
    monkeypatch.setattr(M, "list_camera_devices", lambda **k: DEVS)
    cfg = Config(camera="1", camera_name="Logitech C920", first_run_done=True).validate()
    w = M.MainWindow(cfg, tmp_path, [tmp_path])
    assert w.c_cam.sub.text() == "Logitech C920 (#1)"
    w._cameras_resolved({"camera": "0", "camera_name": "Integrated Webcam", "persist": True,
                         "mapping": {}, "extras": [], "notes": []})
    assert (cfg.camera, cfg.camera_name) == ("0", "Integrated Webcam")
    w._cameras_resolved({"camera": "2", "camera_name": "DroidCam Source 3", "persist": False,
                         "mapping": {}, "extras": [], "notes": []})
    assert cfg.camera == "0" and w._run_cam == ("2", "DroidCam Source 3")
    w._cameras_found(DEVS)
    assert "Logitech C920 (#1)" in w._status
    w.close()
