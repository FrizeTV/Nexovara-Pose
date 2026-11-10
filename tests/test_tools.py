import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import osc_monitor  # noqa: E402
import setup as camsetup  # noqa: E402
from fbt.osc import encode_message  # noqa: E402


def test_monitor_parses_our_messages():
    raw = encode_message("/tracking/trackers/1/position", [0.1, 0.95, -0.2])
    [(addr, args)] = list(osc_monitor.parse(raw))
    assert addr == "/tracking/trackers/1/position"
    assert args == pytest.approx([0.1, 0.95, -0.2])
    raw = encode_message("/VMT/Room/Unity", [0, 7, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    [(_, args)] = list(osc_monitor.parse(raw))
    assert args[:2] == [0, 7] and len(args) == 10


def test_monitor_parses_bundles_and_ignores_garbage():
    m = encode_message("/a", [1])
    bundle = b"#bundle\x00" + b"\x00" * 8 + len(m).to_bytes(4, "big") + m
    assert list(osc_monitor.parse(bundle)) == [("/a", [1])]
    # truncated / malformed input must never hang or crash beyond a caught error
    for junk in (b"", b"abc", b"/x\x00\x00,z\x00\x00", m[:-2], b"#bundle\x00" + b"\xff" * 12):
        try:
            list(osc_monitor.parse(junk))
        except (ValueError, IndexError, struct.error):  # exactly what the monitor catches
            pass


def test_find_steamvr_reads_openvr_paths(tmp_path, monkeypatch):
    rt = tmp_path / "SteamVR"
    rt.mkdir()
    vr = tmp_path / "openvr"
    vr.mkdir()
    (vr / "openvrpaths.vrpath").write_text(json.dumps({"runtime": [str(rt)]}))
    monkeypatch.setattr(camsetup, "IS_WIN", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert camsetup.find_steamvr() == rt
    (vr / "openvrpaths.vrpath").write_text("{broken")
    assert camsetup.find_steamvr() is None


def test_portable_mode(tmp_path, monkeypatch):
    from fbt import config
    monkeypatch.setattr(config, "project_root", lambda: tmp_path)
    (tmp_path / "portable.txt").write_text("1")
    assert config.app_dir() == tmp_path / "data"
