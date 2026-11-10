"""Qt UI tests (offscreen). Skipped if PySide6 is not installed."""
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from fbt import autotune  # noqa: E402
from fbt.config import Config  # noqa: E402
from fbt.pose import ModelManifest  # noqa: E402


@pytest.fixture(scope="module")
def app():
    from fbt.ui.theme import apply_theme
    a = QApplication.instance() or QApplication([])
    apply_theme(a)
    return a


def test_settings_marks_user_changes(app):
    from fbt.ui.settings_dialog import SettingsDialog
    cfg = Config().validate()
    dlg = SettingsDialog(cfg)
    got = []
    dlg.changed.connect(got.append)
    spin = dlg._fields["user_height"][0]
    spin.setValue(1.82)
    assert cfg.user_height == pytest.approx(1.82)
    assert "user_height" in cfg.user_set and got == [["user_height"]]
    assert "●" in dlg._labels["user_height"].text()
    dlg._tracker_boxes["chest"].setChecked(True)
    assert cfg.trackers["chest"] and "trackers" in cfg.user_set
    # refresh() must not mark anything
    before = list(cfg.user_set)
    dlg.refresh()
    assert cfg.user_set == before
    dlg.close()


def test_autosetup_keeps_user_values_unless_ticked(app, tmp_path):
    from fbt.ui.autosetup import AutoSetupDialog
    cfg = Config(model="lite").validate()
    cfg.mark_user_set(["model"])
    dlg = AutoSetupDialog(cfg, tmp_path, [], ModelManifest(None))
    prof = {"modes": [{"width": 1280, "height": 720, "fps": 60.0, "fps_requested": 60,
                       "fourcc": "MJPG"}]}
    bench = {"results": [{"variant": "full", "delegate": "CPU", "ms": 8.0, "detect_rate": 1.0}],
             "hand_ms": None}
    recs = autotune.recommend(cfg, {"threads": 8}, prof, bench, "vrchat_osc", False)
    dlg._on_done({"sysinfo": {"cpu": "x", "threads": 8, "ram_gb": 8}, "profile": prof,
                  "bench": bench, "recs": recs})
    rows = {r.key: i for i, r in enumerate(dlg._rows)}
    assert dlg.table.item(rows["model"], 0).checkState() == Qt.Unchecked   # user's choice
    assert dlg.table.item(rows["width"], 0).checkState() == Qt.Checked
    dlg._apply()
    assert "model" not in dlg.result_values and dlg.result_values["width"] == 1280
    assert dlg.result_extra["camera_profile"] == prof
