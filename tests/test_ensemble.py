"""Regression tests for synchronized fusion and optional model failures."""
import numpy as np

from fbt.config import Config
from fbt.ensemble import JointSample, Member, PoseEnsemble, fuse
from fbt.pose import PoseResult
from tests.test_core import standing_pose_mp


def anchor_pose():
    world = standing_pose_mp(1.7)
    image = world.copy()
    image[:, :2] = world[:, :2] * 0.35 + [0.5, 0.25]
    return PoseResult(world, image, np.ones(33))


def test_fusion_uses_only_same_frame_agreeing_joints_and_keeps_depth():
    anchor = anchor_pose()
    wrist, ankle = 15, 27
    time = 8.1
    shift = np.array([0.015, 0.0])
    samples = [
        JointSample(time, {wrist: (anchor.image[wrist, :2] + shift, 0.95),
                           ankle: (anchor.image[ankle, :2] + 0.75, 0.99)}),
        JointSample(time - 0.001, {ankle: (anchor.image[ankle, :2] + shift, 0.95)}),
    ]
    out = fuse(anchor, samples, time, 1, 0.7)
    assert 0 < out.image[wrist, 0] - anchor.image[wrist, 0] < shift[0]
    np.testing.assert_array_equal(out.image[ankle], anchor.image[ankle])
    np.testing.assert_array_equal(out.world[:, 2], anchor.world[:, 2])


def test_manual_uses_four_adapters_and_auto_can_use_only_anchor():
    rgb = np.zeros((480, 640, 3), np.uint8)
    anchor = anchor_pose()
    calls = []

    class Adapter:
        def infer(self, frame, box):
            calls.append(frame)
            count = 26 if self.model_id == "rtmpose_m" else 17
            return anchor.image[:count, :2] * [640, 480], np.full(count, 0.9)

    ids = ["movenet_lightning", "rtmo", "vitpose", "rtmpose_m"]
    cfg = Config(ensemble_models=["removed-model", *ids, "rtmpose_x"]).validate()
    assert cfg.ensemble_models == ids
    members = []
    for key in cfg.ensemble_models:
        adapter = Adapter()
        adapter.model_id = key
        members.append(Member(key, adapter))
    manual = PoseEnsemble(members, False)
    manual.apply(anchor, rgb, 3.0, 0.7, 20, 19)
    assert len(calls) == 4 and all(frame is rgb for frame in calls)
    assert manual.active == ids
    auto = PoseEnsemble(members, True)
    assert auto.apply(anchor, rgb, 4.0, 0.7, 20, 19) is anchor
    assert len(calls) == 4


def test_failing_optional_adapter_preserves_primary_and_stops_retrying():
    anchor = anchor_pose()
    rgb = np.zeros((480, 640, 3), np.uint8)

    class Broken:
        def infer(self, frame, box):
            raise RuntimeError("bad model")

    member = Member("rtmo", Broken())
    ensemble = PoseEnsemble([member], False)
    for frame in range(5):
        assert ensemble.apply(anchor, rgb, float(frame), 0.7, 30, 10) is anchor
    assert member.errors == 3 and ensemble.active == []
