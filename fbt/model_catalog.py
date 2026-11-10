"""Body models offered by the download manager and ensemble settings."""
from __future__ import annotations

import platform
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    id: str
    title: str
    output: str
    runtime: str
    platforms: tuple[str, ...] = ("Windows", "Linux", "Darwin")
    usable_now: bool = True


MODEL_OPTIONS = (
    ModelOption("lite", "MediaPipe Pose Lite", "33 3D", "MediaPipe"),
    ModelOption("full", "MediaPipe Pose Full", "33 3D", "MediaPipe"),
    ModelOption("heavy", "MediaPipe Pose Heavy", "33 3D", "MediaPipe"),
    ModelOption("rtmpose_m", "RTMPose-m", "Halpe-26", "ONNX Runtime"),
    ModelOption("rtmpose_x", "RTMPose-x", "Halpe-26", "ONNX Runtime"),
    ModelOption("movenet_lightning", "MoveNet Lightning", "COCO-17", "ONNX Runtime"),
    ModelOption("movenet_thunder", "MoveNet Thunder", "COCO-17", "ONNX Runtime"),
    ModelOption("rtmo", "RTMO-m", "COCO-17", "ONNX Runtime"),
    ModelOption("vitpose", "ViTPose-S WholeBody", "COCO WholeBody", "ONNX Runtime"),
    ModelOption("hand", "MediaPipe Hand — пальцы", "21 3D", "MediaPipe"),
)
EXTRA_MODELS = {m.id: m.title for m in MODEL_OPTIONS if m.runtime == "ONNX Runtime"}
MAX_EXTRA_MODELS = 4
# Prefer distinct families and relatively small exports when tuning automatically.
AUTO_ORDER = ("movenet_lightning", "rtmpose_m", "rtmo", "vitpose",
              "movenet_thunder", "rtmpose_x")


def platform_name() -> str:
    return platform.system()


def available_options():
    return [m for m in MODEL_OPTIONS if platform_name() in m.platforms]


def clean_selection(value):
    if not isinstance(value, (list, tuple)):
        return []
    return list(dict.fromkeys(x for x in value if isinstance(x, str) and x in EXTRA_MODELS))[
        :MAX_EXTRA_MODELS]
