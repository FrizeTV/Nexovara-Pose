"""Frame-local ensemble with MediaPipe as the 3D and identity anchor."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import rtmpose
from .model_catalog import EXTRA_MODELS
from .pose import PoseResult

log = logging.getLogger(__name__)
COCO_TO_MP = {k: m for k, m in rtmpose.HALPE_TO_MP.items() if k < 17}


@dataclass(frozen=True)
class JointSample:
    timestamp: float
    points: dict


def fuse(anchor: PoseResult, samples: list[JointSample], timestamp: float,
         aspect: float, strength: float) -> PoseResult:
    """Combine agreeing measurements from this exact capture, preserving depth."""
    scale = rtmpose.world_scale(anchor, aspect)
    if not samples or not strength or scale is None:
        return anchor
    axes = np.array([aspect, 1.0])
    shoulders = anchor.image[[11, 12], :2].mean(axis=0)
    hips = anchor.image[[23, 24], :2].mean(axis=0)
    torso = np.linalg.norm((shoulders - hips) * axes)
    if not np.isfinite(torso) or torso < 1e-3:
        return anchor

    offsets = np.zeros((33, 2))
    weights = np.zeros(33)
    counts = np.zeros(33)
    for sample in samples:
        if sample.timestamp != timestamp:
            continue
        for joint, (xy, confidence) in sample.points.items():
            if not (0 <= joint < 33 and np.isfinite(xy).all() and np.isfinite(confidence)):
                continue
            delta = xy - anchor.image[joint, :2]
            # Tighter gate at the shoulders/hips protects torso orientation.
            gate = torso * (0.16 if joint in (11, 12, 23, 24) else 0.30)
            distance = np.linalg.norm(delta * axes)
            if distance >= gate:
                continue
            weight = np.clip((confidence - 0.30) / 0.40, 0, 1) ** 2
            weight *= 1 - (distance / gate) ** 2
            if weight > 0:
                offsets[joint] += delta * weight
                weights[joint] += weight
                counts[joint] += 1
    if not weights.any():
        return anchor
    # Adding more copies of the same estimate should not increase correction strength.
    prior = 0.35 + 0.65 * np.clip(anchor.visibility, 0, 1)
    shift = strength * offsets / (weights + prior * np.maximum(counts, 1))[:, None]
    image, world = anchor.image.copy(), anchor.world.copy()
    image[:, :2] += shift
    world[:, :2] += shift * axes * scale
    # Keep the hip centre of the metric skeleton fixed.
    world[:, :2] -= (world[[23, 24], :2] - anchor.world[[23, 24], :2]).mean(axis=0)
    return PoseResult(world, image, anchor.visibility)


@dataclass
class Member:
    id: str
    adapter: object
    ms: Optional[float] = None
    errors: int = 0
    measured: int = 0

    def infer(self, rgb, box, timestamp):
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        xy, scores = self.adapter.infer(rgb, box)
        dt = (time.perf_counter() - t0) * 1000
        self.measured += 1
        if self.measured > 1:  # exclude the session's cold first inference
            self.ms = dt if self.ms is None else 0.85 * self.ms + 0.15 * dt
        self.errors = 0
        if self.id in rtmpose.REFINERS:
            points = rtmpose.mapped_points(xy, scores, w, h)
        else:
            points = {m: (xy[k] / [w, h], float(scores[k]))
                      for k, m in COCO_TO_MP.items() if k < len(scores)}
            if self.id == "vitpose" and len(scores) == 133:
                for m, k in ((29, 19), (30, 22), (31, 17), (32, 20)):
                    points[m] = (xy[k] / [w, h], float(scores[k]))
        return JointSample(timestamp, points)


class PoseEnsemble:
    def __init__(self, members, automatic):
        self.members = members
        self.automatic = automatic
        self.active = []
        self.base_ms = None

    def apply(self, anchor, rgb, timestamp, strength, budget_ms, base_ms):
        self.active = []
        if anchor is None or not strength:
            return anchor
        h, w = rgb.shape[:2]
        box = rtmpose.bbox_from_landmarks(anchor.image, anchor.visibility, w, h)
        if box is None:
            return anchor
        self.base_ms = (base_ms if self.base_ms is None else
                        0.9 * self.base_ms + 0.1 * base_ms)
        remaining = budget_ms - max(base_ms, self.base_ms)
        samples = []
        for member in self.members:
            if member.errors >= 3:
                continue
            estimate = max(6.0, 1.2 * (member.ms or 0))
            if self.automatic and estimate > remaining:
                continue
            try:
                t0 = time.perf_counter()
                sample = member.infer(rgb, box, timestamp)
                if sample.points:
                    samples.append(sample)
                    self.active.append(member.id)
            except Exception:
                member.errors += 1
                log.warning("ONNX %s failed (%d/3)", member.id, member.errors, exc_info=True)
            remaining -= (time.perf_counter() - t0) * 1000
            # Auto mode samples at most one as-yet unmeasured model per frame.
            if self.automatic and member.ms is None:
                break
        return fuse(anchor, samples, timestamp, w / h, strength)

    def describe(self):
        names = ", ".join(EXTRA_MODELS[m] for m in self.active)
        return f"Ансамбль: {1 + len(self.active)}/{1 + len(self.members)} · {names or 'MediaPipe'}"

    def close(self):
        for member in self.members:
            member.adapter.close()
