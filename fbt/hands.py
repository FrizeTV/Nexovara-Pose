"""Finger tracking (optional, off by default).

At webcam distance (2–3 m) hands are tiny in the full frame, so — like
MediaPipe Holistic — we crop a square around each wrist using the body pose
and run the Hand Landmarker on that crop. Each hand has its own landmarker in
VIDEO mode, so it tracks between frames instead of re-detecting.

Output per hand: 21 image points (for the preview) and five finger curls
(0 = straight, 1 = fully bent; thumb, index, middle, ring, little).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from .pose import read_model_bytes
from .skeleton import (L_EL, L_INDEX, L_PINKY, L_WR, R_EL, R_INDEX, R_PINKY, R_WR)

log = logging.getLogger(__name__)

# 21-point hand topology
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGERS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
FINGER_NAMES_RU = ["Большой", "Указательный", "Средний", "Безымянный", "Мизинец"]
# total flexion (sum of three joint angles) of a fully bent finger, degrees
FULL_CURL_DEG = [110.0, 230.0, 240.0, 240.0, 230.0]
MIN_CROP_PX = 40


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    return math.degrees(math.acos(float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))


def finger_curls(world: np.ndarray) -> np.ndarray:
    """Curl 0..1 for each finger from 21 3D hand landmarks (any metric frame)."""
    w = np.asarray(world, dtype=np.float64)
    out = np.zeros(5)
    palm_axis = w[9] - w[0]  # wrist -> middle knuckle: fingers fan out, so measure
    for f, (a, b, c, d) in enumerate(FINGERS):  # knuckle flexion against the palm axis
        bones = [w[a] - w[0] if f == 0 else palm_axis, w[b] - w[a], w[c] - w[b], w[d] - w[c]]
        if f == 0:  # thumb: only the two distal joints bend meaningfully
            flex = _angle(bones[1], bones[2]) + _angle(bones[2], bones[3])
        else:
            flex = sum(_angle(bones[i], bones[i + 1]) for i in range(3))
        out[f] = min(max(flex / FULL_CURL_DEG[f], 0.0), 1.0)
    return out


def hand_crop(image_lm: np.ndarray, vis: np.ndarray, side: str, w: int, h: int,
              min_vis: float = 0.3):
    """Square crop (x0, y0, size) in pixels around a wrist, or None."""
    wr, idx, pky, el = ((L_WR, L_INDEX, L_PINKY, L_EL) if side == "left"
                        else (R_WR, R_INDEX, R_PINKY, R_EL))
    if vis[wr] < min_vis:
        return None
    px = image_lm[:, :2] * np.array([w, h])
    knuckles = (px[idx] + px[pky]) / 2
    hand_vec = knuckles - px[wr]
    center = px[wr] + hand_vec * 1.1          # palm centre is past the knuckle markers
    size = max(np.linalg.norm(hand_vec) * 4.5, np.linalg.norm(px[wr] - px[el]) * 0.9)
    size = float(min(size, min(w, h)))
    if not math.isfinite(size) or size < MIN_CROP_PX:
        return None
    x0 = int(round(center[0] - size / 2))
    y0 = int(round(center[1] - size / 2))
    s = int(round(size))
    x0 = min(max(x0, 0), w - s)
    y0 = min(max(y0, 0), h - s)
    if x0 < 0 or y0 < 0:
        return None
    return x0, y0, s


def _is_this_hand(wrist_norm: np.ndarray, pose_image: np.ndarray, vis: np.ndarray, side: str,
                  w: int, h: int, crop_px: int) -> bool:
    """The hand found in a wrist crop must belong to THAT wrist: when both hands
    are close together the crop also contains the other one, and the landmarker
    may pick it — the fingers of one hand would then move the other hand."""
    ours, other = (L_WR, R_WR) if side == "left" else (R_WR, L_WR)
    scale = np.array([w, h], dtype=np.float64)
    p = np.asarray(wrist_norm[:2], dtype=np.float64) * scale
    d_ours = float(np.linalg.norm(p - pose_image[ours, :2] * scale))
    if d_ours > 0.6 * crop_px:
        return False
    if vis[other] >= 0.3:
        d_other = float(np.linalg.norm(p - pose_image[other, :2] * scale))
        if d_other < d_ours:
            return False
    return True


class FingerRange:
    """Per-user range of each finger, learned while you play.

    A relaxed open hand rarely measures exactly 0 (the landmarks are a bit
    noisy and real fingers are slightly bent), so games showed fingers always a
    little curled and the trigger was "touched". The lowest curl seen in the
    last WINDOW_S seconds becomes "open" (at most MAX_BASE)."""

    WINDOW_S = 20.0
    MAX_BASE = 0.25

    def __init__(self):
        self._hist: list = []   # (time, curls)

    def reset(self) -> None:
        self._hist.clear()

    def __call__(self, curls: np.ndarray, now: float) -> np.ndarray:
        c = np.clip(np.asarray(curls, dtype=np.float64), 0.0, 1.0)
        self._hist.append((now, c))
        while self._hist and now - self._hist[0][0] > self.WINDOW_S:
            self._hist.pop(0)
        if len(self._hist) < 10:          # not enough evidence yet: unchanged
            return c
        base = np.minimum(np.min([x for _t, x in self._hist], axis=0), self.MAX_BASE)
        return np.clip((c - base) / (1.0 - base), 0.0, 1.0)


@dataclass
class HandResult:
    image: np.ndarray   # (21, 2) normalized full-frame coordinates
    curls: np.ndarray   # (5,)


class HandTracker:
    def __init__(self, model_path: Path):
        import mediapipe as mp
        self._mp = mp
        vision = mp.tasks.vision
        # model contents, not the path: see pose.read_model_bytes (non-ASCII Windows paths)
        self._model_bytes = read_model_bytes(model_path)

        def make():
            opts = vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_buffer=self._model_bytes),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return vision.HandLandmarker.create_from_options(opts)
        self._lm = {"left": make(), "right": make()}
        self._ts = {"left": -1, "right": -1}

    def process(self, frame_bgr: np.ndarray, pose_image: np.ndarray, vis: np.ndarray,
                ts_ms: int) -> Dict[str, Optional[HandResult]]:
        """`frame_bgr` is the FULL-resolution camera frame: hands are small, so
        they get the sharpest crop; only the crop is converted to RGB."""
        h, w = frame_bgr.shape[:2]
        out: Dict[str, Optional[HandResult]] = {}
        for side in ("left", "right"):
            crop = hand_crop(pose_image, vis, side, w, h)
            if crop is None:
                out[side] = None
                continue
            x0, y0, s = crop
            sub = cv2.cvtColor(frame_bgr[y0:y0 + s, x0:x0 + s], cv2.COLOR_BGR2RGB)
            t = max(int(ts_ms), self._ts[side] + 1)
            self._ts[side] = t
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=sub)
            try:
                res = self._lm[side].detect_for_video(img, t)
            except Exception as e:  # never let finger tracking break body tracking
                log.debug("hand landmarker failed: %s", e)
                out[side] = None
                continue
            if not res.hand_landmarks or not res.hand_world_landmarks:
                out[side] = None
                continue
            lm = res.hand_landmarks[0]
            pts = np.array([[p.x, p.y] for p in lm], dtype=np.float64)
            world = np.array([[p.x, p.y, p.z] for p in res.hand_world_landmarks[0]],
                             dtype=np.float64)
            if pts.shape != (21, 2) or not np.all(np.isfinite(pts)) \
                    or not np.all(np.isfinite(world)):
                out[side] = None
                continue
            full = (pts * s + np.array([x0, y0])) / np.array([w, h])
            if not _is_this_hand(full[0], pose_image, vis, side, w, h, s):
                out[side] = None   # the other hand was in the crop (hands close together)
                continue
            out[side] = HandResult(full, finger_curls(world))
        return out

    def close(self) -> None:
        for lm in self._lm.values():
            try:
                lm.close()
            except Exception:
                pass
