"""The REAL headset and gluing the camera body to it.

Where the headset pose comes from (best first):
  1. the Nexovara Pose driver — it answers every packet of the app with the headset
     pose in SteamVR's raw space (driver/src/protocol.h, "CFBH");
  2. the optional `openvr` package (steamvr.HmdReader), raw space as well.

BodyAnchor puts the camera skeleton (calibrated space: floor y = 0, the user
faced +Z during calibration) where the headset is:
  * yaw: the headset's heading measured during calibration (F6, looking
    straight at the camera); without it a slow automatic estimate is used;
  * position: the camera head is moved onto the headset (eyes -> head centre).
    The offset is One-Euro filtered: at rest the camera's head noise no longer
    shakes the whole body, when you walk it follows the headset with almost
    no lag. (Before, the body was re-glued on every camera frame — every bit of
    head noise moved every tracker.)
"""
from __future__ import annotations

import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .filters import OneEuroFilter
from .mathutil import mat_to_quat, quat_slerp, quat_to_mat, rot_y, yaw_of
from .skeleton import Tracker

FEEDBACK = struct.Struct("<4sHHII3f4f")          # 44 bytes, see protocol.h
FB_VALID, FB_VIRTUAL = 1, 2
# headset (eyes) -> centre of the head (the camera's ear midpoint): back and a
# little down, in the HEAD's own axes — the same geometry the outputs use the
# other way round (outputs.EYE_FORWARD_M / EYE_UP_M)
EYES_TO_HEAD_M = 0.09
EYES_ABOVE_HEAD_M = 0.03
_S = np.diag([1.0, 1.0, -1.0])                    # OpenVR <-> Unity axes


@dataclass
class HmdPose:
    pos: np.ndarray          # Unity axes, SteamVR raw space
    rot: np.ndarray          # 3x3, Unity axes
    t: float                 # time.monotonic() of arrival
    source: str              # "driver" | "openvr" (SteamVR raw space) | "vrchat" (VRChat's space)
    virtual: bool = False    # Nexovara Pose's own virtual headset ("without a headset")


def parse_feedback(data: bytes) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Strict parser of the driver's headset feedback. Returns (pos, rot, flags)
    in Unity axes, or None for anything malformed."""
    if len(data) != FEEDBACK.size:
        return None
    magic, version, flags, _seq, _res, px, py, pz, qx, qy, qz, qw = FEEDBACK.unpack(data)
    if magic != b"CFBH" or version != 1:
        return None
    vals = (px, py, pz, qx, qy, qz, qw)
    if not all(math.isfinite(v) for v in vals) or max(abs(px), abs(py), abs(pz)) > 100.0:
        return None
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not 0.5 < n < 1.5:
        return None
    R_ovr = quat_to_mat((qx / n, qy / n, qz / n, qw / n))
    pos = np.array([px, py, -pz])
    return pos, _S @ R_ovr @ _S, int(flags)


class HmdState:
    """Headset poses, shared between the output thread (driver feedback) and the
    tracking thread (calibration, openvr fallback).

    Besides the newest pose a short HISTORY is kept, so the body seen by the
    camera can be glued to where the headset was AT THE MOMENT THE CAMERA FRAME
    WAS TAKEN (see at()). The headset pose arrives within milliseconds, a camera
    frame is 60–120 ms old by the time it is processed: comparing a fresh
    headset with an old camera head made the whole body slide back and forth
    whenever you moved your head (leaning, nodding, turning)."""

    HISTORY_S = 1.5

    def __init__(self):
        self._lock = threading.Lock()
        self._pose: Optional[HmdPose] = None
        self._hist: deque = deque()

    def update(self, pos, rot, source: str, virtual: bool = False,
               t: Optional[float] = None) -> None:
        p = HmdPose(np.asarray(pos, dtype=np.float64), np.asarray(rot, dtype=np.float64),
                    time.monotonic() if t is None else t, source, virtual)
        with self._lock:
            cur = self._pose
            # the driver is preferred: an openvr pose never replaces a fresh driver one
            if (source != "driver" and cur is not None and cur.source == "driver"
                    and p.t - cur.t < 0.5):
                return
            if cur is not None and (cur.source != p.source or cur.virtual != p.virtual
                                    or p.t < cur.t):
                self._hist.clear()           # another space / clock: don't mix histories
            self._pose = p
            self._hist.append(p)
            while self._hist and p.t - self._hist[0].t > self.HISTORY_S:
                self._hist.popleft()

    def at(self, t: float, max_age: float = 0.5, now: Optional[float] = None
           ) -> Optional[HmdPose]:
        """The headset pose at time `t` (interpolated from the history). The newest
        pose when `t` is outside the history; None if that is stale."""
        newest = self.latest(max_age, now)
        if newest is None:
            return None
        with self._lock:
            hist = list(self._hist)
        if len(hist) < 2 or t >= hist[-1].t or t <= hist[0].t:
            return newest      # outside the history (e.g. a held old pose): where it is now
        lo, hi = 0, len(hist) - 1
        while hi - lo > 1:                       # binary search: hist[lo].t <= t < hist[hi].t
            mid = (lo + hi) // 2
            if hist[mid].t <= t:
                lo = mid
            else:
                hi = mid
        a, b = hist[lo], hist[hi]
        span = b.t - a.t
        u = 0.0 if span <= 1e-9 else (t - a.t) / span
        q = quat_slerp(np.array(mat_to_quat(a.rot)), np.array(mat_to_quat(b.rot)), u)
        return HmdPose(a.pos + (b.pos - a.pos) * u, quat_to_mat(q), t, b.source, b.virtual)

    def latest(self, max_age: float = 0.5, now: Optional[float] = None) -> Optional[HmdPose]:
        with self._lock:
            p = self._pose
        if p is None:
            return None
        now = time.monotonic() if now is None else now
        return p if now - p.t <= max_age else None

    def clear(self) -> None:
        with self._lock:
            self._pose = None
            self._hist.clear()


class BodyAnchor:
    """Rigid transform: camera skeleton -> headset space (see module doc)."""

    AUTO_YAW_TAU = 8.0      # s: automatic heading follows slowly (no calibration yaw)
    SNAP_M = 0.75           # m: a bigger change of the offset is taken at once

    def __init__(self, min_cutoff: float = 0.8, beta: float = 3.0):
        self._off = OneEuroFilter(min_cutoff, beta)
        self._auto_yaw: Optional[float] = None
        self._last_t: Optional[float] = None
        self.last_yaw: Optional[float] = None

    def reset(self) -> None:
        self._off.reset()
        self._auto_yaw = None
        self._last_t = None
        self.last_yaw = None

    @staticmethod
    def head_target(hmd: "HmdPose") -> np.ndarray:
        """Centre of the head behind the headset's eyes."""
        return hmd.pos - hmd.rot[:, 2] * EYES_TO_HEAD_M - hmd.rot[:, 1] * EYES_ABOVE_HEAD_M

    def _yaw(self, calib_yaw: Optional[float], head: Tracker, hmd: HmdPose, t: float) -> float:
        if calib_yaw is not None:
            return float(calib_yaw)
        # no calibration yaw: headset heading minus camera head heading, slowly
        if head.rot is None:
            return self._auto_yaw or 0.0
        diff = yaw_of(hmd.rot[:, 2]) - yaw_of(head.rot[:, 2])
        if self._auto_yaw is None:
            self._auto_yaw = diff
        else:
            d = (diff - self._auto_yaw + 180.0) % 360.0 - 180.0
            dt = 0.0 if self._last_t is None else min(max(t - self._last_t, 0.0), 0.5)
            self._auto_yaw += d * (1.0 - math.exp(-dt / self.AUTO_YAW_TAU))
        return self._auto_yaw

    def apply(self, trackers: List[Tracker], head: Optional[Tracker],
              hands: Dict[str, Tuple[Tracker, Optional[np.ndarray]]], hmd: HmdPose,
              calib_yaw: Optional[float], t: float):
        if head is None:
            return trackers, head, hands
        yaw = self._yaw(calib_yaw, head, hmd, t)
        self._last_t = t
        self.last_yaw = yaw
        R = rot_y(yaw)
        target = self.head_target(hmd)
        raw = target - R @ head.pos
        cur = self._off.value
        if cur is not None and float(np.linalg.norm(raw - cur)) > self.SNAP_M:
            self._off.reset()          # a jump (re-found you, re-calibrated): don't glide
        off = self._off(raw, t)

        def move(tr: Tracker) -> Tracker:
            return Tracker(tr.name, R @ tr.pos + off, None if tr.rot is None else R @ tr.rot)
        return ([move(x) for x in trackers], move(head),
                {s: (move(tr), c) for s, (tr, c) in hands.items()})
