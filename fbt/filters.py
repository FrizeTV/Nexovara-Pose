"""Vectorized One Euro filter (Casiez et al. 2012).

Low jitter when standing still, low lag when moving fast. Works on numpy arrays
of any shape (we filter all 33 landmarks x 3 coordinates at once).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .mathutil import mat_to_quat, quat_angle, quat_slerp, quat_to_mat


def _alpha(dt: float, cutoff):
    tau = 1.0 / (2.0 * math.pi * np.asarray(cutoff))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 1.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x: Optional[np.ndarray] = None
        self._dx: Optional[np.ndarray] = None
        self._t: Optional[float] = None

    def configure(self, min_cutoff: float, beta: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)

    def reset(self) -> None:
        self._x = self._dx = self._t = None

    @property
    def value(self) -> Optional[np.ndarray]:
        return self._x

    def __call__(self, x, t: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None or self._t is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return self._x.copy()
        dt = t - self._t
        if dt <= 1e-6:
            return self._x.copy()
        if dt > 1.0:  # long gap -> restart instead of smearing old data
            self.reset()
            return self(x, t)
        dx = (x - self._x) / dt
        a_d = _alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = _alpha(dt, cutoff)
        x_hat = a * x + (1.0 - a) * self._x
        self._x, self._dx, self._t = x_hat, dx_hat, t
        return x_hat.copy()


# ---------------------------------------------------------------- pose filter
# MediaPipe landmark groups (33 points). Distal joints are noisier (small in the
# image, often occluded) -> they get stronger smoothing than the torso/head.
_TORSO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 23, 24]
_MID = [13, 14, 25, 26]                    # elbows, knees
_HANDS = [15, 16, 17, 18, 19, 20, 21, 22]  # wrists and hand points
_FEET = [27, 28, 29, 30, 31, 32]           # ankles, heels, toes


def _group_scale(torso: float, mid: float, hands: float, feet: float) -> np.ndarray:
    s = np.ones(33)
    s[_TORSO], s[_MID], s[_HANDS], s[_FEET] = torso, mid, hands, feet
    return s


class PoseFilter:
    """Landmark smoothing tuned per body part, with lag compensation.

    1. One Euro per joint; min_cutoff/beta scaled per body group and lowered
       for joints the model is unsure about (visibility).
    2. Lag compensation: the smoothed output is pushed ahead along its own
       (smoothed) velocity — but never past a lightly smoothed measurement,
       so a limb that stops does not overshoot.

    Tuned on recorded MediaPipe output (real photo + webcam-like noise) against
    the previous single One Euro filter. Torso: jitter -17 %, delay -21 ms;
    knees/elbows: -15 %, -21 ms; feet: -10 %, -18 ms, tracking error -11 %;
    hands: unchanged (nothing tested beat the old filter there). No overshoot
    on fast step-and-stop motions.
    Works on (33, 3) arrays in metres."""

    #                       torso  knees/elbows  hands  feet
    MC_SCALE = _group_scale(0.4, 0.4, 1.0, 0.5)
    BETA_SCALE = _group_scale(0.6, 0.6, 1.0, 0.6)
    LEAD_S = _group_scale(0.12, 0.15, 0.0, 0.10)
    VEL_CUTOFF = 1.0      # Hz, smoothing of the velocity used for the lead
    CAP_CUTOFF = 5.0      # Hz, the measurement the lead may not pass
    # sitting / lying: legs (and a little the torso) barely move but are the most
    # occluded and noisy -> calmer min_cutoff there. beta is untouched, so a quick
    # kick or standing up still passes with the usual small delay.
    SIT_SOFT = _group_scale(0.20, 0.40, 0.0, 0.40)
    LIE_SOFT = _group_scale(0.40, 0.50, 0.0, 0.50)

    def __init__(self, min_cutoff: float = 1.2, beta: float = 1.5, lag_comp: bool = True):
        self.min_cutoff, self.beta, self.lag_comp = float(min_cutoff), float(beta), bool(lag_comp)
        self._f = OneEuroFilter(min_cutoff, beta)
        self._soft = np.ones(33)
        self.reset()

    def set_posture(self, sit: float = 0.0, lie: float = 0.0) -> None:
        """Weights 0..1 of sitting / lying (from realism.Realism, already smooth)."""
        sit = float(np.clip(sit, 0.0, 1.0))
        lie = float(np.clip(lie, 0.0, 1.0))
        self._soft = np.clip(1.0 - sit * self.SIT_SOFT - lie * self.LIE_SOFT, 0.3, 1.0)

    @property
    def softness(self) -> np.ndarray:
        return self._soft.copy()

    def configure(self, min_cutoff: float, beta: float, lag_comp: bool = True) -> None:
        self.min_cutoff, self.beta, self.lag_comp = float(min_cutoff), float(beta), bool(lag_comp)

    def reset(self) -> None:
        self._vis: Optional[np.ndarray] = None
        self._prev: Optional[np.ndarray] = None
        self._vel: Optional[np.ndarray] = None
        self._cap: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        self._f.reset()

    @property
    def value(self) -> Optional[np.ndarray]:
        return self._f.value

    def __call__(self, x: np.ndarray, t: float, vis: Optional[np.ndarray] = None) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (33, 3):          # not a pose: plain One Euro
            self._f.min_cutoff, self._f.beta = self.min_cutoff, self.beta
            return self._f(x, t)
        mc = self.min_cutoff * self.MC_SCALE * self._soft
        if vis is not None and len(vis) == 33:
            v = np.clip(np.asarray(vis, dtype=np.float64), 0.0, 1.0)
            self._vis = v if self._vis is None else 0.7 * self._vis + 0.3 * v
            mc = mc * (0.5 + 0.5 * self._vis)
        self._f.min_cutoff = mc[:, None]
        self._f.beta = (self.beta * self.BETA_SCALE)[:, None]
        y = self._f(x, t)
        if self._f.value is None or self._t is None or not (0.0 < t - self._t <= 0.5):
            self._prev, self._vel, self._cap, self._t = y.copy(), None, x.copy(), t
            return y
        dt = t - self._t
        a_v = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * self.VEL_CUTOFF * dt))
        a_c = 1.0 / (1.0 + 1.0 / (2.0 * math.pi * self.CAP_CUTOFF * dt))
        vel = (y - self._prev) / dt
        self._vel = vel if self._vel is None else self._vel + a_v * (vel - self._vel)
        self._cap = self._cap + a_c * (x - self._cap)
        self._prev, self._t = y.copy(), t
        if not self.lag_comp:
            return y
        lead = self._vel * self.LEAD_S[:, None]
        gap = self._cap - y
        same = np.sign(lead) == np.sign(gap)
        return y + np.where(same, np.sign(lead) * np.minimum(np.abs(lead), np.abs(gap)), 0.0)


# ---------------------------------------------------------------- leg swaps
_LEG_L = [25, 27, 29, 31]
_LEG_R = [26, 28, 30, 32]


def swap_legs(P: np.ndarray) -> np.ndarray:
    Q = P.copy()
    Q[_LEG_L], Q[_LEG_R] = P[_LEG_R], P[_LEG_L]
    return Q


class LegSwapGuard:
    """Single-camera models sometimes swap the left and right leg for a few
    frames (legs crossed, side view, fast steps): both foot trackers then jump
    to the other foot — the worst "twitch" in VR. If the swapped assignment
    fits the previous pose much better than the given one, swap back."""

    MIN_ERR = 0.12     # m: below this, any assignment is fine
    RATIO = 0.45       # swapped must be at least this much better

    def __init__(self):
        self.prev: Optional[np.ndarray] = None
        self.swaps = 0

    def reset(self) -> None:
        self.prev = None

    def __call__(self, P: np.ndarray, vis: Optional[np.ndarray] = None):
        """Returns (pose, visibility) — possibly with the legs swapped back."""
        P = np.asarray(P, dtype=np.float64)
        if self.prev is not None and P.shape == (33, 3):
            same = float(np.sum(np.linalg.norm(P[_LEG_L] - self.prev[_LEG_L], axis=1))
                         + np.sum(np.linalg.norm(P[_LEG_R] - self.prev[_LEG_R], axis=1)))
            crossed = float(np.sum(np.linalg.norm(P[_LEG_L] - self.prev[_LEG_R], axis=1))
                            + np.sum(np.linalg.norm(P[_LEG_R] - self.prev[_LEG_L], axis=1)))
            if same > self.MIN_ERR and crossed < self.RATIO * same:
                P = swap_legs(P)
                if vis is not None:
                    vis = np.asarray(vis, dtype=np.float64).copy()
                    vis[_LEG_L], vis[_LEG_R] = vis[_LEG_R].copy(), vis[_LEG_L].copy()
                self.swaps += 1
        self.prev = P.copy()
        return P, vis


# ---------------------------------------------------------------- stabilizer
class Stabilizer:
    """Soft dead-band for tracker positions and rotations (after all filters).

    Movements much smaller than `radius` are mostly ignored, larger ones pass
    unchanged — the tracker stands perfectly still while you stand still, with
    no extra delay when you move (no low-pass involved). Per tracker name."""

    ADAPT = 2.0         # dead-band follows 2x the tracker's own step noise ...
    MAX_GROW = 2.0      # ... up to 2x the base size
    NOISE_ALPHA = 0.05  # ~1 s at 30 fps

    def __init__(self, radius_m: float = 0.004, angle_deg: float = 0.8):
        self.radius, self.angle = float(radius_m), math.radians(angle_deg)
        self._pos: dict = {}
        self._rot: dict = {}
        self._noise: dict = {}
        self._last: dict = {}

    def reset(self) -> None:
        self._pos.clear()
        self._rot.clear()
        self._noise.clear()
        self._last.clear()

    @staticmethod
    def _weight(d: float, r: float) -> float:
        x = d / r if r > 0 else 1.0
        return 1.0 if x >= 1.0 else x * x

    def position(self, name: str, p: np.ndarray, scale: float = 1.0) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        h = self._pos.get(name)
        if h is None or not np.all(np.isfinite(h)):
            self._pos[name] = p.copy()
            self._last[name] = p.copy()
            return p.copy()
        # the tracker's own step noise (small steps only: real motion doesn't count)
        step = float(np.linalg.norm(p - self._last[name]))
        self._last[name] = p.copy()
        base = self.radius * scale
        n = self._noise.get(name, base / self.ADAPT)
        if step < 3.0 * base:
            n += (step - n) * self.NOISE_ALPHA
            self._noise[name] = n
        r = min(max(base, self.ADAPT * n), self.MAX_GROW * base)
        d = p - h
        dist = float(math.sqrt(d @ d))
        h = h + d * self._weight(dist, r)
        self._pos[name] = h
        return h.copy()

    def rotation(self, name: str, R: np.ndarray, scale: float = 1.0) -> np.ndarray:
        q = np.array(mat_to_quat(R))
        h = self._rot.get(name)
        if h is None:
            self._rot[name] = q
            return R
        ang = quat_angle(h, q)
        h = quat_slerp(h, q, self._weight(ang, self.angle * scale))
        self._rot[name] = h
        return quat_to_mat(h)
