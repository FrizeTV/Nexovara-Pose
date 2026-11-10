"""Motion realism (ideas from body-estimation tools such as Standable),
adapted to camera tracking. All steps work in calibrated space (metres,
floor at y = 0, user facing +Z) and are blended smoothly — nothing snaps.

* Knee hinge limit — a knee can only bend forwards. Single-camera depth often
  flips the knee behind the hip–ankle line; we mirror it back in front.
* Pose states — standing / sitting / lying are estimated from hip height and
  torso tilt, with smooth (time-constant) transitions and hysteresis.
* Leg estimation (gap filling) — when the camera can't see a leg (desk, frame
  edge, occlusion) the leg is estimated from the hips and the current pose
  state (straight legs when standing, seated legs when sitting) and blended
  in; when the leg is visible again it blends back to the measurement.
* Auto floor — while standing still the floor height is refined slowly, so
  feet neither float nor sink when the camera moves a little.
* Per-tracker status — measured / estimated / held, for the UI.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mathutil import basis_from_right_up, normalize
from .skeleton import (L_ANK, L_EL, L_HEEL, L_HIP, L_KNEE, L_SH, L_TOE, R_ANK, R_EL, R_HEEL,
                       R_HIP, R_KNEE, R_SH, R_TOE)

LEGS = ((L_HIP, L_KNEE, L_ANK, L_HEEL, L_TOE), (R_HIP, R_KNEE, R_ANK, R_HEEL, R_TOE))
UP = np.array([0.0, 1.0, 0.0])
STATE_TAU = 0.30          # s, pose-state blending
EST_TAU = 0.25            # s, leg-estimation blending
VIS_TAU = 0.15            # s, visibility smoothing
FLOOR_TAU = 2.0           # s, auto floor adaptation
FLOOR_LIMIT = 0.25        # m, max auto-floor correction


def _smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _blend(cur: float, target: float, dt: float, tau: float) -> float:
    return cur + (target - cur) * (1.0 - math.exp(-max(dt, 0.0) / tau))


def fix_knees(P: np.ndarray) -> np.ndarray:
    """Mirror backwards-bent knees to the front (hinge joint limit)."""
    Q = P.copy()
    R = basis_from_right_up(P[R_HIP] - P[L_HIP], (P[L_SH] + P[R_SH]) / 2 - (P[L_HIP] + P[R_HIP]) / 2)
    if R is None:
        return Q
    fwd = R[:, 2]
    for hip, knee, ank, *_ in LEGS:
        axis = normalize(P[ank] - P[hip])
        if axis is None:
            continue
        rel = P[knee] - P[hip]
        closest = P[hip] + axis * float(np.dot(rel, axis))
        d = P[knee] - closest
        # forward direction perpendicular to the leg axis
        f = normalize(fwd - axis * float(np.dot(fwd, axis)))
        if f is None:
            continue
        back = float(np.dot(d, f))
        if back < -0.01:                       # knee is behind the leg line
            Q[knee] = P[knee] - 2.0 * back * f  # mirror to the front
    return Q


def estimate_legs(P: np.ndarray, L: Dict[str, float], sit: float, lie: float) -> np.ndarray:
    """Plausible leg points from the hips: standing -> straight down,
    sitting -> thighs forward, shins down; lying -> along the torso."""
    E = P.copy()
    R = basis_from_right_up(P[R_HIP] - P[L_HIP], (P[L_SH] + P[R_SH]) / 2 - (P[L_HIP] + P[R_HIP]) / 2)
    if R is None:
        return E
    torso_up = R[:, 1]
    fwd3 = R[:, 2]
    fwd_flat = normalize(np.array([fwd3[0], 0.0, fwd3[2]]))
    if fwd_flat is None:
        fwd_flat = np.array([0.0, 0.0, 1.0])
    down = -UP
    stand = max(0.0, 1.0 - sit - lie)
    thigh, shin = L.get("thigh", 0.42), L.get("shin", 0.42)
    heel, foot = L.get("heel", 0.07), L.get("foot", 0.16)
    for hip, knee, ank, hl, toe in LEGS:
        H = P[hip]
        k_stand = H + down * thigh
        k_sit = H + fwd_flat * thigh
        k_lie = H - torso_up * thigh
        E[knee] = stand * k_stand + sit * k_sit + lie * k_lie
        a_stand = k_stand + down * shin
        a_sit = k_sit + down * shin
        a_lie = k_lie - torso_up * shin
        # keep the bone lengths after blending
        kd = normalize(E[knee] - H)
        if kd is not None:
            E[knee] = H + kd * thigh
        a_dir = normalize(stand * (a_stand - k_stand) + sit * (a_sit - k_sit) + lie * (a_lie - k_lie))
        E[ank] = E[knee] + (a_dir if a_dir is not None else down) * shin
        foot_fwd = normalize(fwd_flat * (1 - lie) + fwd3 * lie) if lie < 1 else fwd3
        if foot_fwd is None:
            foot_fwd = fwd_flat
        E[hl] = E[ank] + normalize(down * 0.8 - foot_fwd * 0.6) * heel
        E[toe] = E[ank] + normalize(foot_fwd * 0.9 + down * 0.45) * foot
    return E


@dataclass
class RealismInfo:
    state: str = "стоит"
    weights: Dict[str, float] = field(default_factory=lambda: {"stand": 1.0, "sit": 0.0, "lie": 0.0})
    legs_estimated: List[float] = field(default_factory=lambda: [0.0, 0.0])
    statuses: Dict[str, str] = field(default_factory=dict)
    floor_offset: float = 0.0


class Realism:
    def __init__(self):
        self.reset()

    def reset(self, keep_floor: bool = False) -> None:
        """keep_floor: the learned floor level survives (a moment out of view or a
        settings change must not make the feet jump up or down when you're back)."""
        self.vis: Optional[np.ndarray] = None
        self.w = {"stand": 1.0, "sit": 0.0, "lie": 0.0}
        self.est = [0.0, 0.0]
        self._est_target = [0.0, 0.0]
        self.floor = getattr(self, "floor", 0.0) if keep_floor else 0.0
        self._prev_foot_y: Optional[float] = None

    def process(self, P: np.ndarray, vis: np.ndarray, dt: float, L: Dict[str, float], *,
                states: bool = True, legs: bool = True, auto_floor: bool = True
                ) -> Tuple[np.ndarray, RealismInfo]:
        dt = min(max(dt, 1e-3), 0.25)
        v = np.asarray(vis, dtype=np.float64)
        self.vis = v.copy() if self.vis is None else self.vis + (v - self.vis) * (
            1.0 - math.exp(-dt / VIS_TAU))
        P = P.copy()

        # ---- pose state
        hip_mid = (P[L_HIP] + P[R_HIP]) / 2
        sh_mid = (P[L_SH] + P[R_SH]) / 2
        up = normalize(sh_mid - hip_mid)
        tilt = math.degrees(math.acos(float(np.clip(up @ UP, -1, 1)))) if up is not None else 0.0
        h0 = L.get("hip_height") or (L.get("thigh", 0.42) + L.get("shin", 0.42) + 0.08)
        ratio = float(hip_mid[1]) / max(h0, 0.3)
        if states:
            lie_t = _smoothstep(45.0, 65.0, tilt)
            sit_t = (1.0 - lie_t) * _smoothstep(0.86, 0.70, ratio)
            targets = {"lie": lie_t, "sit": sit_t, "stand": max(0.0, 1.0 - lie_t - sit_t)}
            for k in self.w:
                self.w[k] = _blend(self.w[k], targets[k], dt, STATE_TAU)
            tot = sum(self.w.values()) or 1.0
            for k in self.w:
                self.w[k] /= tot
        else:
            self.w = {"stand": 1.0, "sit": 0.0, "lie": 0.0}

        # ---- leg estimation (gap filling) with hysteresis
        for i, (hip, knee, ank, hl, toe) in enumerate(LEGS):
            leg_vis = float(np.mean(self.vis[[knee, ank, hl, toe]]))
            if not legs:
                self._est_target[i] = 0.0
            elif leg_vis < 0.35:
                self._est_target[i] = 1.0
            elif leg_vis > 0.55:
                self._est_target[i] = 0.0
            self.est[i] = _blend(self.est[i], self._est_target[i], dt, EST_TAU)
        if max(self.est) > 1e-3:
            E = estimate_legs(P, L, self.w["sit"], self.w["lie"])
            for i, ids in enumerate(LEGS):
                k = self.est[i]
                if k > 1e-3:
                    ids = list(ids[1:])
                    P[ids] = P[ids] * (1 - k) + E[ids] * k

        # ---- auto floor (only while standing still on measured feet)
        feet = [L_ANK, L_HEEL, L_TOE, R_ANK, R_HEEL, R_TOE]
        foot_y = float(np.min(P[feet, 1]))
        still = self._prev_foot_y is not None and abs(foot_y - self._prev_foot_y) / dt < 0.10
        self._prev_foot_y = foot_y
        if auto_floor and still and self.w["stand"] > 0.9 and max(self.est) < 0.1:
            target = float(np.clip(-foot_y, -FLOOR_LIMIT, FLOOR_LIMIT))  # lowest foot -> y=0
            self.floor = _blend(self.floor, target, dt, FLOOR_TAU)
        elif not auto_floor:
            self.floor = 0.0
        P[:, 1] += self.floor

        info = RealismInfo(state=max(self.w, key=self.w.get), weights=dict(self.w),
                           legs_estimated=list(self.est), floor_offset=self.floor)
        info.state = {"stand": "стоит", "sit": "сидит", "lie": "лежит"}[info.state]
        info.statuses = self._statuses()
        return P, info

    def _statuses(self) -> Dict[str, str]:
        v = self.vis
        st = {}

        def m(ids):
            return "measured" if float(np.mean(v[ids])) >= 0.5 else "held"
        st["hip"] = m([L_HIP, R_HIP])
        st["chest"] = m([L_SH, R_SH])
        st["left_elbow"] = m([L_EL])
        st["right_elbow"] = m([R_EL])
        for i, side in enumerate(("left", "right")):
            hip, knee, ank, hl, toe = LEGS[i]
            if self.est[i] > 0.5:
                s = "estimated"
            else:
                s = m([knee, ank, hl, toe])
            st[f"{side}_foot"] = s
            st[f"{side}_knee"] = s
        return st
