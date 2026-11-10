"""Follow ONE person when several are in view ("person lock").

MediaPipe can return several people. Each candidate gets a signature:
* appearance — HSV colour histogram of the torso (clothes),
* body proportions — bone-length ratios from the 3D landmarks
  (independent of distance to the camera),
* motion continuity — distance from where the user was a moment ago.

The first lock goes to the biggest, most central person (or to whoever the
user clicks on in the preview; calibration also re-locks to the calibrated
person). Afterwards only a candidate that matches the signature is accepted.
If the user is not found, NOTHING is returned (tracking holds / pauses) —
we never jump to somebody else. Optional: when exactly one person has been in
view for N seconds and nobody matched, lock onto them (the user came back in
different lighting, or swapped clothes).

Everything is computed in memory per frame; nothing is stored or sent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .pose import PoseResult
from .skeleton import (L_ANK, L_EL, L_HIP, L_KNEE, L_SH, L_WR, R_ANK, R_EL, R_HIP, R_KNEE,
                       R_SH, R_WR)

HIST_BINS = [16, 8]
HIST_RANGES = [0, 180, 0, 256]

# bones for the proportion signature (lengths are divided by torso length)
_PROP_BONES = [(L_SH, R_SH), (L_HIP, R_HIP), (L_HIP, L_KNEE), (R_HIP, R_KNEE),
               (L_KNEE, L_ANK), (R_KNEE, R_ANK), (L_SH, L_EL), (R_SH, R_EL),
               (L_EL, L_WR), (R_EL, R_WR)]

MAX_JUMP = 2.0            # while continuously visible: max move, in torso sizes
TRACK_APPEARANCE = 0.60   # while continuously visible: loose colour gate
TRACK_PROPORTIONS = 0.40  # while continuously visible: loose proportion gate
REACQ_APPEARANCE = 0.40   # after a loss: max colour distance (Bhattacharyya, 0..1)
REACQ_PROPORTIONS = 0.25  # after a loss: max mean |log ratio| of bone proportions
CONTINUITY_S = 1.0        # position is trusted this long after the last sighting
# ALONE in view and right where you just were: that's you, even if colours and
# proportions look different — stepping close to the camera crops the body and
# changes both (the lock used to drop you exactly then). Also after a short loss.
SOLO_CONTINUITY_S = 1.5
SOLO_MAX_JUMP = 1.5       # torso sizes (of the bigger of then and now)


@dataclass
class Candidate:
    res: PoseResult
    center: np.ndarray        # hip centre, image coords scaled so height == 1
    size: float               # torso size, same units
    hist: Optional[np.ndarray]
    props: np.ndarray         # bone ratios (nan where not visible)


def _props(res: PoseResult) -> np.ndarray:
    w, v = res.world, res.visibility
    torso = np.linalg.norm((w[L_SH] + w[R_SH]) / 2 - (w[L_HIP] + w[R_HIP]) / 2)
    out = np.full(len(_PROP_BONES), np.nan)
    if torso < 1e-3:
        return out
    for i, (a, b) in enumerate(_PROP_BONES):
        if v[a] > 0.6 and v[b] > 0.6:
            out[i] = np.linalg.norm(w[a] - w[b]) / torso
    return out


def _torso_hist(res: PoseResult, rgb: np.ndarray) -> Optional[np.ndarray]:
    h, w = rgb.shape[:2]
    pts = res.image[[L_SH, R_SH, L_HIP, R_HIP], :2] * np.array([w, h])
    x0, y0 = np.floor(pts.min(axis=0)).astype(int)
    x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
    # shrink to the inner part of the torso (avoid background at the edges)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max((x1 - x0) * 0.35, 3), max((y1 - y0) * 0.4, 3)
    x0, x1 = int(max(cx - hw, 0)), int(min(cx + hw, w))
    y0, y1 = int(max(cy - hh, 0)), int(min(cy + hh, h))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    roi = rgb[y0:y1, x0:x1]
    step = max(1, max(roi.shape[:2]) // 48)        # ~48 px is plenty for a colour histogram
    roi = cv2.cvtColor(np.ascontiguousarray(roi[::step, ::step]), cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([roi], [0, 1], None, HIST_BINS, HIST_RANGES)
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def make_candidates(results: List[PoseResult], rgb: np.ndarray) -> List[Candidate]:
    h, w = rgb.shape[:2]
    scale = np.array([w / h, 1.0])
    out = []
    for r in results:
        img = r.image[:, :2] * scale
        hip = (img[L_HIP] + img[R_HIP]) / 2
        sh = (img[L_SH] + img[R_SH]) / 2
        size = max(float(np.linalg.norm(sh - hip)), float(np.linalg.norm(img[L_SH] - img[R_SH])), 1e-3)
        out.append(Candidate(r, hip, size, _torso_hist(r, rgb), _props(r)))
    return out


def _app_cost(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _prop_cost(a: np.ndarray, b: Optional[np.ndarray]) -> Optional[float]:
    if b is None:
        return None
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if m.sum() < 3:
        return None
    return float(np.mean(np.abs(np.log(a[m] / b[m]))))


class PersonLock:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.sig_hist: Optional[np.ndarray] = None
        self.sig_props: Optional[np.ndarray] = None
        self.last_center: Optional[np.ndarray] = None
        self.velocity = np.zeros(2)
        self.last_size = 0.2
        self.last_seen = -1e9
        self.single_since: Optional[float] = None
        self._pick: Optional[Tuple[float, float]] = None
        self._resign = False

    @property
    def locked(self) -> bool:
        return self.sig_hist is not None or self.sig_props is not None

    def pick(self, x: float, y: float) -> None:
        """User clicked the preview at normalized (x, y): lock onto that person."""
        self._pick = (float(x), float(y))

    def resign(self) -> None:
        """Refresh the signature from the current target (e.g. after calibration)."""
        self._resign = True

    # ------------------------------------------------------------------
    def _lock_to(self, c: Candidate, now: float) -> None:
        self.sig_hist = None if c.hist is None else c.hist.copy()
        self.sig_props = c.props.copy()
        self._seen(c, now)

    def _seen(self, c: Candidate, now: float) -> None:
        if self.last_center is not None and now - self.last_seen < CONTINUITY_S:
            dt = max(now - self.last_seen, 1e-3)
            self.velocity = 0.7 * self.velocity + 0.3 * (c.center - self.last_center) / dt
        else:
            self.velocity = np.zeros(2)
        self.last_center = c.center.copy()
        self.last_size = c.size
        self.last_seen = now

    def _update_signature(self, c: Candidate) -> None:
        if c.hist is not None:
            if self.sig_hist is None:
                self.sig_hist = c.hist.copy()
            else:
                self.sig_hist = 0.92 * self.sig_hist + 0.08 * c.hist
                cv2.normalize(self.sig_hist, self.sig_hist, 1.0, 0.0, cv2.NORM_L1)
        if self.sig_props is None:
            self.sig_props = c.props.copy()
        else:
            m = np.isfinite(c.props)
            old = self.sig_props
            upd = np.where(np.isfinite(old), 0.95 * old + 0.05 * c.props, c.props)
            self.sig_props = np.where(m, upd, old)

    def _costs(self, c: Candidate, now: float) -> Dict[str, Optional[float]]:
        pos = None
        if self.last_center is not None and now - self.last_seen < CONTINUITY_S:
            pred = self.last_center + self.velocity * (now - self.last_seen)
            pos = float(np.linalg.norm(c.center - pred) / max(self.last_size, 0.05))
        return {"pos": pos, "app": _app_cost(c.hist, self.sig_hist),
                "prop": _prop_cost(c.props, self.sig_props)}

    @staticmethod
    def _total(k: Dict[str, Optional[float]]) -> float:
        """Ranking score (lower is better); unknown parts get a neutral penalty."""
        pos = min(k["pos"], 3.0) if k["pos"] is not None else 1.0
        app = k["app"] if k["app"] is not None else 0.5
        prop = k["prop"] if k["prop"] is not None else 0.3
        return 1.0 * pos + 2.0 * app + 1.5 * prop

    @staticmethod
    def _gate(k: Dict[str, Optional[float]], recent: bool) -> bool:
        app, prop = k["app"], k["prop"]
        if recent and k["pos"] is not None:
            # continuously visible: must be close to where the user was AND roughly look alike
            return (k["pos"] < MAX_JUMP and (app is None or app < TRACK_APPEARANCE)
                    and (prop is None or prop < TRACK_PROPORTIONS))
        # coming back after a loss: must clearly look like the user
        if app is None and prop is None:
            return False
        return ((app is None or app < REACQ_APPEARANCE)
                and (prop is None or prop < REACQ_PROPORTIONS))

    def select(self, results: List[PoseResult], rgb: np.ndarray, now: float,
               relock_single_s: float = 3.0) -> Tuple[Optional[PoseResult], Dict]:
        cands = make_candidates(results, rgb)
        info = {"people": len(cands), "locked": self.locked, "status": "", "others": []}
        if not cands:
            self.single_since = None
            info["status"] = "никого нет в кадре"
            return None, info

        chosen: Optional[Candidate] = None
        if self._pick is not None:
            px, py = self._pick
            h, w = rgb.shape[:2]
            p = np.array([px * w / h, py])
            chosen = min(cands, key=lambda c: np.linalg.norm(c.center - p) / max(c.size, 1e-3))
            self._pick = None
            self._lock_to(chosen, now)
            info["status"] = "выбран вручную"
        elif not self.locked:
            # first lock: biggest and most central person
            def score(c: Candidate):
                off = abs(c.center[0] / (rgb.shape[1] / rgb.shape[0]) - 0.5)
                return c.size * (1.0 - off)
            chosen = max(cands, key=score)
            self._lock_to(chosen, now)
            info["status"] = "человек зафиксирован"
        else:
            recent = now - self.last_seen < CONTINUITY_S
            scored = sorted(((self._total(k), i, k, c) for i, c in enumerate(cands)
                             for k in [self._costs(c, now)]), key=lambda x: x[:2])
            passing = [x for x in scored if self._gate(x[2], recent)]
            solo = False
            if not passing and len(cands) == 1 and self.last_center is not None \
                    and now - self.last_seen < (SOLO_CONTINUITY_S if relock_single_s > 0
                                                else CONTINUITY_S):
                c = cands[0]
                jump = float(np.linalg.norm(c.center - self.last_center)
                             / max(self.last_size, c.size, 0.05))
                if jump < SOLO_MAX_JUMP:
                    passing = [scored[0]]
                    solo = True
            if passing:
                best_total, _i, _k, best = passing[0]
                chosen = best
                others = [x[0] for x in scored if x[3] is not best]
                margin = min(others) - best_total if others else 9.9
                if self._resign:
                    self._lock_to(best, now)
                    self._resign = False
                elif not solo and (len(cands) == 1 or margin > 1.0):
                    self._update_signature(best)  # learn slowly, only when unambiguous
                self._seen(best, now)
                info["status"] = "отслеживается"
            elif len(cands) == 1 and relock_single_s > 0:
                if self.single_since is None:
                    self.single_since = now
                if now - self.single_since >= relock_single_s:
                    chosen = cands[0]
                    self._lock_to(chosen, now)
                    info["status"] = "перезахват: в кадре один человек"
                else:
                    info["status"] = "ищу вас…"
            else:
                info["status"] = "ищу вас…"
        if chosen is not None or len(cands) != 1:
            self.single_since = None
        info["locked"] = self.locked
        info["others"] = [c.res.image for c in cands if c is not chosen]
        return (chosen.res if chosen is not None else None), info
