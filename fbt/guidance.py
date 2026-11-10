"""How sure we are about every tracker, and what to change so tracking gets better.

TrackerConfidence — 0..1 per tracker, from what the camera actually sees:
  * visibility of the joints the tracker is built from (MediaPipe's own score),
  * whether those joints are inside the picture (points at the edge are guessed),
  * whether the tracker is measured, estimated (legs filled in) or held,
  * extra cameras that confirm the joint (fused 3D) raise it.
  Smoothed over ~0.3 s, so the bars in the window don't flicker.

Guide — short advice for the user (stand further away, more light, ...).
  A tip appears only after its condition has held for a moment and stays for a
  couple of seconds, so it never blinks. Only numbers of the current frame are
  used; nothing is stored.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .skeleton import (L_ANK, L_EAR, L_EL, L_HEEL, L_HIP, L_KNEE, L_SH, L_TOE, L_WR, NOSE,
                       R_ANK, R_EAR, R_EL, R_HEEL, R_HIP, R_KNEE, R_SH, R_TOE, R_WR)

TRACKER_JOINTS: Dict[str, List[int]] = {
    "hip": [L_HIP, R_HIP],
    "chest": [L_SH, R_SH],
    "left_foot": [L_ANK, L_HEEL, L_TOE],
    "right_foot": [R_ANK, R_HEEL, R_TOE],
    "left_knee": [L_KNEE],
    "right_knee": [R_KNEE],
    "left_elbow": [L_EL],
    "right_elbow": [R_EL],
    "head": [NOSE, L_EAR, R_EAR],
    "left_hand": [L_WR],
    "right_hand": [R_WR],
}
LABELS_RU = {
    "hip": "таз", "chest": "грудь", "left_foot": "левая стопа", "right_foot": "правая стопа",
    "left_knee": "левое колено", "right_knee": "правое колено", "left_elbow": "левый локоть",
    "right_elbow": "правый локоть", "head": "голова", "left_hand": "левая рука",
    "right_hand": "правая рука",
}
EDGE = 0.02            # normalized image margin: closer to the border = "out of frame"
CONF_TAU = 0.3         # s, smoothing of the shown confidence
STATUS_CAP = {"measured": 1.0, "estimated": 0.35, "held": 0.2}


def _in_frame(p: np.ndarray) -> float:
    """1 inside the picture, falls to 0 within EDGE outside of it."""
    x, y = float(p[0]), float(p[1])
    d = min(x, y, 1.0 - x, 1.0 - y)          # distance to the nearest border
    return float(np.clip((d + EDGE) / (2 * EDGE), 0.0, 1.0))


def raw_confidence(name: str, vis: np.ndarray, image: Optional[np.ndarray],
                   status: Optional[str] = None, fused_views: int = 1) -> float:
    ids = TRACKER_JOINTS.get(name)
    if not ids or vis is None:
        return 0.0
    v = float(np.mean(np.clip(np.asarray(vis)[ids], 0.0, 1.0)))
    if image is not None:
        v *= float(np.mean([_in_frame(image[i]) for i in ids]))
    if fused_views >= 2:                       # another camera confirms it
        v = min(1.0, v + 0.15 * (fused_views - 1))
    if status in STATUS_CAP:
        v = min(v, STATUS_CAP[status]) if status != "measured" else v
    return float(np.clip(v, 0.0, 1.0))


class TrackerConfidence:
    def __init__(self):
        self.values: Dict[str, float] = {}
        self._t: Optional[float] = None

    def reset(self) -> None:
        self.values.clear()
        self._t = None

    def update(self, names: Sequence[str], vis: Optional[np.ndarray], image: Optional[np.ndarray],
               statuses: Optional[Dict[str, str]], fused_views: int, t: float
               ) -> Dict[str, float]:
        dt = 0.0 if self._t is None else min(max(t - self._t, 0.0), 1.0)
        self._t = t
        k = 1.0 if not self.values else 1.0 - math.exp(-dt / CONF_TAU)
        out = {}
        for n in names:
            target = 0.0 if vis is None else raw_confidence(
                n, vis, image, (statuses or {}).get(n), fused_views)
            cur = self.values.get(n)
            cur = target if cur is None else cur + (target - cur) * k
            out[n] = cur
        self.values = out
        return dict(out)


# ---------------------------------------------------------------- advice
def _body_extent(image: np.ndarray, vis: np.ndarray) -> Optional[Tuple[float, float]]:
    """Top (head) and bottom (feet) of the body in normalized image y."""
    top = min(float(image[NOSE, 1]), float(min(image[L_EAR, 1], image[R_EAR, 1])))
    head_extra = abs(float(image[L_SH, 1] + image[R_SH, 1]) / 2 - top) * 0.45
    top -= head_extra                              # crown is above the nose/ears
    feet = [L_HEEL, R_HEEL, L_TOE, R_TOE, L_ANK, R_ANK]
    bottom = max(float(image[i, 1]) for i in feet)
    if not (math.isfinite(top) and math.isfinite(bottom)) or bottom <= top:
        return None
    return top, bottom


def advice(image: Optional[np.ndarray], vis: Optional[np.ndarray], aspect: float,
           brightness: Optional[float], cam_fps: float, is_camera: bool,
           calibrated: bool, confidences: Dict[str, float]) -> List[Tuple[str, str]]:
    """Current problems as (key, text), most important first."""
    out: List[Tuple[str, str]] = []
    if brightness is not None and brightness < 45:
        out.append(("dark", "Мало света — включите свет: в темноте камера снижает частоту "
                            "кадров и сильно шумит."))
    if is_camera and 0 < cam_fps < 20:
        out.append(("fps", f"Камера даёт всего {cam_fps:.0f} кадров/с — добавьте света или "
                           "выберите режим MJPG (Настройки → Камера)."))
    if image is None or vis is None:
        return out
    feet = [L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE]
    feet_out = sum(1 for i in feet if image[i, 1] > 1.0 - EDGE or vis[i] < 0.3)
    if feet_out >= 4:
        out.append(("feet", "Не видно стоп — отойдите дальше от камеры или наклоните её "
                            "вниз, чтобы ноги были в кадре целиком."))
    if image[NOSE, 1] < EDGE or (vis[NOSE] < 0.3 and image[NOSE, 1] < 0.1):
        out.append(("head", "Голова выходит за кадр — отойдите дальше или поднимите "
                            "камеру."))
    ext = _body_extent(image, vis)
    if ext is not None and not any(k in ("feet", "head") for k, _ in out):
        height = ext[1] - ext[0]
        if height < 0.4:
            out.append(("far", f"Вы далеко (занимаете {height * 100:.0f} % высоты кадра) — "
                               "подойдите на шаг-два ближе: точность выше, когда вы "
                               "занимаете 60–90 % кадра."))
    hip_x = float(image[L_HIP, 0] + image[R_HIP, 0]) / 2
    if hip_x < 0.15 or hip_x > 0.85:
        out.append(("center", "Сместитесь ближе к центру кадра — у края руки и ноги "
                              "выходят из кадра."))
    if not calibrated:
        sh = float(np.hypot((image[R_SH, 0] - image[L_SH, 0]) * aspect,
                            image[R_SH, 1] - image[L_SH, 1]))
        torso = float(np.hypot(((image[L_SH, 0] + image[R_SH, 0]) - (image[L_HIP, 0]
                                                                     + image[R_HIP, 0])) / 2 * aspect,
                               ((image[L_SH, 1] + image[R_SH, 1]) - (image[L_HIP, 1]
                                                                     + image[R_HIP, 1])) / 2))
        if torso > 1e-3 and sh / torso < 0.45:
            out.append(("side", "Для калибровки встаньте лицом к камере (сейчас вы стоите "
                                "боком)."))
    weak = [LABELS_RU.get(n, n) for n, c in confidences.items()
            if c < 0.35 and n in TRACKER_JOINTS and not n.endswith("_hand") and n != "head"]
    if weak and not any(k in ("feet", "head") for k, _ in out):
        out.append(("weak", "Плохо видно: " + ", ".join(weak[:3]) +
                            " — не закрывайте их и не уходите к краю кадра."))
    return out


class Guide:
    """Debounced advice: a tip is shown after its problem lasted SHOW_AFTER
    seconds and stays at least KEEP seconds (no blinking)."""
    SHOW_AFTER = 0.8
    KEEP = 2.5

    def __init__(self):
        self._since: Dict[str, float] = {}
        self._shown: Dict[str, Tuple[str, float]] = {}

    def reset(self) -> None:
        self._since.clear()
        self._shown.clear()

    def update(self, items: List[Tuple[str, str]], t: float) -> List[str]:
        keys = [k for k, _ in items]
        for k in list(self._since):
            if k not in keys:
                del self._since[k]
        order = []
        for k, text in items:
            start = self._since.setdefault(k, t)
            if t - start >= self.SHOW_AFTER:
                self._shown[k] = (text, t)
            order.append(k)
        out = []
        for k in order:                              # current problems first, in priority
            if k in self._shown:
                out.append(self._shown[k][0])
        for k, (text, last) in list(self._shown.items()):
            if k in order:
                continue
            if t - last <= self.KEEP:                # recently fixed: keep a little longer
                out.append(text)
            else:
                del self._shown[k]
        return out
