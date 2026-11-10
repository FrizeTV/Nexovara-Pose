"""From 33 MediaPipe landmarks to a calibrated virtual skeleton and trackers.

MediaPipe world landmarks: metres, origin at the hip centre, x = image right,
y = image down, z = away from the camera (right-handed). Negating all three
axes gives Unity space for a person facing the camera: +X = person's right,
+Y = up, +Z = person's forward (left-handed), which is what VRChat uses (the driver output mirrors Z for OpenVR).

Accuracy helpers (active once calibrated):
* bone-length constraints — limb lengths measured at calibration are enforced
  every frame (directions come from the model, lengths stay constant). This
  removes most of the "legs stretching" jitter caused by single-camera depth.
* floor clamp — feet never go below the floor.
* foot lock — a foot that rests on the floor stays planted (no skating).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .mathutil import (Y_AXIS, Z_AXIS, basis_from_forward_up, basis_from_right_up,
                       basis_from_up_forward, cross3, is_finite, mat_to_quat, normalize,
                       quat_slerp, quat_to_mat, rotation_between)

NOSE = 0
L_EAR, R_EAR = 7, 8
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28
L_HEEL, R_HEEL = 29, 30
L_TOE, R_TOE = 31, 32
FOOT_IDS = [L_ANK, R_ANK, L_HEEL, R_HEEL, L_TOE, R_TOE]
HEAD_IDS = list(range(0, 11))
EAR_HEIGHT_RATIO = 0.93  # ear height / body height for an average adult

# Bones drawn in the previews
CONNECTIONS = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_WR, L_INDEX), (L_WR, L_PINKY), (L_WR, L_THUMB),
    (R_WR, R_INDEX), (R_WR, R_PINKY), (R_WR, R_THUMB),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (L_ANK, L_HEEL), (L_HEEL, L_TOE), (L_ANK, L_TOE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANK), (R_ANK, R_HEEL), (R_HEEL, R_TOE), (R_ANK, R_TOE),
    (L_EAR, R_EAR), (NOSE, L_EAR), (NOSE, R_EAR),
]

# (parent, child, symmetric-partner-name) chains, ordered from the root outwards
_CHAINS = [
    (L_HIP, L_KNEE, "thigh"), (L_KNEE, L_ANK, "shin"),
    (L_ANK, L_HEEL, "heel"), (L_ANK, L_TOE, "foot"),
    (R_HIP, R_KNEE, "thigh"), (R_KNEE, R_ANK, "shin"),
    (R_ANK, R_HEEL, "heel"), (R_ANK, R_TOE, "foot"),
    (L_SH, L_EL, "upper_arm"), (L_EL, L_WR, "forearm"),
    (L_WR, L_INDEX, "hand_index"), (L_WR, L_PINKY, "hand_pinky"), (L_WR, L_THUMB, "hand_thumb"),
    (R_SH, R_EL, "upper_arm"), (R_EL, R_WR, "forearm"),
    (R_WR, R_INDEX, "hand_index"), (R_WR, R_PINKY, "hand_pinky"), (R_WR, R_THUMB, "hand_thumb"),
]


def mp_to_unity(world: np.ndarray) -> np.ndarray:
    return -np.asarray(world, dtype=np.float64)


def estimate_root(image: np.ndarray, world: np.ndarray, aspect: float,
                  vfov_deg: float) -> Optional[np.ndarray]:
    """Approximate hip-centre position relative to the camera (Unity axes).

    Depth comes from comparing the metric torso length (world landmarks) with
    its length in the image (pinhole camera model).
    """
    f = 0.5 / math.tan(math.radians(vfov_deg) / 2.0)  # focal length, image-height units
    scale = np.array([aspect, 1.0])
    sh_img = (image[L_SH, :2] + image[R_SH, :2]) / 2
    hip_img = (image[L_HIP, :2] + image[R_HIP, :2]) / 2
    sh_w = (world[L_SH] + world[R_SH]) / 2
    hip_w = (world[L_HIP] + world[R_HIP]) / 2
    # use the larger of torso length and shoulder width -> robust to leaning/turning
    seg_img = max(np.linalg.norm((sh_img - hip_img) * scale),
                  np.linalg.norm((image[L_SH, :2] - image[R_SH, :2]) * scale))
    seg_w = max(np.linalg.norm(sh_w - hip_w), np.linalg.norm(world[L_SH] - world[R_SH]))
    if seg_img < 1e-3 or seg_w < 1e-3:
        return None
    depth = f * seg_w / seg_img
    if not (0.3 < depth < 15.0):
        return None
    X = (hip_img[0] - 0.5) * aspect * depth / f
    Y = (hip_img[1] - 0.5) * depth / f
    return -np.array([X, Y, depth])


# ---------------------------------------------------------------- bones
def measure_bones(P: np.ndarray) -> Dict[str, float]:
    """Symmetric (left/right averaged) bone lengths of a pose."""
    sums: Dict[str, List[float]] = {}
    for a, b, name in _CHAINS:
        sums.setdefault(name, []).append(float(np.linalg.norm(P[b] - P[a])))
    L = {k: float(np.mean(v)) for k, v in sums.items()}
    hip_mid = (P[L_HIP] + P[R_HIP]) / 2
    sh_mid = (P[L_SH] + P[R_SH]) / 2
    L["hip_width"] = float(np.linalg.norm(P[R_HIP] - P[L_HIP]))
    L["shoulder_width"] = float(np.linalg.norm(P[R_SH] - P[L_SH]))
    L["torso"] = float(np.linalg.norm(sh_mid - hip_mid))
    L["hip_height"] = float(hip_mid[1])  # meaningful in calibrated space (floor = 0)
    return L


def apply_bone_lengths(P: np.ndarray, L: Dict[str, float]) -> np.ndarray:
    """Keep every bone's direction from the model but force its calibrated length."""
    Q = P.copy()
    hip_mid = (P[L_HIP] + P[R_HIP]) / 2
    d = normalize(P[R_HIP] - P[L_HIP])
    if d is not None:
        Q[L_HIP] = hip_mid - d * L["hip_width"] / 2
        Q[R_HIP] = hip_mid + d * L["hip_width"] / 2
    sh_old = (P[L_SH] + P[R_SH]) / 2
    up = normalize(sh_old - hip_mid)
    sh_mid = hip_mid + up * L["torso"] if up is not None else sh_old
    ds = normalize(P[R_SH] - P[L_SH])
    if ds is not None:
        Q[L_SH] = sh_mid - ds * L["shoulder_width"] / 2
        Q[R_SH] = sh_mid + ds * L["shoulder_width"] / 2
    Q[HEAD_IDS] = P[HEAD_IDS] + (sh_mid - sh_old)
    for a, b, name in _CHAINS:
        u = normalize(P[b] - P[a])
        Q[b] = Q[a] + (u * L[name] if u is not None else (P[b] - P[a]))
    return Q


# ---------------------------------------------------------------- calibration
@dataclass
class Calibration:
    R: np.ndarray            # 3x3 tilt + yaw correction
    scale: float
    offset: np.ndarray       # translation after rotation + scale
    hmd_yaw: Optional[float] = None  # SteamVR HMD yaw at calibration time
    lengths: Dict[str, float] = field(default_factory=dict)  # output-space bone lengths
    root_motion: bool = False
    vrc_yaw: Optional[float] = None  # VRChat's headset yaw at calibration (OSCQuery)

    def apply(self, P: np.ndarray) -> np.ndarray:
        return self.scale * (P @ self.R.T) + self.offset

    def to_dict(self) -> dict:
        return {"R": self.R.reshape(-1).tolist(), "scale": float(self.scale),
                "offset": self.offset.tolist(), "hmd_yaw": self.hmd_yaw,
                "vrc_yaw": self.vrc_yaw,
                "lengths": {k: float(v) for k, v in self.lengths.items()},
                "root_motion": bool(self.root_motion)}

    @staticmethod
    def from_dict(d) -> Optional["Calibration"]:
        try:
            R = np.array(d["R"], dtype=np.float64).reshape(3, 3)
            scale = float(d["scale"])
            off = np.array(d["offset"], dtype=np.float64).reshape(3)
            hy = d.get("hmd_yaw")
            hy = float(hy) if hy is not None and math.isfinite(float(hy)) else None
            vy = d.get("vrc_yaw")
            vy = float(vy) if vy is not None and math.isfinite(float(vy)) else None
            lengths = {}
            raw = d.get("lengths") or {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    v = float(v)
                    if not (math.isfinite(v) and 0.0 < v < 2.0):
                        raise ValueError("bad bone length")
                    lengths[str(k)] = v
            root_motion = bool(d.get("root_motion", False))
        except (KeyError, TypeError, ValueError, AttributeError):
            return None
        if not is_finite(R, off) or not (0.2 < scale < 5.0):
            return None
        if np.abs(R @ R.T - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1) > 1e-3:
            return None
        if np.abs(off).max() > 50:
            return None
        needed = {n for _, _, n in _CHAINS} | {"hip_width", "shoulder_width", "torso"}
        if lengths and not needed.issubset(lengths):
            lengths = {}
        return Calibration(R, scale, off, hy, lengths, root_motion, vy)


def compute_calibration(samples: List[np.ndarray], user_height: float,
                        root_motion: bool = False) -> Calibration:
    """User stands straight, facing the camera, feet on the floor."""
    if not samples:
        raise ValueError("no samples")
    P = np.median(np.stack(samples), axis=0)
    ankle_mid = (P[L_ANK] + P[R_ANK]) / 2
    sh_mid = (P[L_SH] + P[R_SH]) / 2
    # 1) camera pitch/roll: body axis must point straight up
    R1 = rotation_between(sh_mid - ankle_mid, Y_AXIS)
    Q = P @ R1.T
    # 2) yaw: hips' right vector -> +X (so the user faces +Z)
    right = Q[R_HIP] - Q[L_HIP]
    right[1] = 0.0
    R2 = rotation_between(right, np.array([1.0, 0.0, 0.0]))
    R = R2 @ R1
    Q = P @ R.T
    floor = float(np.min(Q[FOOT_IDS, 1]))
    ear_y = float((Q[L_EAR, 1] + Q[R_EAR, 1]) / 2)
    measured = (ear_y - floor) / EAR_HEIGHT_RATIO
    if measured < 0.3:
        raise ValueError("тело видно не целиком (нужны голова и стопы)")
    scale = float(np.clip(user_height / measured, 0.5, 2.0))
    hip_mid = (Q[L_HIP] + Q[R_HIP]) / 2
    offset = -scale * np.array([hip_mid[0], floor, hip_mid[2]])
    cal = Calibration(R, scale, offset, root_motion=root_motion)
    cal.lengths = measure_bones(cal.apply(P))
    return cal


def uncalibrated_transform(P: np.ndarray) -> np.ndarray:
    """Without calibration: just put the lowest foot point on the floor."""
    out = P.copy()
    out[:, 1] -= float(np.min(P[FOOT_IDS, 1]))
    return out


# ---------------------------------------------------------------- feet
class FootStabilizer:
    """Floor clamp + foot planting for both feet (calibrated space, floor y=0)."""
    LOCK_SPEED = 0.12      # m/s: slower than this near the floor -> plant the foot
    UNLOCK_SPEED = 0.35    # m/s: faster than this -> release
    LOCK_HEIGHT = 0.16     # ankle height (m) considered "on the floor"
    MAX_SLIP = 0.08        # m: if the planted foot drifts further -> release
    RELEASE_TAU = 0.08     # s: smooth hand-over after release

    SIDES = ((L_ANK, L_HEEL, L_TOE), (R_ANK, R_HEEL, R_TOE))

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._prev = [None, None]
        self._anchor = [None, None]
        self._offset = [np.zeros(2), np.zeros(2)]
        self.locked = [False, False]

    def apply(self, Q: np.ndarray, dt: float, lock: bool, clamp: bool) -> np.ndarray:
        Q = Q.copy()
        dt = max(dt, 1e-3)
        for s, ids in enumerate(self.SIDES):
            ids = list(ids)
            if clamp:
                low = float(np.min(Q[ids, 1]))
                if low < 0.0:
                    Q[ids, 1] -= low
            if not lock:
                self.locked[s] = False
                self._anchor[s] = None
                self._offset[s][:] = 0
                continue
            xz = Q[ids[0], [0, 2]].copy()
            prev = self._prev[s]
            speed = float(np.linalg.norm(xz - prev) / dt) if prev is not None else 0.0
            self._prev[s] = xz
            height = float(Q[ids[0], 1])
            if self.locked[s]:
                slip = np.linalg.norm(self._anchor[s] - xz)
                if speed > self.UNLOCK_SPEED or height > self.LOCK_HEIGHT * 1.6 \
                        or slip > self.MAX_SLIP:
                    self.locked[s] = False
                    self._anchor[s] = None
                else:
                    self._offset[s] = self._anchor[s] - xz
            elif speed < self.LOCK_SPEED and height < self.LOCK_HEIGHT and prev is not None:
                self.locked[s] = True
                self._anchor[s] = xz + self._offset[s]  # no jump when re-locking
            if not self.locked[s]:
                self._offset[s] *= math.exp(-dt / self.RELEASE_TAU)
            Q[ids, 0] += self._offset[s][0]
            Q[ids, 2] += self._offset[s][1]
        return Q


class FootRotationHold:
    """A planted foot keeps its orientation too (not only its position): while
    the foot stabilizer has it locked, the rotation follows the camera only very
    slowly; released, it catches up within a few frames — continuously, no snap."""
    LOCKED_TAU = 0.8      # s
    FREE_TAU = 0.04       # s

    def __init__(self):
        self._q: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._q.clear()

    def __call__(self, name: str, R: Optional[np.ndarray], locked: bool, dt: float
                 ) -> Optional[np.ndarray]:
        if R is None:
            return None
        q = np.array(mat_to_quat(R))
        h = self._q.get(name)
        if h is None:
            self._q[name] = q
            return R
        tau = self.LOCKED_TAU if locked else self.FREE_TAU
        h = quat_slerp(h, q, 1.0 - math.exp(-max(dt, 0.0) / tau))
        self._q[name] = h
        return quat_to_mat(h)


# ---------------------------------------------------------------- trackers
@dataclass
class Tracker:
    name: str
    pos: np.ndarray
    rot: Optional[np.ndarray]  # None when orientation could not be computed


FOOT_YAW_TO_BODY = 0.25   # foot heading: 25 % from the pelvis (the heel->toe line is short
                          # and its depth noisy; feet mostly point where the body does)
FOOT_SHIN_TILT = 0.2      # foot "up": mostly the world's up, a little of the shin


def _flat(v: np.ndarray) -> Optional[np.ndarray]:
    return normalize(np.array([v[0], 0.0, v[2]]))


def _blend_dirs(a: Optional[np.ndarray], b: Optional[np.ndarray], wb: float) -> Optional[np.ndarray]:
    if a is None:
        return b
    if b is None:
        return a
    if float(a @ b) < 0.0:      # opposite: don't average through zero
        return a
    return normalize(a * (1.0 - wb) + b * wb)


def body_right(P: np.ndarray) -> np.ndarray:
    """Left->right axis of the pelvis, steadied by the shoulders: the hip
    landmarks are the least precise ones of the model and only ~25 cm apart."""
    hips = P[R_HIP] - P[L_HIP]
    sh = P[R_SH] - P[L_SH]
    if float(hips @ sh) < 0.0:      # shoulders turned >90 deg vs hips: ignore them
        return hips
    return hips + 0.35 * sh * (np.linalg.norm(hips) / max(np.linalg.norm(sh), 1e-6))


def foot_rotation(P: np.ndarray, side: str, body_fwd: np.ndarray) -> Optional[np.ndarray]:
    """Orientation of a foot like a real tracker strapped onto it.

    Forward = heel->toe (pitch kept: tiptoes and raised toes are real), heading
    partly stabilised by the pelvis; up = the world's up with a little shin
    tilt. (Before, "up" came from the shin alone, so bending the knees tilted
    the avatar's foot although the real one stayed flat on the floor.)"""
    a, h, t, k = (L_ANK, L_HEEL, L_TOE, L_KNEE) if side == "left" else (R_ANK, R_HEEL, R_TOE, R_KNEE)
    fwd = P[t] - P[h]
    n = normalize(fwd)
    if n is None:
        return None
    pitch_y = float(n[1])
    yaw = _blend_dirs(_flat(fwd), _flat(body_fwd), FOOT_YAW_TO_BODY)
    if yaw is None:
        return None
    horiz = math.sqrt(max(0.0, 1.0 - pitch_y * pitch_y))
    fwd2 = yaw * horiz + np.array([0.0, pitch_y, 0.0])
    shin = normalize(P[k] - P[a])
    up = Y_AXIS if shin is None else normalize(Y_AXIS * (1.0 - FOOT_SHIN_TILT) + shin * FOOT_SHIN_TILT)
    return basis_from_forward_up(fwd2, up if up is not None else Y_AXIS)


def compute_trackers(P: np.ndarray, names: List[str]) -> Dict[str, Tracker]:
    hip_mid = (P[L_HIP] + P[R_HIP]) / 2
    sh_mid = (P[L_SH] + P[R_SH]) / 2
    torso_up = sh_mid - hip_mid
    hip_R = basis_from_right_up(body_right(P), torso_up)
    body_fwd = hip_R[:, 2] if hip_R is not None else Z_AXIS
    out: Dict[str, Tracker] = {}
    for n in names:
        if n == "hip":
            out[n] = Tracker(n, hip_mid, hip_R)
        elif n == "chest":
            out[n] = Tracker(n, hip_mid + 0.7 * torso_up,
                             basis_from_right_up(P[R_SH] - P[L_SH], torso_up))
        elif n in ("left_foot", "right_foot"):
            side = "left" if n == "left_foot" else "right"
            a = L_ANK if side == "left" else R_ANK
            out[n] = Tracker(n, P[a], foot_rotation(P, side, body_fwd))
        elif n in ("left_knee", "right_knee"):
            hp, k = (L_HIP, L_KNEE) if n == "left_knee" else (R_HIP, R_KNEE)
            out[n] = Tracker(n, P[k], basis_from_up_forward(P[hp] - P[k], body_fwd))
        elif n in ("left_elbow", "right_elbow"):
            s, e = (L_SH, L_EL) if n == "left_elbow" else (R_SH, R_EL)
            out[n] = Tracker(n, P[e], basis_from_up_forward(P[s] - P[e], body_fwd))
    return out


def compute_head(P: np.ndarray) -> Tracker:
    ear_mid = (P[L_EAR] + P[R_EAR]) / 2
    sh_mid = (P[L_SH] + P[R_SH]) / 2
    R = basis_from_right_up(P[R_EAR] - P[L_EAR], ear_mid - sh_mid)
    return Tracker("head", ear_mid, R)


class HeadModel:
    """The head when the camera can't see it properly.

    Step close to the camera (or bend down) and the head leaves the picture:
    the network then GUESSES the ears and nose, often wildly. The head position
    matters a lot — VRChat lines up all trackers with it and the SteamVR mode
    glues the body to the headset through it — so a guessed head moved the
    whole body away ("trackers a kilometre away"). Here the head is rebuilt from
    the shoulders (reliable when you're close): neck + the measured
    shoulders-to-head distance along the torso. The two are blended by how well
    the head is seen, so nothing jumps."""

    TAU = 0.25            # s, smoothing of the head's reliability
    LEN_TAU = 3.0         # s, learning of the shoulders-to-head distance

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.ok = 1.0
        self.length: Optional[float] = None

    @staticmethod
    def reliability(vis: np.ndarray, image: Optional[np.ndarray]) -> float:
        ids = [NOSE, L_EAR, R_EAR]
        v = float(np.mean(np.clip(vis[ids], 0.0, 1.0)))
        if image is not None:
            y = float(np.min(image[ids, 1]))
            x0, x1 = float(np.min(image[ids, 0])), float(np.max(image[ids, 0]))
            if y < 0.02 or x0 < 0.01 or x1 > 0.99:          # (partly) out of the picture
                v *= 0.2
        return v

    def __call__(self, head: Tracker, P: np.ndarray, vis: Optional[np.ndarray],
                 image: Optional[np.ndarray], dt: float) -> Tracker:
        if vis is None or len(vis) != 33:
            return head
        dt = min(max(dt, 1e-3), 0.25)
        target = self.reliability(vis, image)
        self.ok += (target - self.ok) * (1.0 - math.exp(-dt / self.TAU))
        neck = (P[L_SH] + P[R_SH]) / 2
        up = normalize(neck - (P[L_HIP] + P[R_HIP]) / 2)
        if up is None:
            return head
        d = float(np.linalg.norm(head.pos - neck))
        if self.ok > 0.8 and 0.08 < d < 0.45:               # head seen well: learn the distance
            self.length = d if self.length is None else \
                self.length + (d - self.length) * (1.0 - math.exp(-dt / self.LEN_TAU))
        w = float(np.clip((self.ok - 0.3) / 0.4, 0.0, 1.0))  # 1 = trust the camera's head
        if w >= 1.0:
            return head
        model_pos = neck + up * (self.length if self.length is not None else 0.22)
        rot = head.rot
        body = basis_from_right_up(P[R_SH] - P[L_SH], up)
        if w <= 0.0 or rot is None:
            rot = body if body is not None else rot
        return Tracker(head.name, head.pos * w + model_pos * (1.0 - w), rot)


def compute_hand(P: np.ndarray, side: str) -> Tracker:
    """Hand (controller) pose from body landmarks: Z along the fingers, Y = back of hand."""
    if side == "left":
        wr, idx, pky, el = L_WR, L_INDEX, L_PINKY, L_EL
    else:
        wr, idx, pky, el = R_WR, R_INDEX, R_PINKY, R_EL
    knuckles = (P[idx] + P[pky]) / 2
    fwd = knuckles - P[wr]
    if normalize(fwd) is None:
        fwd = P[wr] - P[el]
    across = P[idx] - P[pky]  # thumb side
    # back of the hand: left hand -> fwd x across, right hand -> across x fwd
    up = cross3(fwd, across) if side == "left" else cross3(across, fwd)
    return Tracker(f"{side}_hand", P[wr], basis_from_forward_up(fwd, up))
