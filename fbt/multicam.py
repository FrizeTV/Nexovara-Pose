"""Several cameras at once (USB webcams, phones as webcams via DroidCam/Iriun/Camo).

Why it is more accurate
-----------------------
A single camera measures a joint well ACROSS its view (x/y) but poorly ALONG
its viewing ray (depth). With two cameras at an angle, each one's precise
directions cover the other's depth error. We fuse per joint with an
information-weighted mean: every camera contributes a 3D estimate with an
anisotropic covariance (small across the view, ~3x larger in depth, larger
when the joint is barely visible); the fused point is
    x = (Σ Λ_c)⁻¹ Σ Λ_c x_c,   Λ_c = R_c D_c⁻¹ R_cᵀ
— the maximum-likelihood combination of Gaussian measurements.

Camera rig calibration (no checkerboard)
----------------------------------------
MediaPipe's 3D "world" landmarks are centred on the hips and expressed in
each camera's orientation. So the same body seen by two cameras differs only
by the cameras' relative ROTATION. While the user moves for a few seconds we
collect corresponding joints and solve the weighted Kabsch problem (SVD) for
the rotation aux→main. No translation is needed because the frames share the
hip origin.

Each extra camera runs in its own thread with its own model instance; the
inference runs in native code, so cameras are processed truly in parallel.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .camera import CameraSource, best_from_profile, is_video_file, profile_for
from .person import PersonLock
from .pose import PoseEstimator, PoseResult, make_estimator, prepare_rgb

log = logging.getLogger(__name__)

MAX_EXTRA_CAMERAS = 3
MAX_AGE_S = 0.15          # older aux results are ignored for fusion
DEPTH_FACTOR = 3.0        # depth is ~3x less precise than the in-image directions
AGREE_LIMIT_M = 0.30      # aux pose must agree with the main pose (wrong person -> ignored)
MIN_RIG_FRAMES = 45


# --------------------------------------------------------------------- math
def kabsch(src: np.ndarray, dst: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Rotation R minimising Σ w‖R·src_i − dst_i‖² (both point sets centred already)."""
    if w is None:
        w = np.ones(len(src))
    H = (src * w[:, None]).T @ dst
    U, _S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def _info(vis: np.ndarray, R: np.ndarray) -> np.ndarray:
    """(33,3,3) information matrices of one camera's joints in the main frame."""
    # metres, across the view: 2 cm for a clearly visible joint, growing fast as
    # visibility drops (vis 0.5 -> 6 cm, vis 0.05 -> 1.8 m = practically ignored)
    sigma = 0.02 / np.clip(vis, 0.05, 1.0) ** 1.5
    inv = np.zeros((len(vis), 3, 3))
    inv[:, 0, 0] = inv[:, 1, 1] = 1.0 / sigma ** 2
    inv[:, 2, 2] = 1.0 / (DEPTH_FACTOR * sigma) ** 2          # z = viewing direction
    return np.einsum("ij,njk,lk->nil", R, inv, R)             # R · D⁻¹ · Rᵀ


def fuse(views: List[Tuple[np.ndarray, PoseResult, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """views: (R_to_main, result, weight). Returns fused world (33,3) + visibility (33,)."""
    A = np.zeros((33, 3, 3))
    b = np.zeros((33, 3))
    vis = np.zeros(33)
    for R, res, weight in views:
        x = res.world @ R.T
        L = _info(res.visibility, R) * weight
        A += L
        b += np.einsum("nij,nj->ni", L, x)
        vis = np.maximum(vis, res.visibility)
    return np.linalg.solve(A, b[..., None])[..., 0], vis


def agreement(R: np.ndarray, aux: PoseResult, main: PoseResult) -> float:
    """How far (m) the rotated aux pose is from the main pose: the 80th percentile
    of per-joint distances over body joints both cameras see. A mean would hide a
    different person whose torso happens to line up; the percentile reacts when a
    good share of the limbs disagree, while depth noise stays well below the limit."""
    body = np.arange(11, 33)
    m = (aux.visibility[body] > 0.6) & (main.visibility[body] > 0.6)
    if m.sum() < 6:
        return 0.0
    idx = body[m]
    d = np.linalg.norm(aux.world[idx] @ R.T - main.world[idx], axis=1)
    return float(np.percentile(d, 80))


class RigCalibrator:
    """Collects corresponding poses and solves each extra camera's rotation."""

    def __init__(self, sources: List[str], seconds: float):
        self.deadline = time.monotonic() + seconds
        self.seconds = seconds
        self.pairs: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {s: [] for s in sources}

    @property
    def progress(self) -> float:
        return 1.0 - max(0.0, self.deadline - time.monotonic()) / self.seconds

    def add(self, main: PoseResult, aux: Dict[str, PoseResult]) -> None:
        for src, res in aux.items():
            if src in self.pairs:
                w = np.minimum(main.visibility, res.visibility)
                if (w > 0.7).sum() >= 10:
                    self.pairs[src].append((res.world.copy(), main.world.copy(), w))

    def done(self) -> bool:
        return time.monotonic() >= self.deadline

    def solve(self) -> Dict[str, Dict]:
        out = {}
        for src, pairs in self.pairs.items():
            if len(pairs) < MIN_RIG_FRAMES:
                out[src] = {"error": f"мало кадров, где вас видят обе камеры ({len(pairs)})"}
                continue
            a = np.concatenate([p[0] for p in pairs])
            m = np.concatenate([p[1] for p in pairs])
            w = np.concatenate([p[2] for p in pairs])
            keep = w > 0.7
            R = kabsch(a[keep], m[keep], w[keep])
            rms = float(np.sqrt(np.average(np.sum((a[keep] @ R.T - m[keep]) ** 2, axis=1),
                                           weights=w[keep])))
            out[src] = {"R": R.reshape(-1).tolist(), "rms_cm": round(rms * 100, 1),
                        "frames": len(pairs), "time": int(time.time())}
        return out


def rotation_angle_deg(A: np.ndarray, B: np.ndarray) -> float:
    c = (float(np.trace(A.T @ B)) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


class AutoRig:
    """Camera rig calibration in the background — no special movements.

    While you simply use the program, every frame where BOTH cameras see you
    clearly (and nobody else is in view) is collected. Because MediaPipe's 3D
    joints are hip-centred and in each camera's orientation, even an ordinary
    standing pose fixes the rotation between the cameras (any 3+ joints not on
    one line do); a couple of seconds of normal moving makes it robust.

    * a camera without calibration gets one after ~2 s of good frames;
    * a calibrated camera is checked all the time: if it no longer agrees with
      the main camera (it was moved/turned) for several seconds, it is solved
      again from fresh frames and replaced;
    * a solution is accepted only if it is clearly good: small residual (< 6 cm)
      and stable (rotations from the first and second half of the frames agree
      within a few degrees). A wrong person or a bad view never gets saved.
    """

    MIN_FRAMES = 50          # ~2 s at 25-30 fps where both see you well
    WINDOW = 150             # newest frames kept per camera
    MAX_RMS_M = 0.06
    MAX_SPLIT_DEG = 6.0      # first half vs second half
    DRIFT_S = 4.0            # disagreement this long -> re-solve
    MIN_INTERVAL_S = 3.0     # don't solve the same camera more often
    MIN_MOTION_M = 0.03      # mean joint movement (std) over the frames
    RECHECK_S = 10.0         # calibrated cameras: compare with a fresh fit this often
    MOVED_DEG = 4.0          # ... and replace it when it differs more than this

    def __init__(self):
        self.frames: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self.bad_since: Dict[str, float] = {}
        self.last_try: Dict[str, float] = {}
        self._seen: Dict[str, float] = {}
        self.solved = 0

    def reset(self) -> None:
        self.frames.clear()
        self.bad_since.clear()

    def add(self, src: str, main: PoseResult, aux: PoseResult, crowded: bool) -> None:
        if crowded:          # several people: a wrong pair would teach a wrong rotation
            return
        w = np.minimum(main.visibility, aux.visibility)
        if (w > 0.7).sum() < 12:
            return
        buf = self.frames.setdefault(src, [])
        buf.append((aux.world.copy(), main.world.copy(), w))
        if len(buf) > self.WINDOW:
            del buf[:len(buf) - self.WINDOW]

    def note_agreement(self, src: str, dist_m: Optional[float], now: float) -> None:
        """Called for a calibrated camera: how far its rotated pose is from the main one."""
        if dist_m is None:
            return
        if dist_m > AGREE_LIMIT_M * 0.8:
            self.bad_since.setdefault(src, now)
        else:
            self.bad_since.pop(src, None)

    @staticmethod
    def _solve(pairs) -> Tuple[np.ndarray, float]:
        a = np.concatenate([p[0] for p in pairs])
        m = np.concatenate([p[1] for p in pairs])
        w = np.concatenate([p[2] for p in pairs])
        keep = w > 0.7
        R = kabsch(a[keep], m[keep], w[keep])
        rms = float(np.sqrt(np.average(np.sum((a[keep] @ R.T - m[keep]) ** 2, axis=1),
                                       weights=w[keep])))
        return R, rms

    def step(self, src: str, current: Optional[np.ndarray], now: float) -> Optional[Dict]:
        """A new rig entry for `src` when one should be (re)made, else None.
        `current`: the camera's stored rotation (None = not calibrated yet)."""
        buf = self.frames.get(src) or []
        has_rig = current is not None
        drift = has_rig and src in self.bad_since and now - self.bad_since[src] > self.DRIFT_S
        first = self._seen.setdefault(src, now)
        # first re-check of a calibrated camera ~5 s after start, then every 10 s
        recheck = has_rig and now - self.last_try.get(src, first - self.RECHECK_S / 2) \
            >= self.RECHECK_S
        if has_rig and not (drift or recheck):
            return None
        if len(buf) < self.MIN_FRAMES or now - self.last_try.get(src, -1e9) < self.MIN_INTERVAL_S:
            return None
        if drift:   # use only the newest frames: after the camera was moved
            buf = buf[-max(self.MIN_FRAMES, len(buf) // 2):]
        # you must move a little: then a good fit proves both cameras see the SAME
        # person (two different people standing still could fit by chance)
        main = np.stack([p[1] for p in buf])
        motion = float(np.mean(np.std(main[:, 11:29], axis=0)))
        if motion < self.MIN_MOTION_M:
            return None
        self.last_try[src] = now
        half = len(buf) // 2
        R, rms = self._solve(buf)
        R1, _ = self._solve(buf[:half])
        R2, _ = self._solve(buf[half:])
        split = rotation_angle_deg(R1, R2)
        if rms > self.MAX_RMS_M or split > self.MAX_SPLIT_DEG:
            log.info("auto rig %s rejected: rms %.3f m, split %.1f deg", src, rms, split)
            if drift:
                self.frames[src] = []      # collect fresh frames for the next try
            return None
        if has_rig and not drift:
            # periodic re-check of a calibrated camera: replace only a clearly
            # different (moved) camera, and only if the new fit is better
            if rotation_angle_deg(R, current) < self.MOVED_DEG:
                return None
            old = np.concatenate([p[0] for p in buf]) @ current.T
            m = np.concatenate([p[1] for p in buf])
            old_rms = float(np.sqrt(np.mean(np.sum((old - m) ** 2, axis=1))))
            if rms > 0.7 * old_rms:
                return None
        self.solved += 1
        self.frames[src] = []
        self.bad_since.pop(src, None)
        return {"R": R.reshape(-1).tolist(), "rms_cm": round(rms * 100, 1),
                "frames": len(buf), "time": int(time.time()), "auto": True}


def rig_rotation(entry) -> Optional[np.ndarray]:
    try:
        R = np.array(entry["R"], dtype=np.float64).reshape(3, 3)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.all(np.isfinite(R)) or np.abs(R @ R.T - np.eye(3)).max() > 1e-3 \
            or abs(np.linalg.det(R) - 1) > 1e-3:
        return None
    return R


# --------------------------------------------------------------------- workers
@dataclass
class AuxState:
    source: str
    fps: float = 0.0
    people: int = 0
    found: bool = False
    error: Optional[str] = None


class AuxCamera(threading.Thread):
    """One extra camera: capture -> model -> person lock, newest result kept."""

    def __init__(self, spec: Dict, profile: Optional[Dict], model_path: Path, num_poses: int,
                 relock_s: float, person_lock: bool):
        super().__init__(name=f"aux-{spec.get('source')}", daemon=True)
        self.spec = spec
        self.source = str(spec.get("source", ""))
        self.name = str(spec.get("name", "") or "")
        best = best_from_profile(profile) if profile else None
        if best is not None:
            w, h, fps, fcc = best.width, best.height, int(round(best.fps)), best.fourcc
        else:
            w, h, fps, fcc = 1280, 720, 30, "MJPG"
        self.cam = CameraSource(self.source, w, h, fps, fcc if fcc in ("MJPG", "YUY2") else "auto",
                                name=self.name)
        self.model_path, self.num_poses = model_path, num_poses
        self.relock_s, self.use_lock = relock_s, person_lock
        self.lock = PersonLock()
        self._stop = threading.Event()
        self._mx = threading.Lock()
        self._latest: Optional[Tuple[float, PoseResult]] = None
        self.state = AuxState(self.source)
        self.preview: Optional[np.ndarray] = None

    def latest(self) -> Optional[Tuple[float, PoseResult]]:
        with self._mx:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=3.0)

    def run(self) -> None:
        est = None
        try:
            self.cam.start(wait=True)
            est = make_estimator(self.model_path, num_poses=self.num_poses, cls=PoseEstimator)
            seq, frames, t0 = 0, 0, time.monotonic()
            rot = int(self.spec.get("rotate", 0) or 0)
            mirror = bool(self.spec.get("mirror", False))
            while not self._stop.is_set():
                item = self.cam.get_latest(seq, 0.5)
                if item is None:
                    if self.cam.error:
                        raise RuntimeError(self.cam.error)
                    continue
                seq, frame, ts = item
                if rot in (90, 180, 270):
                    frame = cv2.rotate(frame, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                                               270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rot])
                if mirror:
                    frame = cv2.flip(frame, 1)
                rgb = prepare_rgb(frame)
                people = est.detect(rgb, int(ts * 1000))
                if self.use_lock:
                    res, info = self.lock.select(people, rgb, ts, self.relock_s)
                else:
                    res, info = (people[0] if people else None), {"people": len(people)}
                with self._mx:
                    if res is not None:
                        self._latest = (ts, res)
                self.preview = rgb
                self.state.people = int(info.get("people", 0))
                self.state.found = res is not None
                frames += 1
                now = time.monotonic()
                if now - t0 >= 1.0:
                    self.state.fps = frames / (now - t0)
                    frames, t0 = 0, now
        except Exception as e:
            log.warning("extra camera %s stopped: %s", self.source, e)
            self.state.error = str(e) or e.__class__.__name__
        finally:
            self.cam.stop()
            if est is not None:
                est.close()


class AuxCameras:
    def __init__(self, specs: List[Dict], profiles: Dict, model_path: Path, num_poses: int,
                 relock_s: float, person_lock: bool):
        self.workers: List[AuxCamera] = []
        for spec in specs[:MAX_EXTRA_CAMERAS]:
            src = str(spec.get("source", "")).strip()
            if not src or not spec.get("enabled", True):
                continue
            if not is_video_file(src) and not src.isdigit():
                continue
            self.workers.append(AuxCamera(spec, profile_for(profiles, src, spec.get("name", "")),
                                          model_path, num_poses,
                                          relock_s, person_lock))

    def start(self) -> None:
        for w in self.workers:
            w.start()

    def stop(self) -> None:
        for w in self.workers:
            w.stop()

    def fresh(self, now: float) -> Dict[str, Tuple[float, PoseResult]]:
        out = {}
        for w in self.workers:
            item = w.latest()
            if item is not None and now - item[0] <= MAX_AGE_S:
                out[w.source] = item
        return out

    def states(self) -> List[AuxState]:
        return [w.state for w in self.workers]
