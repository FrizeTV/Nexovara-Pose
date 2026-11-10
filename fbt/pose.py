"""MediaPipe Pose Landmarker (Tasks API) wrapper + safe model download.

The legacy `mediapipe.solutions` API was removed from recent MediaPipe
releases, so we use the Tasks API (`PoseLandmarker`) in VIDEO mode, which
reuses the previous frame's result for fast, temporally stable tracking.

Model integrity: after a model is downloaded (or first seen) its SHA-256 is
recorded in <data>/models.json; on every later load the file is re-hashed and
refused if it was changed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np

from . import net

log = logging.getLogger(__name__)

_BASE = "https://storage.googleapis.com/mediapipe-models/"
# Official Google-hosted models. Only these exact HTTPS URLs are ever fetched.
MODEL_URLS = {
    "lite": _BASE + "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": _BASE + "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": _BASE + "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
    "hand": _BASE + "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
}
MODEL_FILES = {
    "lite": "pose_landmarker_lite.task",
    "full": "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
    "hand": "hand_landmarker.task",
}
ALLOWED_HOSTS = {"storage.googleapis.com"}
MIN_MODEL_BYTES = 1_000_000
MAX_MODEL_BYTES = 80_000_000
_manifest_lock = threading.Lock()


def model_filename(variant: str) -> str:
    return MODEL_FILES[variant]


def _looks_valid(path: Path) -> bool:
    try:
        size = path.stat().st_size
    except OSError:
        return False
    if not (MIN_MODEL_BYTES <= size <= MAX_MODEL_BYTES):
        return False
    # .task files are zip bundles of .tflite models; also accept a bare
    # TFLite flatbuffer ("TFL3" identifier at offset 4) just in case.
    if zipfile.is_zipfile(path):
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(8)
        return head[4:8] == b"TFL3"
    except OSError:
        return False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelManifest:
    """Trust-on-first-use record of model hashes (data dir, JSON)."""

    def __init__(self, path: Optional[Path]):
        self.path = path

    def _load(self) -> Dict[str, str]:
        if self.path is None:
            return {}
        try:
            if self.path.is_file() and self.path.stat().st_size < 100_000:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    return {str(k): str(v) for k, v in d.items()}
        except (OSError, ValueError):
            pass
        return {}

    def _save(self, d: Dict[str, str]) -> None:
        if self.path is None:
            return
        try:
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=1)
            os.replace(tmp, self.path)
        except OSError as e:
            log.warning("Cannot write model manifest: %s", e)

    def verify_or_record(self, path: Path) -> bool:
        key = str(path.resolve())
        digest = sha256_file(path)
        with _manifest_lock:
            d = self._load()
            known = d.get(key)
            if known is None:
                d[key] = digest
                self._save(d)
                log.info("Model registered: %s sha256=%s", path.name, digest)
                return True
            if known != digest:
                log.error("Model %s changed on disk (sha256 mismatch)", path)
                return False
            return True

    def forget(self, path: Path) -> None:
        with _manifest_lock:
            d = self._load()
            d.pop(str(path.resolve()), None)
            self._save(d)


def find_model(variant: str, search_dirs: Sequence[Path]) -> Optional[Path]:
    for d in search_dirs:
        p = Path(d) / model_filename(variant)
        if p.is_file() and _looks_valid(p):
            return p
    return None


def _check_url(url: str) -> None:
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS:
        raise RuntimeError(f"неожиданный адрес загрузки модели: {url}")


def _download(url: str, dest: Path, progress) -> None:
    """Download over HTTPS: directly first, then via the system proxy (see net.py)."""
    _check_url(url)
    fd, tmp = tempfile.mkstemp(prefix=".dl-", suffix=".task", dir=str(dest.parent))
    os.close(fd)

    def consume(r) -> None:
        total = int(r.headers.get("Content-Length") or 0)
        if total > MAX_MODEL_BYTES:
            raise RuntimeError("model file is unexpectedly large")
        done = 0
        with open(tmp, "wb") as f:      # truncated on every attempt
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                done += len(chunk)
                if done > MAX_MODEL_BYTES:
                    raise RuntimeError("model file is unexpectedly large")
                f.write(chunk)
                if progress and total:
                    progress(done / total)

    try:
        net.fetch(url, consume, _check_url, headers={"User-Agent": "Nexovara Pose"}, timeout=30)
        if not _looks_valid(Path(tmp)):
            raise RuntimeError("downloaded model failed validation")
        shutil.move(tmp, dest)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_model(variant: str, models_dir: Path, search_dirs: Sequence[Path] = (),
                 progress: Optional[Callable[[float], None]] = None,
                 manifest: Optional[ModelManifest] = None) -> Path:
    """Return path to a verified model, downloading it once if missing."""
    if variant not in MODEL_URLS:
        raise ValueError(f"unknown model variant {variant!r}")
    manifest = manifest or ModelManifest(None)
    found = find_model(variant, [*search_dirs, models_dir])
    if found is not None:
        if manifest.verify_or_record(found):
            return found
        if Path(found).parent.resolve() != Path(models_dir).resolve():
            raise RuntimeError(f"Файл модели {found} был изменён — удалите его, "
                               "программа скачает оригинал заново")
        log.warning("Re-downloading modified model %s", found)
        manifest.forget(found)
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / model_filename(variant)
    log.info("Downloading model %s", MODEL_URLS[variant])
    _download(MODEL_URLS[variant], dest, progress)
    manifest.forget(dest)
    manifest.verify_or_record(dest)
    return dest


# --------------------------------------------------------------------- pose
POSE_INPUT_MAX_W = 960  # the model itself runs at 256 px; bigger frames only cost copies


def prepare_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """Fast model input: downscale (bilinear, ~0.2 ms) + BGR->RGB on the small image.
    Landmarks are normalized, so they stay valid for the full-resolution frame."""
    h, w = frame_bgr.shape[:2]
    if w > POSE_INPUT_MAX_W:
        frame_bgr = cv2.resize(frame_bgr, (POSE_INPUT_MAX_W, max(1, round(h * POSE_INPUT_MAX_W / w))),
                               interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


@dataclass
class PoseResult:
    world: np.ndarray       # (33, 3) metres, origin at hip centre (MediaPipe axes)
    image: np.ndarray       # (33, 3) normalized image coords (x, y in 0..1)
    visibility: np.ndarray  # (33,)


def read_model_bytes(model_path: Path) -> bytes:
    """Read a .task model into memory.

    MediaPipe is given the model CONTENTS instead of the file path: its native
    code on Windows cannot open paths with non-ASCII characters (for example
    a Cyrillic user name: C:\\Users\\Пользователь\\...), which fails with
    "Unable to open file at ...". Python opens such paths without problems."""
    p = Path(model_path)
    if not p.is_file():
        raise RuntimeError(f"Файл модели не найден: {p}")
    if not _looks_valid(p):
        raise RuntimeError(f"Файл модели повреждён — удалите его, он скачается заново: {p}")
    return p.read_bytes()


def _base_options(mp, model_bytes: bytes, gpu: bool):
    BaseOptions = mp.tasks.BaseOptions
    delegate = BaseOptions.Delegate.GPU if gpu else BaseOptions.Delegate.CPU
    return BaseOptions(model_asset_buffer=model_bytes, delegate=delegate)


class PoseEstimator:
    def __init__(self, model_path: Path, min_detection=0.5, min_presence=0.5,
                 min_tracking=0.5, use_gpu=False, num_poses: int = 1):
        import mediapipe as mp  # imported lazily: heavy module
        self._mp = mp
        vision = mp.tasks.vision
        # keep the bytes alive as long as the landmarker exists
        self._model_bytes = read_model_bytes(model_path)

        def make(gpu: bool):
            opts = vision.PoseLandmarkerOptions(
                base_options=_base_options(mp, self._model_bytes, gpu),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=max(1, min(int(num_poses), 5)),
                min_pose_detection_confidence=float(min_detection),
                min_pose_presence_confidence=float(min_presence),
                min_tracking_confidence=float(min_tracking),
                output_segmentation_masks=False,
            )
            return vision.PoseLandmarker.create_from_options(opts)

        self.delegate = "CPU"
        if use_gpu:
            try:
                self._lm = make(True)
                self.delegate = "GPU"
            except Exception as e:  # GPU delegate not available on this platform
                log.warning("GPU delegate unavailable (%s), falling back to CPU", e)
                self._lm = make(False)
        else:
            self._lm = make(False)
        self._last_ts = -1

    def detect(self, rgb: np.ndarray, ts_ms: int) -> List[PoseResult]:
        """All people found in the frame (up to num_poses)."""
        ts_ms = max(int(ts_ms), self._last_ts + 1)  # must be strictly increasing
        self._last_ts = ts_ms
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        res = self._lm.detect_for_video(img, ts_ms)
        out: List[PoseResult] = []
        for w, n in zip(res.pose_world_landmarks or [], res.pose_landmarks or []):
            if len(w) < 33 or len(n) < 33:
                continue
            world = np.array([[p.x, p.y, p.z] for p in w], dtype=np.float64)
            image = np.array([[p.x, p.y, p.z] for p in n], dtype=np.float64)
            vis = np.array([(p.visibility if p.visibility is not None else 1.0) for p in n],
                           dtype=np.float64)
            if np.all(np.isfinite(world)) and np.all(np.isfinite(image)):
                out.append(PoseResult(world, image, np.nan_to_num(vis, nan=0.0)))
        return out

    def process_rgb(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseResult]:
        """First (most confident) person only."""
        found = self.detect(rgb, ts_ms)
        return found[0] if found else None

    def process(self, frame_bgr: np.ndarray, ts_ms: int) -> Optional[PoseResult]:
        return self.process_rgb(prepare_rgb(frame_bgr), ts_ms)

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass


class AdaptivePoseEstimator:
    """Several people only when there ARE several people.

    MediaPipe looking for N > 1 poses runs its person DETECTOR on every frame
    while fewer than N people are tracked — i.e. always, when you are alone in
    the room. Measured: 37 ms instead of 20 ms per frame (lite model, CPU).

    So the usual frame goes through a 1-person landmarker (tracking only), and
    a N-person landmarker looks for other people every CHECK_S seconds, after
    the body was lost, and on every frame while several people are in view
    (then the person lock can pick you among them, exactly as before)."""

    CHECK_S = 0.5     # how often to look for other people when alone
    HOLD_S = 2.0      # keep checking every frame this long after seeing others

    def __init__(self, model_path: Path, min_detection=0.5, min_presence=0.5,
                 min_tracking=0.5, use_gpu=False, num_poses: int = 3, cls=None):
        cls = cls or PoseEstimator
        self.single = cls(model_path, min_detection, min_presence, min_tracking,
                          use_gpu, num_poses=1)
        try:
            self.multi = cls(model_path, min_detection, min_presence, min_tracking,
                             use_gpu, num_poses=num_poses)
        except Exception:
            self.single.close()
            raise
        self.delegate = self.single.delegate
        self._multi_until = -1e9
        self._last_check = -1e9
        self.multi_frames = 0
        self.single_frames = 0

    @property
    def crowded(self) -> bool:
        """Several people were seen recently (multi-person mode)."""
        return self._last_check < self._multi_until

    def detect(self, rgb: np.ndarray, ts_ms: int) -> List[PoseResult]:
        now = ts_ms / 1000.0
        if now < self._multi_until:              # several people: one consistent source
            people = self.multi.detect(rgb, ts_ms)
            self.multi_frames += 1
            if len(people) > 1:
                self._multi_until = now + self.HOLD_S
            self._last_check = now
            return people
        # Your pose ALWAYS comes from the 1-person landmarker. (v4.2 took it from
        # the other landmarker on the "look around" frames — two models with
        # different internal state gave slightly different poses, so the body
        # twitched twice a second.) The N-person one only counts people here.
        people = self.single.detect(rgb, ts_ms)
        self.single_frames += 1
        if not people or now - self._last_check >= self.CHECK_S:
            others = self.multi.detect(rgb, ts_ms)
            self.multi_frames += 1
            self._last_check = now
            if len(others) > 1:
                self._multi_until = now + self.HOLD_S
                return others                    # the person lock picks you among them
            if not people:
                return others                    # lost: filled by the full search
        return people

    def process_rgb(self, rgb: np.ndarray, ts_ms: int) -> Optional[PoseResult]:
        found = self.detect(rgb, ts_ms)
        return found[0] if found else None

    def close(self) -> None:
        self.single.close()
        self.multi.close()


def make_estimator(model_path: Path, min_detection=0.5, min_presence=0.5, min_tracking=0.5,
                   use_gpu=False, num_poses: int = 1, cls=None):
    """PoseEstimator for one person, AdaptivePoseEstimator for several."""
    cls = cls or PoseEstimator
    if int(num_poses) > 1:
        return AdaptivePoseEstimator(model_path, min_detection, min_presence, min_tracking,
                                     use_gpu, num_poses=num_poses, cls=cls)
    return cls(model_path, min_detection, min_presence, min_tracking, use_gpu, num_poses=1)

