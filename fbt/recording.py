"""Session recording and replay — only the NUMBERS of your pose, never video.

What is recorded (per camera frame, only the person Nexovara Pose follows):
  time, 33 joints in 3D (metres), 33 joints in the picture (0..1), visibility.
No images, no other people, no names of devices. Files are written to
<data>/recordings/*.npz (plain NumPy arrays, loaded with allow_pickle=False and
strictly checked, like the config). "Удалить все мои данные" removes them too.

Why:
  * replay: choose a recording instead of a camera — the whole pipeline
    (filters, calibration, outputs) runs on it exactly as on live tracking,
    without a camera or the neural network;
  * filter tuning on YOUR movements: tune_filter() tries smoothing settings on
    the recording and suggests one only if it is measurably better than the
    current one (less jitter while standing AND no larger tracking error while
    moving).
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .pose import PoseResult

log = logging.getLogger(__name__)

REC_VERSION = 1
REC_SUFFIX = ".npz"
MAX_SECONDS = 15 * 60          # longer sessions are cut (≈ 15 MB at 60 fps)
MAX_FRAMES = MAX_SECONDS * 120
MAX_FILE_BYTES = 200_000_000
_KEYS = ("t", "world", "image", "vis", "meta")


def recordings_dir(data_dir: Path) -> Path:
    d = Path(data_dir) / "recordings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def is_recording(source: str) -> bool:
    return str(source).strip().lower().endswith(REC_SUFFIX)


def list_recordings(data_dir: Path) -> List[Path]:
    d = Path(data_dir) / "recordings"
    if not d.is_dir():
        return []
    return sorted((p for p in d.glob("*" + REC_SUFFIX) if p.is_file()),
                  key=lambda p: p.stat().st_mtime, reverse=True)


@dataclass
class Recording:
    t: np.ndarray            # (N,) seconds from the start
    world: np.ndarray        # (N, 33, 3)
    image: np.ndarray        # (N, 33, 3)
    vis: np.ndarray          # (N, 33)
    meta: Dict = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return int(len(self.t))

    @property
    def seconds(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def fps(self) -> float:
        return (self.frames - 1) / self.seconds if self.seconds > 0 else 30.0

    def result(self, i: int) -> PoseResult:
        return PoseResult(self.world[i].astype(np.float64), self.image[i].astype(np.float64),
                          self.vis[i].astype(np.float64))


class Recorder:
    """Collects frames in memory (float32) and writes one .npz at the end."""

    def __init__(self, path: Path, meta: Optional[Dict] = None):
        self.path = Path(path)
        self.meta = dict(meta or {})
        self._t: List[float] = []
        self._w: List[np.ndarray] = []
        self._i: List[np.ndarray] = []
        self._v: List[np.ndarray] = []
        self._t0: Optional[float] = None
        self.full = False

    @property
    def frames(self) -> int:
        return len(self._t)

    @property
    def seconds(self) -> float:
        return (self._t[-1] - self._t[0]) if len(self._t) > 1 else 0.0

    def add(self, ts: float, res: PoseResult) -> None:
        if self.full or res is None or res.image is None:
            return
        if self._t0 is None:
            self._t0 = ts
        t = ts - self._t0
        if t > MAX_SECONDS or len(self._t) >= MAX_FRAMES:
            self.full = True
            return
        if self._t and t <= self._t[-1]:
            return
        w, im, v = res.world, res.image, res.visibility
        if w.shape != (33, 3) or im.shape != (33, 3) or v.shape != (33,):
            return
        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(im)) and np.all(np.isfinite(v))):
            return
        self._t.append(float(t))
        self._w.append(w.astype(np.float32))
        self._i.append(im.astype(np.float32))
        self._v.append(v.astype(np.float32))

    def save(self) -> Optional[Path]:
        if len(self._t) < 2:
            return None
        meta = dict(self.meta, version=REC_VERSION, frames=len(self._t),
                    seconds=round(self.seconds, 2))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".rec-", suffix=REC_SUFFIX, dir=str(self.path.parent))
        os.close(fd)
        try:
            np.savez_compressed(tmp, t=np.asarray(self._t, dtype=np.float64),
                                world=np.stack(self._w), image=np.stack(self._i),
                                vis=np.stack(self._v),
                                meta=np.array(json.dumps(meta, ensure_ascii=False)))
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        return self.path


def new_recording_path(data_dir: Path) -> Path:
    return recordings_dir(data_dir) / time.strftime(f"запись-%Y%m%d-%H%M%S{REC_SUFFIX}")


def load_recording(path: Path) -> Recording:
    """Strict loader: plain arrays only, shapes/values checked."""
    p = Path(path)
    if not p.is_file() or not is_recording(str(p)):
        raise ValueError("файл записи не найден")
    if p.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("файл записи слишком большой")
    with np.load(p, allow_pickle=False) as z:
        if set(z.files) != set(_KEYS):
            raise ValueError("это не запись Nexovara Pose")
        t = np.asarray(z["t"], dtype=np.float64)
        world = np.asarray(z["world"], dtype=np.float32)
        image = np.asarray(z["image"], dtype=np.float32)
        vis = np.asarray(z["vis"], dtype=np.float32)
        meta_raw = z["meta"]
        if meta_raw.dtype.kind != "U" or meta_raw.size != 1 or len(str(meta_raw)) > 10_000:
            raise ValueError("повреждённые сведения о записи")
        try:
            meta = json.loads(str(meta_raw))
        except ValueError:
            raise ValueError("повреждённые сведения о записи") from None
    n = len(t)
    if not (2 <= n <= MAX_FRAMES):
        raise ValueError("в записи нет кадров")
    if world.shape != (n, 33, 3) or image.shape != (n, 33, 3) or vis.shape != (n, 33):
        raise ValueError("повреждённая запись (размеры)")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(world)) and np.all(np.isfinite(image))
            and np.all(np.isfinite(vis))):
        raise ValueError("повреждённая запись (числа)")
    if np.any(np.diff(t) <= 0) or t[-1] - t[0] > MAX_SECONDS + 1:
        raise ValueError("повреждённая запись (время)")
    if np.abs(world).max() > 10 or np.abs(image).max() > 10:
        raise ValueError("повреждённая запись (координаты)")
    return Recording(t - t[0], world, image, np.clip(vis, 0.0, 1.0),
                     meta if isinstance(meta, dict) else {})


# ---------------------------------------------------------------- replay
class ReplaySource:
    """Plays a recording like a camera (CameraSource interface), in a loop and in
    real time. The 'frame' is a plain dark picture: the skeleton is drawn on it."""

    def __init__(self, rec: Recording, size: Tuple[int, int] = (640, 360)):
        self.rec = rec
        aspect = float(rec.meta.get("aspect") or 16 / 9)
        aspect = aspect if 0.3 < aspect < 4 else 16 / 9
        w = int(size[0])
        h = max(90, int(round(w / aspect)))
        self._frame = np.full((h, w, 3), 24, dtype=np.uint8)
        self.actual_size = (w, h)
        self.actual_fourcc = "запись"
        self.measured_fps = 0.0
        self.error: Optional[str] = None
        self._cond = threading.Condition()
        self._seq = 0
        self._idx = -1
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._results: Dict[int, int] = {}

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, name="replay", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        t = self.rec.t
        frames, t_fps = 0, time.monotonic()
        try:
            while not self._stop.is_set():
                start = time.monotonic()
                for i in range(len(t)):
                    d = start + float(t[i]) - time.monotonic()
                    if d > 0 and self._stop.wait(d):
                        return
                    with self._cond:
                        self._seq += 1
                        self._idx = i
                        self._results[self._seq] = i
                        if len(self._results) > 8:
                            self._results.pop(min(self._results))
                        self._ts = time.monotonic()
                        self._cond.notify_all()
                    frames += 1
                    now = time.monotonic()
                    if now - t_fps >= 1.0:
                        self.measured_fps = frames / (now - t_fps)
                        frames, t_fps = 0, now
                self._stop.wait(1.0 / max(self.rec.fps, 1.0))     # loop: one frame pause
        finally:
            with self._cond:
                self._cond.notify_all()

    def get_latest(self, last_seq: int, timeout: float = 0.5):
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self._stop.is_set(), timeout)
            if self._seq == last_seq or self._idx < 0:
                return None
            return self._seq, self._frame, self._ts

    def index_of(self, seq: int) -> Optional[int]:
        with self._cond:
            return self._results.get(seq, self._idx if self._idx >= 0 else None)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


class ReplayEstimator:
    """Stands in for the neural network: returns the recorded pose of the frame."""

    delegate = "запись"

    def __init__(self, source: ReplaySource):
        self.source = source
        self.seq = 0

    def detect(self, _rgb, _ts_ms) -> List[PoseResult]:
        i = self.source.index_of(self.seq)
        return [] if i is None else [self.source.rec.result(i)]

    def close(self) -> None:
        pass


# ---------------------------------------------------------------- filter tuning
TUNE_JOINTS = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]       # hips, knees, feet
MC_GRID = (0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.6)
BETA_GRID = (0.5, 1.0, 1.5, 2.5, 4.0)
STILL_SPEED = 0.06          # m/s
REF_WINDOW_S = 0.17         # centred moving average = the "true" path for the error


@dataclass
class TuneResult:
    ok: bool
    message: str
    min_cutoff: float = 0.0
    beta: float = 0.0
    jitter_old: float = 0.0     # mm per frame while standing
    jitter_new: float = 0.0
    error_old: float = 0.0      # mm while moving
    error_new: float = 0.0
    better: bool = False


def _centred_mean(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, w | 1)
    if w == 1:
        return x.copy()
    k = np.ones(w) / w
    pad = np.pad(x, ((w // 2, w // 2), (0, 0), (0, 0)), mode="edge")
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        for a in range(3):
            out[:, j, a] = np.convolve(pad[:, j, a], k, "valid")
    return out


def _evaluate(rec_world: np.ndarray, t: np.ndarray, vis: np.ndarray, mc: float, beta: float,
              lag_comp: bool, still: np.ndarray, moving: np.ndarray, ref: np.ndarray
              ) -> Tuple[float, float]:
    from .filters import PoseFilter
    f = PoseFilter(mc, beta, lag_comp)
    out = np.empty_like(rec_world)
    for i in range(len(t)):
        out[i] = f(rec_world[i], float(t[i]), vis[i])
    J = TUNE_JOINTS
    steps = np.linalg.norm(np.diff(out[:, J], axis=0), axis=2)          # (N-1, J)
    s_mask = still[1:] & still[:-1]
    jitter = float(np.sqrt(np.mean(steps[s_mask] ** 2))) * 1000 if s_mask.any() else 0.0
    err = np.linalg.norm(out[moving][:, J] - ref[moving][:, J], axis=2)
    error = float(np.sqrt(np.mean(err ** 2))) * 1000 if moving.any() else 0.0
    return jitter, error


def tune_filter(rec: Recording, cur_mc: float, cur_beta: float, lag_comp: bool = True,
                progress: Optional[Callable[[float], None]] = None,
                stop: Optional[Callable[[], bool]] = None, max_seconds: float = 180.0
                ) -> TuneResult:
    """Grid search over One Euro settings on the recording. Suggests a setting
    only if it beats the current one on the weighted sum AND makes neither
    jitter nor tracking error more than 10 % worse."""
    from .skeleton import mp_to_unity
    t = rec.t
    keep = t >= t[-1] - max_seconds
    t = t[keep]
    W = mp_to_unity(rec.world[keep].astype(np.float64))
    V = rec.vis[keep].astype(np.float64)
    if len(t) < 90:
        return TuneResult(False, "Запись слишком короткая — нужно хотя бы 30 секунд.")
    fps = (len(t) - 1) / max(t[-1] - t[0], 1e-3)
    ref = _centred_mean(W, int(round(REF_WINDOW_S * fps)))
    # still / moving: displacement over half a second of a strongly smoothed path
    # (frame-to-frame differences would mostly measure the model's noise)
    slow = _centred_mean(W[:, TUNE_JOINTS], int(round(0.5 * fps)))
    k = max(1, int(round(0.25 * fps)))
    sp = np.zeros(len(t))
    disp = np.linalg.norm(slow[2 * k:] - slow[:-2 * k], axis=2)          # (N-2k, J)
    sp[k:-k] = disp.max(axis=1) / np.maximum(t[2 * k:] - t[:-2 * k], 1e-3)
    sp[:k], sp[-k:] = sp[k], sp[-k - 1]
    still = sp < STILL_SPEED
    moving = sp > 3 * STILL_SPEED
    # only joints the camera saw well
    good = V[:, TUNE_JOINTS].mean(axis=1) > 0.5
    still &= good
    moving &= good
    if still.sum() < 60 or moving.sum() < 60:
        return TuneResult(False, "В записи мало спокойных или мало подвижных моментов: "
                                 "запишите минуту — постойте спокойно секунд 20, потом "
                                 "походите, присядьте, подвигайте ногами.")
    jit0, err0 = _evaluate(W, t, V, cur_mc, cur_beta, lag_comp, still, moving, ref)
    best = (2.0, cur_mc, cur_beta, jit0, err0)
    n = len(MC_GRID) * len(BETA_GRID)
    k = 0
    for mc in MC_GRID:
        for beta in BETA_GRID:
            if stop is not None and stop():
                return TuneResult(False, "Подбор остановлен.")
            k += 1
            if progress is not None:
                progress(k / n)
            if math.isclose(mc, cur_mc) and math.isclose(beta, cur_beta):
                continue
            jit, err = _evaluate(W, t, V, mc, beta, lag_comp, still, moving, ref)
            if jit > jit0 * 1.10 or err > err0 * 1.10:
                continue
            score = jit / max(jit0, 1e-6) + err / max(err0, 1e-6)
            if score < best[0]:
                best = (score, mc, beta, jit, err)
    score, mc, beta, jit, err = best
    better = score < 1.9          # at least ~5 % better overall
    if not better:
        return TuneResult(True, "Текущие настройки сглаживания уже лучшие для ваших движений.",
                          cur_mc, cur_beta, jit0, jit0, err0, err0, False)
    return TuneResult(True, f"Найдены настройки лучше: дрожание {jit0:.1f} → {jit:.1f} мм, "
                            f"ошибка в движении {err0:.1f} → {err:.1f} мм.",
                      mc, beta, jit0, jit, err0, err, True)
