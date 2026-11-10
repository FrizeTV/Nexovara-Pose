"""RTMPose (OpenMMLab) as a precision add-on to MediaPipe — on the GPU or CPU.

Why an add-on and not a replacement
-----------------------------------
RTMPose is a top-down 2D model: it finds joints in the picture very precisely
(feet included: heel, big and small toe — "Halpe-26"), but it gives no 3D. One
camera's 3D (and the person detector, tracking, several people) stays with
MediaPipe; RTMPose sharpens WHERE in the picture every joint is:

  1. MediaPipe as before -> 3D joints + picture joints + the box around you;
  2. RTMPose looks at that box (no extra detector needed) -> 26 precise joints;
  3. where RTMPose is confident and agrees with MediaPipe (no jump to another
     limb/person), the picture position moves towards RTMPose and the 3D joint
     moves by the same amount across the view (metres per pixel come from
     MediaPipe's own 3D-vs-picture scale). Depth stays MediaPipe's.
  Disagreeing or unsure joints are left alone, so it can't make things worse
  than MediaPipe alone in the worst case — and in the window you see a live
  comparison (jitter of the feet with and without it).

Runtimes (ONNX Runtime, pre-built wheels only):
  * Windows, any GPU (NVIDIA / AMD / Intel): onnxruntime-directml (DirectML);
  * NVIDIA with CUDA: onnxruntime-gpu; otherwise CPU (onnxruntime).

Models: official OpenMMLab "onnx_sdk" archives from download.openmmlab.com
(Apache-2.0), HTTPS + host allowlist, only end2end.onnx is extracted (size
checked), SHA-256 recorded on first use and checked on every start.
Pre/post-processing follows rtmlib (Apache-2.0, github.com/Tau-J/rtmlib),
with sub-bin (centroid) decoding of the SimCC output added.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import net
from .pose import ModelManifest, PoseResult

log = logging.getLogger(__name__)

_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
ALLOWED_HOSTS = {"download.openmmlab.com"}
MAX_ZIP_BYTES = 450_000_000
MAX_ONNX_BYTES = 500_000_000
MIN_ONNX_BYTES = 1_000_000


@dataclass(frozen=True)
class RefinerModel:
    id: str
    title: str
    url: str
    file: str
    input_wh: Tuple[int, int]      # model input (width, height)


REFINERS: Dict[str, RefinerModel] = {
    "rtmpose_m": RefinerModel(
        "rtmpose_m", "RTMPose-m (быстрая, GPU или мощный CPU)",
        _BASE + "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
        "rtmpose-m_halpe26_256x192.onnx", (192, 256)),
    "rtmpose_x": RefinerModel(
        "rtmpose_x", "RTMPose-x (самая точная, нужна видеокарта)",
        _BASE + "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
        "rtmpose-x_halpe26_384x288.onnx", (288, 384)),
}
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)    # RGB
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)
SIMCC_SPLIT = 2.0
N_KPT = 26
GPU_PROVIDERS = ("DmlExecutionProvider", "CUDAExecutionProvider", "ROCMExecutionProvider",
                 "CoreMLExecutionProvider")
PROVIDER_NAMES = {"DmlExecutionProvider": "GPU (DirectML)", "CUDAExecutionProvider": "GPU (CUDA)",
                  "ROCMExecutionProvider": "GPU (ROCm)", "CoreMLExecutionProvider": "GPU (CoreML)",
                  "CPUExecutionProvider": "CPU"}

# Halpe-26 -> MediaPipe-33 (same "left" = the person's left)
HALPE_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16,
               11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28, 24: 29, 25: 30}
# MediaPipe "foot index" = tip of the foot: mostly the big toe, a bit of the small toe
TOES = {31: (20, 22), 32: (21, 23)}
TOE_BIG = 0.65


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def available_providers() -> List[str]:
    try:
        import onnxruntime as ort
        return list(ort.get_available_providers())
    except Exception:
        return []


# ---------------------------------------------------------------- download
def _check_url(url: str) -> None:
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS:
        raise RuntimeError(f"неожиданный адрес загрузки модели: {url}")


def _looks_like_onnx(path: Path) -> bool:
    try:
        size = path.stat().st_size
        if not MIN_ONNX_BYTES <= size <= MAX_ONNX_BYTES:
            return False
        with open(path, "rb") as f:
            head = f.read(2)
        return head[:1] == b"\x08"          # protobuf field 1 (ir_version), varint
    except OSError:
        return False


def extract_onnx(zip_path: Path, dest: Path) -> None:
    """Take only end2end.onnx out of an OpenMMLab onnx_sdk archive (no paths
    from the archive are used, size is limited while copying)."""
    with zipfile.ZipFile(zip_path) as z:
        members = [i for i in z.infolist()
                   if not i.is_dir() and i.filename.replace("\\", "/").split("/")[-1]
                   == "end2end.onnx"]
        if len(members) != 1:
            raise RuntimeError("в архиве модели нет end2end.onnx")
        info = members[0]
        if info.file_size > MAX_ONNX_BYTES:
            raise RuntimeError("модель в архиве слишком большая")
        fd, tmp = tempfile.mkstemp(prefix=".onnx-", suffix=".onnx", dir=str(dest.parent))
        os.close(fd)
        try:
            done = 0
            with z.open(info) as src, open(tmp, "wb") as out:
                while True:
                    chunk = src.read(1 << 20)
                    if not chunk:
                        break
                    done += len(chunk)
                    if done > MAX_ONNX_BYTES:
                        raise RuntimeError("модель в архиве слишком большая")
                    out.write(chunk)
            if not _looks_like_onnx(Path(tmp)):
                raise RuntimeError("модель из архива не похожа на ONNX")
            shutil.move(tmp, dest)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


def ensure_refiner_model(model_id: str, models_dir: Path, search_dirs: Sequence[Path] = (),
                         progress: Optional[Callable[[float], None]] = None,
                         manifest: Optional[ModelManifest] = None, *, download: bool = True) -> Path:
    m = REFINERS[model_id]
    manifest = manifest or ModelManifest(None)
    for d in [*search_dirs, models_dir]:
        p = Path(d) / m.file
        if p.is_file() and _looks_like_onnx(p):
            if manifest.verify_or_record(p):
                return p
            raise RuntimeError(f"Файл модели {p} был изменён — удалите его, программа "
                               "скачает оригинал заново")
    if not download:
        raise FileNotFoundError(f"{m.title}: скачайте модель в окне «Модели»")
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / m.file
    fd, tmp_zip = tempfile.mkstemp(prefix=".dl-", suffix=".zip", dir=str(models_dir))
    os.close(fd)

    def consume(r) -> None:
        total = int(r.headers.get("Content-Length") or 0)
        if total > MAX_ZIP_BYTES:
            raise RuntimeError("архив модели неожиданно большой")
        done = 0
        with open(tmp_zip, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                done += len(chunk)
                if done > MAX_ZIP_BYTES:
                    raise RuntimeError("архив модели неожиданно большой")
                f.write(chunk)
                if progress and total:
                    progress(done / total)
    try:
        log.info("Downloading %s", m.url)
        net.fetch(m.url, consume, _check_url, headers={"User-Agent": "Nexovara Pose"}, timeout=30)
        if not zipfile.is_zipfile(tmp_zip):
            raise RuntimeError("скачанный архив модели повреждён")
        extract_onnx(Path(tmp_zip), dest)
    finally:
        try:
            os.unlink(tmp_zip)
        except OSError:
            pass
    manifest.forget(dest)
    manifest.verify_or_record(dest)
    return dest


# ---------------------------------------------------------------- pre/post-processing
def bbox_from_landmarks(image: np.ndarray, vis: np.ndarray, w: int, h: int
                        ) -> Optional[np.ndarray]:
    """Box (x1, y1, x2, y2) in pixels around the MediaPipe person (crown included)."""
    pts = image[:, :2] * np.array([w, h])
    ok = vis > 0.3
    if ok.sum() < 6:
        ok = np.ones(len(vis), dtype=bool)
    p = pts[ok]
    x1, y1 = p.min(axis=0)
    x2, y2 = p.max(axis=0)
    bh = y2 - y1
    y1 -= 0.12 * bh                        # the crown is above the eyes/ears
    if x2 - x1 < 4 or y2 - y1 < 4 or not np.all(np.isfinite([x1, y1, x2, y2])):
        return None
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _warp(center: np.ndarray, scale: np.ndarray, out_wh: Tuple[int, int]) -> np.ndarray:
    """Affine matrix: the box (center, scale=(w, h)) -> model input."""
    ow, oh = out_wh
    sx, sy = ow / scale[0], oh / scale[1]
    return np.array([[sx, 0.0, ow / 2 - sx * center[0]],
                     [0.0, sy, oh / 2 - sy * center[1]]], dtype=np.float64)


def preprocess(rgb: np.ndarray, bbox: np.ndarray, input_wh: Tuple[int, int],
               padding: float = 1.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (NCHW float32 input, center, scale) as in mmpose top-down affine."""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    scale = np.array([(x2 - x1), (y2 - y1)]) * padding
    w, h = input_wh
    aspect = w / h
    if scale[0] > scale[1] * aspect:
        scale = np.array([scale[0], scale[0] / aspect])
    else:
        scale = np.array([scale[1] * aspect, scale[1]])
    M = _warp(center, scale, input_wh)
    img = cv2.warpAffine(rgb, M, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), center, scale


def decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """(K, Wx), (K, Wy) -> locations (K, 2) in SimCC bins with sub-bin precision,
    scores (K,) = mean of both peak values (as mmpose).

    Sub-bin: a parabola through the peak and its two neighbours — on the
    logarithm when all three are positive (exact for the Gaussian-shaped SimCC
    response), else on the raw values. The shift is limited to ±0.5 bin, so it
    can only refine the argmax, never move to another peak."""
    def axis(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        K, L = v.shape
        rows = np.arange(K)
        i = np.argmax(v, axis=1)
        peak = v[rows, i]
        il, ir = np.clip(i - 1, 0, L - 1), np.clip(i + 1, 0, L - 1)
        a, b, c = v[rows, il], peak, v[rows, ir]
        pos_ok = (a > 1e-9) & (b > 1e-9) & (c > 1e-9)
        la, lb, lc = (np.log(np.where(pos_ok, x, 1.0)) for x in (a, b, c))
        a2, b2, c2 = np.where(pos_ok, la, a), np.where(pos_ok, lb, b), np.where(pos_ok, lc, c)
        den = a2 - 2.0 * b2 + c2
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(den < -1e-12, 0.5 * (a2 - c2) / den, 0.0)
        off = np.where((i > 0) & (i < L - 1), np.clip(off, -0.5, 0.5), 0.0)
        return i + off, peak
    px, vx = axis(np.asarray(simcc_x, dtype=np.float64))
    py, vy = axis(np.asarray(simcc_y, dtype=np.float64))
    scores = 0.5 * (vx + vy)
    return np.stack([px, py], axis=1), scores


def to_image(locs: np.ndarray, center: np.ndarray, scale: np.ndarray,
             input_wh: Tuple[int, int]) -> np.ndarray:
    k = locs / SIMCC_SPLIT
    return k / np.asarray(input_wh, dtype=np.float64) * scale + center - scale / 2


# ---------------------------------------------------------------- model
def open_session(data: bytes, use_gpu: bool, threads: int = 0):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    providers = [p for p in GPU_PROVIDERS if use_gpu and p in avail] + ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if "DmlExecutionProvider" in providers:
        so.enable_mem_pattern = False                       # required by DirectML
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads:
        so.intra_op_num_threads = int(threads)
    try:
        return ort.InferenceSession(data, sess_options=so, providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        log.warning("GPU ONNX session failed, falling back to CPU", exc_info=True)
        return ort.InferenceSession(data, sess_options=so, providers=["CPUExecutionProvider"])


class RtmPose:
    def __init__(self, model_path: Path, input_wh: Tuple[int, int], use_gpu: bool = True,
                 threads: int = 0):
        # Bytes also work with non-ASCII Windows paths.
        self.sess = open_session(Path(model_path).read_bytes(), use_gpu, threads)
        used = self.sess.get_providers()
        self.device = PROVIDER_NAMES.get(used[0], used[0]) if used else "CPU"
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_wh = tuple(int(v) for v in input_wh)
        shape = list(inp.shape)
        if len(shape) == 4 and all(isinstance(v, int) for v in shape[2:]):
            if (shape[3], shape[2]) != self.input_wh:
                raise RuntimeError(f"неожиданный вход модели {shape}")
        outs = self.sess.get_outputs()
        if len(outs) != 2:
            raise RuntimeError("неожиданные выходы модели")
        self.output_names = [o.name for o in outs]

    def infer(self, rgb: np.ndarray, bbox: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """-> keypoints (26, 2) in pixels of `rgb`, scores (26,)."""
        x, center, scale = preprocess(rgb, bbox, self.input_wh)
        sx, sy = self.sess.run(self.output_names, {self.input_name: x})
        sx, sy = np.asarray(sx)[0], np.asarray(sy)[0]
        if sx.ndim != 2 or sy.ndim != 2 or sx.shape[0] != N_KPT or sy.shape[0] != N_KPT:
            raise RuntimeError("неожиданный выход модели")
        locs, scores = decode_simcc(sx, sy)
        return to_image(locs, center, scale, self.input_wh), scores

    def close(self) -> None:
        self.sess = None


# ---------------------------------------------------------------- fusion with MediaPipe
GATE_TORSO = 0.35          # max disagreement, in torso lengths (else: other limb/person)
SCORE_LO, SCORE_HI = 0.30, 0.70


def mapped_points(kpts_px: np.ndarray, scores: np.ndarray, w: int, h: int
                  ) -> Dict[int, Tuple[np.ndarray, float]]:
    """RTMPose points as MediaPipe indices, normalized image coords."""
    wh = np.array([w, h], dtype=np.float64)
    out = {m: (kpts_px[k] / wh, float(scores[k])) for k, m in HALPE_TO_MP.items()}
    for m, (big, small) in TOES.items():
        p = (TOE_BIG * kpts_px[big] + (1 - TOE_BIG) * kpts_px[small]) / wh
        out[m] = (p, float(min(scores[big], scores[small])))
    return out


def world_scale(res: PoseResult, aspect: float) -> Optional[float]:
    """Metres per normalized image height, from MediaPipe's own 3D vs picture."""
    ids = [i for i in range(11, 33) if res.visibility[i] > 0.5]
    if len(ids) < 6:
        return None
    img = res.image[ids, :2] * np.array([aspect, 1.0])
    wor = res.world[ids, :2]
    img = img - img.mean(axis=0)
    wor = wor - wor.mean(axis=0)
    den = float(np.sum(img * img))
    if den < 1e-6:
        return None
    k = float(np.sum(img * wor)) / den
    return k if 0.05 < k < 20.0 else None


def refine(res: PoseResult, points: Dict[int, Tuple[np.ndarray, float]], aspect: float,
           strength: float = 0.7) -> Tuple[PoseResult, Dict[str, float]]:
    """MediaPipe result moved towards confident, agreeing RTMPose points."""
    k = world_scale(res, aspect)
    image = res.image.copy()
    world = res.world.copy()
    vis = res.visibility.copy()
    sh = (res.image[11, :2] + res.image[12, :2]) / 2
    hp = (res.image[23, :2] + res.image[24, :2]) / 2
    torso = float(np.hypot((sh[0] - hp[0]) * aspect, sh[1] - hp[1]))
    used, gated, shift = 0, 0, []
    if k is None or torso < 1e-3:
        return res, {"used": 0, "gated": 0, "shift": 0.0}
    for m, (p, s) in points.items():
        if not np.all(np.isfinite(p)):
            continue
        d = p - res.image[m, :2]
        dist = float(np.hypot(d[0] * aspect, d[1]))
        if dist > GATE_TORSO * torso:
            gated += 1
            continue
        wgt = strength * float(np.clip((s - SCORE_LO) / (SCORE_HI - SCORE_LO), 0.0, 1.0))
        if wgt <= 0.0:
            continue
        image[m, :2] += wgt * d
        world[m, 0] += wgt * d[0] * aspect * k     # MediaPipe world x = picture right
        world[m, 1] += wgt * d[1] * k              # world y = picture down
        if s > SCORE_HI:
            vis[m] = max(vis[m], min(0.9, s))
        used += 1
        shift.append(dist)
    return (PoseResult(world, image, vis),
            {"used": used, "gated": gated, "shift": float(np.mean(shift)) if shift else 0.0})


class JitterMeter:
    """Live A/B: frame-to-frame jitter of the feet in the picture, MediaPipe alone
    vs with RTMPose, measured only while you stand still (per 1000 px of height)."""
    FEET = [27, 28, 29, 30, 31, 32]
    ALPHA = 0.03

    def __init__(self):
        self.prev: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.mp = None
        self.ref = None
        self.samples = 0

    def update(self, before: np.ndarray, after: np.ndarray) -> None:
        a, b = before[self.FEET, :2], after[self.FEET, :2]
        if self.prev is not None:
            sa = np.linalg.norm(a - self.prev[0], axis=1) * 1000
            sb = np.linalg.norm(b - self.prev[1], axis=1) * 1000
            if float(np.median(sa)) < 4.0:            # standing still (< 4 px / 1000)
                ja, jb = float(np.sqrt(np.mean(sa ** 2))), float(np.sqrt(np.mean(sb ** 2)))
                self.mp = ja if self.mp is None else self.mp + (ja - self.mp) * self.ALPHA
                self.ref = jb if self.ref is None else self.ref + (jb - self.ref) * self.ALPHA
                self.samples += 1
        self.prev = (a.copy(), b.copy())


class Refiner:
    """Everything the engine needs: model + fusion + timing + A/B numbers."""

    def __init__(self, model_id: str, model_path: Path, use_gpu: bool = True):
        m = REFINERS[model_id]
        self.model_id = model_id
        self.net = RtmPose(model_path, m.input_wh, use_gpu)
        self.device = self.net.device
        self.ms = 0.0
        self.meter = JitterMeter()
        self.last: Dict[str, float] = {}
        self.errors = 0

    def __call__(self, res: Optional[PoseResult], rgb: np.ndarray, strength: float
                 ) -> Optional[PoseResult]:
        if res is None or res.image is None:
            return res
        h, w = rgb.shape[:2]
        bbox = bbox_from_landmarks(res.image, res.visibility, w, h)
        if bbox is None:
            return res
        t0 = time.perf_counter()
        try:
            kp, sc = self.net.infer(rgb, bbox)
        except Exception as e:
            self.errors += 1
            if self.errors in (1, 10, 100):
                log.warning("RTMPose failed (%d): %s", self.errors, e)
            return res
        self.ms += ((time.perf_counter() - t0) * 1000 - self.ms) * 0.1
        out, info = refine(res, mapped_points(kp, sc, w, h), w / h, strength)
        self.last = info
        self.meter.update(res.image, out.image)
        return out

    def describe(self) -> str:
        s = f"RTMPose {self.ms:.0f} мс" + (" GPU" if self.device.startswith("GPU") else "")
        if self.meter.mp is not None and self.meter.samples > 30:
            s += f" · стопы {self.meter.mp:.1f}→{self.meter.ref:.1f}"
        return s

    def close(self) -> None:
        self.net.close()
