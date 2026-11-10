"""ONNX downloads and adapters. Every adapter returns pixel xy + confidence."""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from . import net, rtmpose
from .model_catalog import EXTRA_MODELS
from .pose import ModelManifest


@dataclass(frozen=True)
class Export:
    url: str
    sha256: str = ""

    @property
    def archive(self):
        return self.url.endswith(".zip")


_HF = "https://huggingface.co/"
EXPORTS = {
    "movenet_lightning": Export(
        _HF + "Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx",
        "1ad4f8d6c2f776a9967db3993c9ca740bc350104f9d37c151dc183fc29a464ad"),
    "movenet_thunder": Export(
        _HF + "Xenova/movenet-singlepose-thunder/resolve/main/onnx/model.onnx",
        "3dca9f6e5f8a64dc9935a5be06fd8bf81bf01e696c9c05c6f2a650e0a401b763"),
    "rtmo": Export("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
                   "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip",
                   "6b9d3be1323cc030444f731bbe593ea61f59b9481e23da2f8b60cb4b72136e16"),
    "vitpose": Export(
        _HF + "JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-s-wholebody.onnx",
        "ccac893a8ad6da7d43f9655047acf48e482e66922db7f4273a3b615cea989127"),
}


def _check_url(url):
    u = urlsplit(url)
    allowed = {"download.openmmlab.com", "huggingface.co", "cdn-lfs.huggingface.co",
               "cdn-lfs-us-1.huggingface.co", "cdn-lfs-eu-1.huggingface.co",
               "cas-bridge.xethub.hf.co", "us.aws.cdn.hf.co", "eu.aws.cdn.hf.co"}
    if u.scheme != "https" or u.hostname not in allowed or u.username is not None:
        raise RuntimeError("Недопустимый адрес загрузки модели")


def find_extra(model_id, dirs):
    if model_id not in EXTRA_MODELS:
        return None
    ref = rtmpose.REFINERS.get(model_id)
    name = ref.file if ref else model_id + ".onnx"
    return next((p for d in dirs if rtmpose._looks_like_onnx(p := Path(d) / name)), None)


def ensure_extra(model_id, models_dir, search_dirs=(), progress=None, manifest=None,
                 *, download=True):
    if model_id in rtmpose.REFINERS:
        return rtmpose.ensure_refiner_model(model_id, models_dir, search_dirs, progress,
                                           manifest, download=download)
    spec = EXPORTS[model_id]
    manifest = manifest or ModelManifest(None)
    p = find_extra(model_id, [*search_dirs, models_dir])
    if p is not None:
        if manifest.verify_or_record(p):
            return p
        raise RuntimeError(f"Файл {p.name} изменён. Удалите его и скачайте заново.")
    if not download:
        raise FileNotFoundError(f"{EXTRA_MODELS[model_id]}: скачайте модель в окне «Модели»")
    models_dir.mkdir(parents=True, exist_ok=True)
    dest = models_dir / (model_id + ".onnx")

    def consume(response, path):
        total = int(response.headers.get("Content-Length") or 0)
        cap = rtmpose.MAX_ZIP_BYTES if spec.archive else rtmpose.MAX_ONNX_BYTES
        done = 0
        digest = hashlib.sha256()
        with path.open("wb") as out:
            while chunk := response.read(1 << 20):
                done += len(chunk)
                if max(total, done) > cap:
                    raise RuntimeError("Слишком большой файл модели")
                out.write(chunk)
                digest.update(chunk)
                if progress and total:
                    progress(min(1.0, done / total))
        if (total and done != total) or (spec.sha256 and digest.hexdigest() != spec.sha256):
            raise RuntimeError("Загрузка оборвалась или SHA-256 файла не совпадает")

    with tempfile.TemporaryDirectory(prefix=".pose-", dir=models_dir) as temp:
        path = Path(temp) / "download"
        net.fetch(spec.url, lambda r: consume(r, path), _check_url,
                  headers={"User-Agent": "Nexovara Pose"}, timeout=30)
        if spec.archive:
            model = path.with_suffix(".onnx")
            rtmpose.extract_onnx(path, model)
        else:
            model = path
        if not rtmpose._looks_like_onnx(model):
            raise RuntimeError("Скачанный файл не является ONNX-моделью")
        os.replace(model, dest)
    manifest.forget(dest)
    manifest.verify_or_record(dest)
    return dest


def _heatmap(heat):
    if heat.ndim != 3 or heat.shape[0] != 133 or min(heat.shape[1:]) < 2:
        raise RuntimeError(f"Неожиданный выход ViTPose: {heat.shape}")
    k, h, w = heat.shape
    flat = heat.reshape(k, -1)
    idx = np.argmax(flat, axis=1)
    row = np.arange(k)
    return np.column_stack((idx % w / (w - 1), idx // w / (h - 1), flat[row, idx]))


def _match(boxes, poses, bbox):
    """Take the detection overlapping the already selected MediaPipe person."""
    size = np.maximum(boxes[:, 2:4] - boxes[:, :2], 0)
    inter = np.maximum(np.minimum(boxes[:, 2:4], bbox[2:]) -
                       np.maximum(boxes[:, :2], bbox[:2]), 0).prod(axis=1)
    union = size.prod(axis=1) + np.prod(bbox[2:] - bbox[:2]) - inter
    iou = inter / np.maximum(union, 1e-6)
    good = np.isfinite(iou) & (iou > 0.25) & (boxes[:, 4] > 0.30)
    if not np.any(good):
        return np.empty((0, 3))
    return poses[np.argmax(np.where(good, iou * boxes[:, 4], -1))]


class OnnxPose:
    def __init__(self, model_id, path, use_gpu, threads):
        defaults = {"movenet_lightning": (192, 192), "movenet_thunder": (256, 256),
                    "vitpose": (192, 256), "rtmo": (640, 640)}
        default_wh = defaults[model_id]
        self.model_id = model_id
        self.sess = rtmpose.open_session(path.read_bytes(), use_gpu, threads)
        inp, = self.sess.get_inputs()
        self.input_name = inp.name
        self.device = rtmpose.PROVIDER_NAMES.get(self.sess.get_providers()[0], "ONNX")
        shape = inp.shape
        self.nhwc = len(shape) == 4 and shape[-1] == 3
        if len(shape) != 4 or (not self.nhwc and shape[1] != 3):
            raise RuntimeError(f"Неожиданный вход {model_id}: {shape}")
        dims = (shape[2], shape[1]) if self.nhwc else (shape[3], shape[2])
        self.wh = tuple(n if isinstance(n, int) else default
                        for n, default in zip(dims, default_wh))
        self.dtype = {"tensor(int32)": np.int32, "tensor(int64)": np.int64,
                      "tensor(uint8)": np.uint8, "tensor(float)": np.float32}[inp.type]

    def _run(self, image):
        tensor = image if self.nhwc else image.transpose(2, 0, 1)
        return self.sess.run(None, {self.input_name: np.ascontiguousarray(
            tensor[None], dtype=self.dtype)})

    def infer(self, rgb, bbox):
        mid = self.model_id
        if mid == "vitpose":
            x, center, scale = rtmpose.preprocess(rgb, bbox, self.wh)
            heat, = self._run(x[0].transpose(1, 2, 0))
            pts = _heatmap(heat[0])
            pts[:, :2] = pts[:, :2] * scale + center - scale / 2
        elif mid.startswith("movenet"):
            center = (bbox[:2] + bbox[2:]) / 2
            scale = np.repeat(1.25 * max(bbox[2:] - bbox[:2]), 2)
            crop = cv2.warpAffine(rgb, rtmpose._warp(center, scale, self.wh), self.wh)
            output, = self._run(crop)
            if output.shape != (1, 1, 17, 3):
                raise RuntimeError(f"Неожиданный выход MoveNet: {output.shape}")
            pts = output[0, 0][:, [1, 0, 2]].astype(float)
            pts[:, :2] = pts[:, :2] * scale + center - scale / 2
        else:
            pts = self._detect_rtmo(rgb, bbox)
        return pts[:, :2], pts[:, 2]

    def _detect_rtmo(self, rgb, bbox):
        w, h = self.wh
        height, width = rgb.shape[:2]
        ratio = min(w / width, h / height)
        rw, rh = max(1, round(width * ratio)), max(1, round(height * ratio))
        canvas = np.full((h, w, 3), 114, dtype=np.uint8)
        canvas[:rh, :rw] = cv2.resize(rgb, (rw, rh))
        dets, keypoints = self._run(canvas[..., ::-1])
        boxes, poses = dets[0].copy(), keypoints[0].copy()
        if boxes.ndim != 2 or boxes.shape[1] != 5 or poses.shape != (len(boxes), 17, 3):
            raise RuntimeError(f"Неожиданный выход RTMO: {boxes.shape}, {poses.shape}")
        scale = np.array([rw / width, rh / height])
        boxes[:, :4] /= np.tile(scale, 2)
        poses[..., :2] /= scale
        return _match(boxes, poses, bbox)

    def close(self):
        self.sess = None
