"""Automatic tuning for the user's PC, camera and target app.

* system_info()  — CPU, threads, RAM, OS (shown to the user; only the numbers
                   needed for tuning are saved, never names or serials).
* benchmark()    — runs each pose model on LIVE camera frames (the user stands
                   in view) and measures real per-frame time on this PC.
* recommend()    — turns camera probe + benchmark + target into a list of
                   recommended settings with human-readable reasons.
"""
from __future__ import annotations

import logging
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .camera import CameraSource, best_from_profile
from .config import Config
from .presets import PRESETS

log = logging.getLogger(__name__)

HEADROOM = 0.75         # leave ~25% of the frame time for the game / VR runtime
MIN_GOOD_FPS = 25.0


def _cpu_name() -> str:
    if sys.platform.startswith("win"):
        try:
            import winreg  # read-only registry access
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine() or "?"


def system_info() -> Dict[str, Any]:
    threads = os.cpu_count() or 1
    cores, ram_gb = None, None
    try:
        import psutil
        cores = psutil.cpu_count(logical=False)
        ram_gb = round(psutil.virtual_memory().total / 2 ** 30, 1)
    except Exception:
        pass
    return {
        "cpu": _cpu_name(),
        "threads": threads,
        "cores": cores or max(1, threads // 2),
        "ram_gb": ram_gb,
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


# ------------------------------------------------------------- benchmark
@dataclass
class BenchResult:
    variant: str
    delegate: str
    ms: float            # median processing time per frame
    detect_rate: float   # share of frames with a body found


def benchmark(cfg: Config, model_paths: Dict[str, Path], hand_path: Optional[Path] = None,
              seconds_each: float = 2.5,
              progress: Optional[Callable[[float, str], None]] = None,
              stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Measure each available model on live frames from the configured camera."""
    from .pose import make_estimator, prepare_rgb
    cam = CameraSource(cfg.camera, cfg.width, cfg.height, cfg.fps, cfg.fourcc,
                       name=cfg.camera_name)
    cam.start()
    results: List[BenchResult] = []
    hand_ms = None
    try:
        variants = [v for v in ("lite", "full", "heavy") if v in model_paths]
        plans = [(v, False) for v in variants]
        if not sys.platform.startswith("win"):   # MediaPipe GPU delegate: Linux/macOS only
            plans += [(v, True) for v in variants if v == "full"]
        total = len(plans) + (1 if hand_path else 0)
        seq = 0
        for i, (variant, gpu) in enumerate(plans):
            if stop and stop():
                break
            label = f"Модель {variant}{' (GPU)' if gpu else ''}"
            if progress:
                progress(i / total, f"{label}: замер…")
            try:
                est = make_estimator(model_paths[variant], use_gpu=gpu,
                                     num_poses=cfg.max_people if cfg.person_lock else 1)
            except Exception as e:
                log.info("benchmark %s gpu=%s unavailable: %s", variant, gpu, e)
                continue
            if gpu and est.delegate != "GPU":
                est.close()
                continue
            times, found, n = [], 0, 0
            t_end = time.monotonic() + seconds_each
            warm = 5
            while time.monotonic() < t_end and not (stop and stop()):
                item = cam.get_latest(seq, 0.5)
                if item is None:
                    if cam.error:
                        raise RuntimeError(cam.error)
                    continue
                seq, frame, ts = item
                t0 = time.perf_counter()
                rgb = prepare_rgb(frame)
                res = est.process_rgb(rgb, int(ts * 1000))
                dt = (time.perf_counter() - t0) * 1000
                if warm > 0:
                    warm -= 1
                    continue
                n += 1
                times.append(dt)
                found += res is not None
            est.close()
            if times:
                times.sort()
                results.append(BenchResult(variant, est.delegate, round(times[len(times) // 2], 1),
                                           round(found / max(n, 1), 2)))
        if hand_path and not (stop and stop()):
            if progress:
                progress(len(plans) / total, "Пальцы: замер…")
            hand_ms = _bench_hands(cam, model_paths.get("full") or next(iter(model_paths.values())),
                                   hand_path, seconds_each, stop)
    finally:
        cam.stop()
    if progress:
        progress(1.0, "Готово")
    return {
        "results": [r.__dict__ for r in results],
        "hand_ms": hand_ms,
        "camera_fps": round(cam.measured_fps, 1),
        "time": int(time.time()),
    }


def _bench_hands(cam, pose_path, hand_path, seconds, stop) -> Optional[float]:
    from .hands import HandTracker
    from .pose import PoseEstimator, prepare_rgb
    try:
        est = PoseEstimator(pose_path, num_poses=1)
        ht = HandTracker(hand_path)
    except Exception as e:
        log.info("hand benchmark unavailable: %s", e)
        return None
    times = []
    seq = 0
    t_end = time.monotonic() + seconds
    try:
        while time.monotonic() < t_end and not (stop and stop()):
            item = cam.get_latest(seq, 0.5)
            if item is None:
                continue
            seq, frame, ts = item
            res = est.process_rgb(prepare_rgb(frame), int(ts * 1000))
            if res is None:
                continue
            t0 = time.perf_counter()
            ht.process(frame, res.image, res.visibility, int(ts * 1000))
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        est.close()
        ht.close()
    if len(times) < 3:
        return None
    times.sort()
    return round(times[len(times) // 2], 1)


# ------------------------------------------------------------- recommend
@dataclass
class Recommendation:
    key: str
    value: Any
    label: str
    reason: str


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "вкл" if v else "выкл"
    if isinstance(v, dict):
        on = [k for k, x in v.items() if x]
        return ", ".join(on) if on else "—"
    return str(v)


def recommend(cfg: Config, sysinfo: Dict[str, Any], cam_profile: Optional[Dict[str, Any]],
              bench: Optional[Dict[str, Any]], target: str, fingers: bool) -> List[Recommendation]:
    recs: List[Recommendation] = []
    # ---- camera
    cam_fps = float(cfg.fps)
    if cam_profile:
        best = best_from_profile(cam_profile, prefer_hd=fingers)
        if best is not None:
            cam_fps = best.fps
            why = f"лучший режим камеры по замеру: {best.label()}"
            recs += [Recommendation("width", best.width, "Ширина кадра", why),
                     Recommendation("height", best.height, "Высота кадра", why),
                     Recommendation("fps", int(round(min(best.fps, 120))), "FPS камеры", why),
                     Recommendation("fourcc", best.fourcc if best.fourcc in ("MJPG", "YUY2")
                                    else "auto", "Формат кадров", why)]
            if best.fps < MIN_GOOD_FPS:
                recs.append(Recommendation(
                    "filter_min_cutoff", 0.8, "Сглаживание",
                    f"камера даёт только {best.fps:.0f} fps — больше сглаживания против рывков "
                    "(часто помогает больше света: камера снижает FPS в темноте)"))
    target_fps = min(max(cam_fps, 15.0), 60.0)
    budget = 1000.0 / target_fps * HEADROOM

    # ---- model / hands
    hand_ms = (bench or {}).get("hand_ms")
    results = [r for r in (bench or {}).get("results", []) if r.get("ms")]
    person_seen = any(r.get("detect_rate", 0) >= 0.5 for r in results)
    order = {"lite": 0, "full": 1, "heavy": 2}
    chosen = None
    if results and person_seen:
        for r in sorted(results, key=lambda r: (-order[r["variant"]], r["delegate"] != "GPU")):
            extra = (hand_ms or 0) / 2 if fingers else 0  # hands run every 2nd frame
            if r["ms"] + extra <= budget:
                chosen = r
                break
        if chosen is None:
            chosen = min(results, key=lambda r: r["ms"])
        rate = 1000.0 / max(chosen["ms"] + ((hand_ms or 0) / 2 if fingers else 0), 1)
        recs.append(Recommendation(
            "model", chosen["variant"], "Модель нейросети",
            f"{chosen['ms']:.0f} мс/кадр на вашем ПК → до {min(rate, cam_fps):.0f} fps с запасом "
            f"для игры (бюджет {budget:.0f} мс)"))
        recs.append(Recommendation("use_gpu", chosen["delegate"] == "GPU", "GPU",
                                   "по результатам замера"))
        if rate < MIN_GOOD_FPS:
            recs.append(Recommendation("filter_min_cutoff", 0.8, "Сглаживание",
                                       f"ПК обрабатывает ~{rate:.0f} fps — больше сглаживания"))
    else:
        threads = int(sysinfo.get("threads") or 4)
        variant = "full" if threads >= 6 else "lite"
        recs.append(Recommendation(
            "model", variant, "Модель нейросети",
            f"оценка по процессору ({threads} потоков); для точного замера встаньте в кадр "
            "и повторите автонастройку"))
    if fingers:
        if hand_ms:
            base = chosen["ms"] if chosen else budget / 2
            every = 1 if base + hand_ms <= budget else (2 if base + hand_ms / 2 <= budget else 3)
            recs.append(Recommendation("hand_every", every, "Пальцы: каждый N-й кадр",
                                       f"распознавание рук ≈ {hand_ms:.0f} мс"))
        else:
            recs.append(Recommendation("hand_every", 2, "Пальцы: каждый N-й кадр",
                                       "значение по умолчанию"))
    if cfg.output_smoothing:
        recs.append(Recommendation("send_rate", 90, "Частота отправки",
                                   "сглаживание вывода: 90 Гц, как у VR-шлема"))
    else:
        recs.append(Recommendation("send_rate", int(min(max(target_fps, 30), 90)),
                                   "Частота отправки", "под частоту обработки кадров"))

    # ---- target app
    preset = PRESETS.get(target)
    if preset is not None:
        recs.append(Recommendation("target", target, "Игра / приложение", preset.title))
        for k, v in preset.values.items():
            recs.append(Recommendation(k, v, _LABELS.get(k, k), f"для режима «{preset.title}»"))
    recs.append(Recommendation("fingers", fingers, "Отслеживание пальцев", "ваш выбор"))
    # keep only the last recommendation per key
    out: Dict[str, Recommendation] = {}
    for r in recs:
        out[r.key] = r
    return list(out.values())


def pc_profile_from(sysinfo: Dict[str, Any], bench: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What we store: numbers only (no CPU model name, no identifiers)."""
    return {"threads": sysinfo.get("threads"), "ram_gb": sysinfo.get("ram_gb"),
            "bench": (bench or {}).get("results", []), "hand_ms": (bench or {}).get("hand_ms"),
            "time": int(time.time())}


_LABELS = {
    "output_vrchat": "Вывод в VRChat (OSC)",
    "vrchat_port": "Порт VRChat", "send_head": "Отправлять голову",
    "hmd_anchor": "Привязка к шлему", "root_motion": "Перемещение по кадру",
    "trackers": "Трекеры", "finger_output": "Вывод пальцев",
    "output_driver": "Драйвер Nexovara Pose", "driver_hmd": "SteamVR без шлема",
}
format_value = _fmt
