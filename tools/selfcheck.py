"""Nexovara Pose self-check: run inside .venv (setup does this automatically).

  python -I tools/selfcheck.py [--mode test|vrchat|steamvr|auto|keep] [--no-models] [--no-camera]

Checks libraries, downloads and test-runs the models, looks for cameras and
(optionally) selects the game mode. Camera modes and the PC benchmark are done
by the program itself (first start / "Автонастройка").
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")  # hide noisy camera-probe warnings
sys.path.insert(0, str(ROOT))

problems = []
MODE_TO_TARGET = {"test": "test", "vrchat": "vrchat_osc", "steamvr": "steamvr_driver",
                  "auto": "auto"}


def ok(msg):
    print(f"    \033[32m✔\033[0m {msg}", flush=True)


def warn(msg):
    problems.append(msg)
    print(f"    \033[33m!\033[0m {msg}", flush=True)


def main() -> None:
    if sys.platform.startswith("win"):
        os.system("")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "vrchat", "steamvr", "auto", "keep"],
                    default="keep")
    ap.add_argument("--no-models", action="store_true")
    ap.add_argument("--no-camera", action="store_true")
    args = ap.parse_args()

    # 1. libraries
    import cv2
    import mediapipe as mp
    import numpy as np
    ok(f"mediapipe {mp.__version__}, OpenCV {cv2.__version__}, NumPy {np.__version__}")
    try:
        import PySide6
        import psutil
        ok(f"интерфейс Qt (PySide6 {PySide6.__version__}), psutil {psutil.__version__}")
    except ImportError as e:
        warn(f"не хватает компонента интерфейса: {e}")
    if sys.platform.startswith("win"):
        try:
            import cv2_enumerate_cameras
            ok(f"названия камер: cv2-enumerate-cameras {cv2_enumerate_cameras.__version__}")
        except ImportError:
            warn("нет пакета cv2-enumerate-cameras — камеры будут видны только по номерам "
                 "(запустите setup.bat ещё раз)")

    from fbt.config import apply_auto, app_dir, config_path, load_config, save_config
    from fbt.pose import ModelManifest, PoseEstimator, ensure_model
    from fbt.presets import PRESETS

    # 2. game mode (calibration and the user's own settings are kept)
    cfg = load_config()
    if args.mode != "keep":
        target = MODE_TO_TARGET[args.mode]
        values = {"target": target}
        if target in PRESETS:
            values.update(PRESETS[target].values)
        apply_auto(cfg, values, force=False)
        save_config(cfg)
        ok(f"режим «{PRESETS[target].title if target in PRESETS else 'авто'}» записан в "
           f"{config_path()}")
    ok(f"папка данных: {app_dir()}")

    # 3. models (hash recorded for later integrity checks)
    models_dir = ROOT / "models"
    manifest = ModelManifest(app_dir() / "models.json")
    model = None
    if not args.no_models:
        for variant in ("full", "lite", "hand"):
            try:
                last = [-1]

                def prog(p, v=variant):
                    pct = int(p * 100)
                    if pct // 25 != last[0] // 25:
                        last[0] = pct
                        print(f"      {v}: {pct}%", flush=True)
                path = ensure_model(variant, models_dir, (), prog, manifest)
                ok(f"модель {variant}: {path.name} ({path.stat().st_size / 1e6:.1f} МБ)")
                if variant != "hand" and (variant == cfg.model or model is None):
                    model = path
            except Exception as e:
                warn(f"модель {variant} не скачалась ({e}). Программа попробует скачать её "
                     f"при запуске, или положите файл вручную в {models_dir}")

    # 4. test inference
    if model is not None:
        try:
            est = PoseEstimator(model)
            frame = np.full((480, 640, 3), 90, np.uint8)
            t0 = time.perf_counter()
            for i in range(5):
                est.process(frame, i * 33)
            ms = (time.perf_counter() - t0) / 5 * 1000
            est.close()
            ok(f"нейросеть работает ({est.delegate}), ~{ms:.0f} мс на пустой кадр")
        except Exception as e:
            warn(f"нейросеть не запустилась: {e}")

    # 5. cameras
    if not args.no_camera:
        from fbt.camera import camera_label, find_device, list_camera_devices
        print("    поиск камер… (может мигнуть индикатор камеры)", flush=True)
        devs = list_camera_devices(4) or []
        if devs:
            ok("найдены камеры: " + "; ".join(d.label() for d in devs))
            if cfg.camera.isdigit():
                dev, st = find_device(devs, cfg.camera, cfg.camera_name)
                if dev is None:
                    dev = devs[0]
                    ok(f"камера «{camera_label(cfg.camera, cfg.camera_name)}» не найдена — "
                       f"выбрана «{dev.label()}»")
                elif st == "moved":
                    ok(f"камера «{dev.name}» теперь под номером {dev.index}")
                if (str(dev.index), dev.name) != (cfg.camera, cfg.camera_name):
                    cfg.camera, cfg.camera_name = str(dev.index), dev.name
                    save_config(cfg)
                ok(f"используется камера: {dev.label()}")
        else:
            warn("камера не найдена (не подключена или занята Discord/OBS/Zoom). "
                 "Для теста можно выбрать видеофайл в программе.")

    # 6. optional modules (new features; everything else works without them)
    try:
        import onnxruntime as ort
        prov = [p.replace("ExecutionProvider", "") for p in ort.get_available_providers()]
        ok(f"ONNX Runtime {ort.__version__} (RTMPose): " + ", ".join(prov))
    except Exception:
        print("    · ONNX Runtime не установлен — вторая нейросеть RTMPose недоступна "
              "(необязательно)", flush=True)
    try:
        import zeroconf
        ok(f"zeroconf {zeroconf.__version__} (поза шлема из VRChat)")
    except Exception:
        print("    · zeroconf не установлен — «поза шлема из VRChat» недоступна "
              "(необязательно)", flush=True)

    # 7. own SteamVR driver (optional, C++)
    from fbt import steamvr
    if steamvr.driver_built():
        ok("драйвер SteamVR Nexovara Pose собран" + (", установлен в SteamVR"
                                              if steamvr.driver_registered() else ""))
    else:
        print("    · драйвер SteamVR Nexovara Pose не собран (нужен только для режимов «драйвер Nexovara Pose» "
              "и «SteamVR без шлема»: build_driver.bat)", flush=True)

    if problems:
        print(f"\n    Предупреждений: {len(problems)} (см. выше). Остальное в порядке.")
    else:
        ok("самопроверка пройдена")
    print("    При первом запуске программа сама проверит режимы камеры и предложит "
          "автонастройку под ваш ПК.")


if __name__ == "__main__":
    main()
