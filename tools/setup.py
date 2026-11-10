"""Nexovara Pose: automatic, safe setup into the project folder.

Run via setup.bat / setup.sh (or: python -I tools/setup.py [--yes] ...).
Uses only the Python standard library.

What it does:
  1. creates a private virtual environment in <project>/.venv
  2. installs dependencies from PyPI using ONLY pre-built wheels
     (--only-binary=:all:  -> no third-party build scripts are executed)
  3. turns on portable mode: config, logs, models live in <project>/data
  4. downloads the pose models from Google's official URL (HTTPS, size and
     format checks) into <project>/models
  5. checks for SteamVR. Only if it is installed (or you ask for it) installs
     `openvr` (read-only access to the headset pose; optional — the Nexovara Pose
     driver reports the headset pose itself). The SteamVR driver is built with
     build_driver.bat and installed from the program.
  6. optional modules (asked, failure is only a warning): ONNX Runtime
     (onnxruntime-directml on Windows — the RTMPose add-on on any GPU) and
     zeroconf (the headset pose from VRChat via OSCQuery)
  7. records the chosen game mode (default: automatic) and runs a self-check

Nothing is installed system-wide, nothing needs admin rights, nothing in
SteamVR/VRChat is modified. To uninstall, delete the project folder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
IS_WIN = sys.platform.startswith("win")
VENV_PY = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
MIN_PY, MAX_PY = (3, 10), (3, 14)


# ----------------------------------------------------------------- output
def _enable_ansi() -> None:
    if IS_WIN:
        os.system("")  # turns on VT100 colour codes in the Windows console


def step(msg: str) -> None:
    print(f"\n\033[1;36m==> {msg}\033[0m", flush=True)


def ok(msg: str) -> None:
    print(f"    \033[32m✔\033[0m {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"    \033[33m!\033[0m {msg}", flush=True)


def fail(msg: str) -> "None":
    print(f"\n\033[31m✖ {msg}\033[0m", flush=True)
    sys.exit(1)


def ask(question: str, default: bool, assume_yes: bool) -> bool:
    if assume_yes:
        return default
    hint = "[Y/n]" if default else "[y/N]"
    try:
        a = input(f"    {question} {hint}: ").strip().lower()
    except EOFError:
        return default
    if not a:
        return default
    return a in ("y", "yes", "д", "да")


def run(cmd, **kw) -> None:
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=str(ROOT), **kw)
    if r.returncode != 0:
        fail(f"Команда завершилась с ошибкой (код {r.returncode})")


# ----------------------------------------------------------------- steps
def check_python() -> None:
    v = sys.version_info[:2]
    if not (MIN_PY <= v <= MAX_PY):
        fail(f"Нужен Python {MIN_PY[0]}.{MIN_PY[1]}–{MAX_PY[0]}.{MAX_PY[1]}, "
             f"а запущен {v[0]}.{v[1]}. Подойдёт любой из этих.")
    if sys.maxsize <= 2 ** 32:
        fail("Нужен 64-битный Python (MediaPipe не работает на 32-битном).")
    ok(f"Python {sys.version.split()[0]} ({sys.executable})")


def make_venv() -> None:
    step("Виртуальное окружение в папке проекта (.venv)")
    if VENV_PY.is_file():
        r = subprocess.run([str(VENV_PY), "-c", "import sys; print(sys.version_info[:2])"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            ok("уже есть, используем его")
            return
        warn("окружение повреждено, пересоздаём")
        shutil.rmtree(VENV, ignore_errors=True)
    run([sys.executable, "-m", "venv", VENV])
    ok("создано")


def pip_install(*args: str) -> None:
    run([VENV_PY, "-m", "pip", "install", "--disable-pip-version-check",
         "--only-binary=:all:", *args])


def install_requirements() -> None:
    step("Установка зависимостей (только готовые пакеты с PyPI)")
    pip_install("--upgrade", "pip")
    pip_install("-r", str(ROOT / "requirements.txt"))
    # opencv-python conflicts with opencv-contrib-python (both provide cv2)
    r = subprocess.run([str(VENV_PY), "-m", "pip", "show", "opencv-python"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        warn("найден конфликтующий opencv-python — удаляем и переустанавливаем contrib")
        run([VENV_PY, "-m", "pip", "uninstall", "-y", "opencv-python"])
        pip_install("--force-reinstall", "--no-deps", "opencv-contrib-python>=4.8,<6")
    ok("зависимости установлены")


def enable_portable() -> None:
    step("Портативный режим (настройки, журнал и модели — в папке проекта)")
    marker = ROOT / "portable.txt"
    if not marker.exists():
        marker.write_text("Этот файл включает портативный режим Nexovara Pose: настройки и журнал "
                          "хранятся в папке data рядом с программой.\n"
                          "Удалите его, чтобы хранить их в профиле пользователя.\n",
                          encoding="utf-8")
    (ROOT / "data").mkdir(exist_ok=True)
    ok(f"данные: {ROOT / 'data'}")


def find_steamvr() -> Path | None:
    """Locate the SteamVR runtime via OpenVR's own registry file (read-only)."""
    cands = []
    if IS_WIN:
        la = os.environ.get("LOCALAPPDATA")
        if la:
            cands.append(Path(la) / "openvr" / "openvrpaths.vrpath")
    else:
        cands.append(Path.home() / ".config" / "openvr" / "openvrpaths.vrpath")
    for p in cands:
        try:
            if p.is_file() and p.stat().st_size < 100_000:
                data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                for rt in data.get("runtime") or []:
                    if isinstance(rt, str) and Path(rt).is_dir():
                        return Path(rt)
        except (OSError, ValueError, AttributeError):
            continue
    return None


# ----------------------------------------------------------------- main
def main() -> None:
    _enable_ansi()
    ap = argparse.ArgumentParser(description="Nexovara Pose setup")
    ap.add_argument("--yes", "-y", action="store_true", help="не задавать вопросов")
    ap.add_argument("--mode", choices=["auto", "test", "vrchat", "steamvr"], default="auto",
                    help="test — проверка без игр; vrchat — VRChat OSC; "
                         "steamvr — SteamVR (драйвер Nexovara Pose)")
    ap.add_argument("--no-models", action="store_true", help="не скачивать модели")
    args = ap.parse_args()

    print("\033[1mNexovara Pose — автоматическая установка\033[0m")
    print(f"Папка проекта: {ROOT}")
    step("Проверка Python")
    check_python()
    make_venv()
    install_requirements()
    enable_portable()

    step("Поиск SteamVR")
    steamvr = find_steamvr()
    if steamvr:
        ok(f"SteamVR найден: {steamvr}")
    else:
        warn("SteamVR не найден — компоненты для SteamVR не нужны")

    mode = args.mode  # auto: the program picks the game mode itself at every start
    want_steamvr = mode == "steamvr" or (
        mode == "auto" and steamvr is not None
        and ask("Установить компоненты для SteamVR-игр (openvr — запасной способ узнать позу "
                "шлема)?", True, args.yes))
    if want_steamvr:
        step("Компоненты для SteamVR")
        try:
            pip_install("openvr>=1.26,<3")
            ok("openvr установлен (только чтение позы шлема)")
        except SystemExit:
            warn("openvr не установился — не страшно: позу шлема сообщает драйвер Nexovara Pose")

    want_extras = ask("Установить модули для дополнительных функций (ONNX Runtime — вторая "
                      "нейросеть RTMPose на видеокарте/процессоре, ~20–60 МБ; zeroconf — поза "
                      "шлема из VRChat)? Без них всё остальное работает.", True, args.yes)
    if want_extras:
        step("Дополнительные модули (необязательные)")
        ort = "onnxruntime-directml>=1.20,<2" if IS_WIN else "onnxruntime>=1.20,<2"
        for spec, why in ((ort, "RTMPose (уточнение суставов"
                                + (", любая видеокарта через DirectML)" if IS_WIN else ")")),
                          ("zeroconf>=0.130,<1", "поза шлема из VRChat (OSCQuery)")):
            try:
                pip_install(spec)
                ok(f"{spec.split('>=')[0]} — {why}")
            except SystemExit:
                warn(f"{spec.split('>=')[0]} не установился — функция «{why}» будет "
                     "недоступна, остальное работает")

    step("Модели нейросети и самопроверка")
    cmd = [VENV_PY, "-I", ROOT / "tools" / "selfcheck.py", "--mode", mode]
    if args.no_models:
        cmd.append("--no-models")
    run(cmd)

    print("\n\033[1;32mГотово!\033[0m")
    print("  1. Запустите run.bat. При первом запуске откроется автонастройка: программа")
    print("     проверит камеру, замерит скорость вашего ПК и предложит настройки.")
    print("  2. Нажмите «Старт» (F5), встаньте в кадр целиком и нажмите «Калибровка» (F6).")
    print("  Тест без игр: запустите osc_monitor.bat — там видно, какие данные уходят.")
    if want_steamvr:
        print("  SteamVR: запустите build_driver.bat, затем в Nexovara Pose: Настройки → Игра и вывод →")
        print("  «Проверка и настройка SteamVR» → «Исправить».")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        fail("Прервано пользователем")
