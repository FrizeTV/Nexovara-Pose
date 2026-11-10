"""Target applications (game presets) and automatic target detection.

Detection only compares running process NAMES against a short fixed list on
this computer; nothing is stored or sent anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from . import steamvr

BASIC_TRACKERS = {"hip": True, "left_foot": True, "right_foot": True,
                  "left_knee": False, "right_knee": False, "chest": False,
                  "left_elbow": False, "right_elbow": False}


@dataclass
class Preset:
    id: str
    title: str
    description: str
    values: Dict[str, Any] = field(default_factory=dict)


PRESETS: Dict[str, Preset] = {
    "vrchat_osc": Preset(
        "vrchat_osc", "VRChat (OSC)",
        "Трекеры уходят прямо в VRChat по OSC. Нужен VR-шлем: VRChat в обычном режиме "
        "без шлема трекеры тела НЕ принимает (это ограничение VRChat) — без шлема выберите "
        "«SteamVR без шлема». В VRChat включите OSC и откалибруйте FBT.",
        dict(output_vrchat=True, output_driver=False, driver_hmd=False, vrchat_port=9000,
             send_head=True, hmd_anchor=False, root_motion=False, finger_output="preview",
             trackers=dict(BASIC_TRACKERS))),
    "steamvr_driver": Preset(
        "steamvr_driver", "SteamVR — драйвер Nexovara Pose",
        "Трекеры появляются в SteamVR как настоящие (VRChat через SteamVR, ChilloutVR, "
        "Resonite и др.). Шлем — ваш настоящий: драйвер сам узнаёт, где он, и «приклеивает» "
        "тело к нему. Драйвер нужно один раз собрать и установить (build_driver.bat, затем "
        "«Проверка и настройка SteamVR»).",
        dict(output_vrchat=False, output_driver=True, driver_hmd=False,
             hmd_anchor=True, root_motion=False, trackers=dict(BASIC_TRACKERS))),
    "steamvr_nohmd": Preset(
        "steamvr_nohmd", "SteamVR без шлема",
        "Шлема нет: драйвер Nexovara Pose создаёт виртуальный шлем, голова и руки тоже берутся с "
        "камеры, картинка игры — в окне на мониторе. Кнопки контроллеров — с клавиатуры: "
        "F8 — меню (в VRChat Quick Menu), F9 — оба курка (калибровка FBT), F10 — правая "
        "рука следует за мышью, клик — курок. С пальцами: согнуть указательный = курок.",
        dict(output_vrchat=False, output_driver=True, driver_hmd=True,
             hmd_anchor=False, root_motion=True, finger_output="driver",
             trackers=dict(BASIC_TRACKERS))),
    "test": Preset(
        "test", "Тест без игр",
        "Данные идут на этот ПК (порт 9000) — смотрите их в osc_monitor.bat.",
        dict(output_vrchat=True, output_driver=False, driver_hmd=False, vrchat_port=9000,
             send_head=True, hmd_anchor=False, root_motion=False, finger_output="preview")),
}

GAME_PROCS = {
    "vrchat.exe": "VRChat", "vrchat": "VRChat",
    "chilloutvr.exe": "ChilloutVR", "resonite.exe": "Resonite", "neosvr.exe": "NeosVR",
}


def running_apps() -> Dict[str, bool]:
    found = {"steamvr": False}
    try:
        import psutil
        for proc in psutil.process_iter(["name"]):
            n = (proc.info.get("name") or "").lower()
            if n in steamvr.STEAMVR_PROCS:
                found["steamvr"] = True
            elif n in GAME_PROCS:
                found[GAME_PROCS[n]] = True
    except Exception:
        pass
    return found


def detect_target() -> Tuple[str, str]:
    """Best preset for what is running right now, plus a human-readable reason."""
    apps = running_apps()
    own = steamvr.driver_registered()
    games = [g for g in ("VRChat", "ChilloutVR", "Resonite", "NeosVR") if apps.get(g)]
    if apps["steamvr"]:
        if own:
            if steamvr.no_hmd_mode_enabled():
                return "steamvr_nohmd", "запущен SteamVR с драйвером Nexovara Pose в режиме без шлема"
            return "steamvr_driver", "запущен SteamVR, драйвер Nexovara Pose установлен"
        # without our driver in SteamVR, OSC is the only way that can work
        return "vrchat_osc", "запущен SteamVR, драйвер Nexovara Pose не установлен → OSC (VRChat)"
    if own and steamvr.no_hmd_mode_enabled():
        return "steamvr_nohmd", "SteamVR настроен на режим без шлема (запустите SteamVR)"
    if "VRChat" in games:
        return "vrchat_osc", "запущен VRChat"
    if games and own:
        return "steamvr_driver", f"запущен {games[0]} — нужен SteamVR"
    return "vrchat_osc", "ничего не запущено — режим по умолчанию"
