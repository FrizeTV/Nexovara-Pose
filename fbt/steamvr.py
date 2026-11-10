"""SteamVR helpers.

* HmdReader — read the headset pose through `openvr` (optional package). Only a
  fallback: the Nexovara Pose driver itself reports the headset pose (see hmd.py).
  Read-only.
* find_paths / is_running — detection (read-only).
* assign_tracker_roles — writes tracker roles (Waist, Left Foot, ...) for the
  devices of the Nexovara Pose driver into steamvr.vrsettings. Only on explicit user
  request, only while SteamVR is closed, after a timestamped backup,
  atomically, touching nothing but the "trackers" entries of our own devices.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

_S = np.diag([1.0, 1.0, -1.0])  # OpenVR (right-handed) <-> Unity (left-handed)

ROLE_BY_TRACKER = {
    "hip": "TrackerRole_Waist",
    "left_foot": "TrackerRole_LeftFoot",
    "right_foot": "TrackerRole_RightFoot",
    "left_knee": "TrackerRole_LeftKnee",
    "right_knee": "TrackerRole_RightKnee",
    "chest": "TrackerRole_Chest",
    "left_elbow": "TrackerRole_LeftElbow",
    "right_elbow": "TrackerRole_RightElbow",
}
STEAMVR_PROCS = {"vrserver.exe", "vrmonitor.exe", "vrserver", "vrmonitor"}
MAX_SETTINGS_BYTES = 5_000_000


class HmdReader:
    def __init__(self):
        import openvr  # optional dependency
        self._openvr = openvr
        self._vr = openvr.init(openvr.VRApplication_Background)
        log.info("Connected to SteamVR (background app)")

    def get_pose(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(position, rotation) of the HMD in SteamVR's RAW space, Unity axes —
        the same space the Nexovara Pose driver reports its devices in."""
        ovr = self._openvr
        try:
            poses = self._vr.getDeviceToAbsoluteTrackingPose(
                ovr.TrackingUniverseRawAndUncalibrated, 0.0, ovr.k_unMaxTrackedDeviceCount)
            p = poses[ovr.k_unTrackedDeviceIndex_Hmd]
            if not p.bPoseIsValid:
                return None
            m = p.mDeviceToAbsoluteTracking
            M = np.array([[m[i][j] for j in range(4)] for i in range(3)], dtype=np.float64)
        except Exception as e:
            log.debug("HMD pose read failed: %s", e)
            return None
        pos = _S @ M[:, 3]
        rot = _S @ M[:, :3] @ _S
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(rot))):
            return None
        return pos, rot

    def close(self) -> None:
        try:
            self._openvr.shutdown()
        except Exception:
            pass


def try_connect() -> Tuple[Optional[HmdReader], str]:
    if not is_running():
        return None, "SteamVR не запущен — привязка к шлему отключена"
    try:
        return HmdReader(), "SteamVR: шлем подключён"
    except ImportError:
        return None, "SteamVR: модуль openvr не установлен (setup.bat --mode steamvr)"
    except Exception as e:
        return None, f"SteamVR недоступен: {e}"


# ------------------------------------------------------------- detection
@dataclass
class OpenVRPaths:
    runtime: List[Path] = field(default_factory=list)
    config: List[Path] = field(default_factory=list)
    external_drivers: List[Path] = field(default_factory=list)


def _vrpath_file() -> Optional[Path]:
    if sys.platform.startswith("win"):
        la = os.environ.get("LOCALAPPDATA")
        return Path(la) / "openvr" / "openvrpaths.vrpath" if la else None
    return Path.home() / ".config" / "openvr" / "openvrpaths.vrpath"


def find_paths() -> Optional[OpenVRPaths]:
    p = _vrpath_file()
    try:
        if p is None or not p.is_file() or p.stat().st_size > 200_000:
            return None
        d = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None

    def paths(key):
        v = d.get(key) or []
        return [Path(x) for x in v if isinstance(x, str) and x][:32] if isinstance(v, list) else []
    out = OpenVRPaths(paths("runtime"), paths("config"), paths("external_drivers"))
    out.runtime = [r for r in out.runtime if r.is_dir()]
    return out if out.runtime else None


def is_installed() -> bool:
    return find_paths() is not None


def is_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    try:
        for proc in psutil.process_iter(["name"]):
            n = (proc.info.get("name") or "").lower()
            if n in STEAMVR_PROCS:
                return True
    except Exception:
        pass
    return False


def settings_file() -> Optional[Path]:
    paths = find_paths()
    if paths is None:
        return None
    for c in paths.config:
        f = c / "steamvr.vrsettings"
        if f.is_file():
            return f
    return None


# ------------------------------------------------------------- roles
def driver_device_path(name: str) -> str:
    """Device path of a body tracker created by the Nexovara Pose driver (serial = Nexovara Pose-<name>)."""
    # SteamVR device paths are protocol identifiers; keep the legacy serial
    # prefix so existing role assignments survive the product rename.
    return f"/devices/nexovarapose/Nexovara Pose-{name}"


def planned_roles(enabled: List[str]) -> Dict[str, str]:
    return {driver_device_path(n): ROLE_BY_TRACKER[n] for n in enabled if n in ROLE_BY_TRACKER}


def _edit_settings(edit, settings: Optional[Path] = None) -> Path:
    """Safely change steamvr.vrsettings: SteamVR closed, size/format checked,
    timestamped backup, atomic write. `edit(data)` modifies the dict in place."""
    if is_running():
        raise RuntimeError("Закройте SteamVR: он перезаписывает настройки при выходе")
    settings = settings or settings_file()
    if settings is None:
        raise RuntimeError("Файл настроек SteamVR (steamvr.vrsettings) не найден")
    if settings.stat().st_size > MAX_SETTINGS_BYTES:
        raise RuntimeError("Файл настроек SteamVR слишком большой — не трогаю")
    data = json.loads(settings.read_text(encoding="utf-8"))  # broken file -> ValueError, nothing changed
    if not isinstance(data, dict):
        raise RuntimeError("Неожиданный формат steamvr.vrsettings")
    edit(data)
    backup = settings.with_name(f"steamvr.vrsettings.nexovarapose-backup-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(settings, backup)
    fd, tmp = tempfile.mkstemp(dir=str(settings.parent), prefix=".steamvr-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=3, ensure_ascii=False)
        os.replace(tmp, settings)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return backup


def assign_tracker_roles(enabled: List[str], settings: Optional[Path] = None) -> Path:
    """Write roles for the trackers of the Nexovara Pose driver. Returns the backup path."""
    roles = planned_roles(enabled)
    if not roles:
        raise RuntimeError("Нет включённых трекеров")

    def edit(data):
        trackers = data.get("trackers")
        if trackers is None:
            trackers = {}
        if not isinstance(trackers, dict):
            raise RuntimeError("Неожиданный формат раздела trackers")
        trackers.update(roles)
        data["trackers"] = trackers
    backup = _edit_settings(edit, settings)
    log.info("SteamVR tracker roles written (%d), backup %s", len(roles), backup.name)
    return backup


# ------------------------------------------------------------- own driver
DRIVER_NAME = "nexovarapose"
DRIVER_SECTION = "driver_nexovarapose"


def driver_dist_dir() -> Path:
    """Folder produced by build_driver.bat: driver/dist/nexovarapose."""
    from .config import project_root
    return project_root() / "driver" / "dist" / DRIVER_NAME


def _binary_in(folder: Path) -> Path:
    d = folder / "bin"
    if sys.platform.startswith("win"):
        return d / "win64" / f"driver_{DRIVER_NAME}.dll"
    return d / "linux64" / f"driver_{DRIVER_NAME}.so"


def driver_binary() -> Path:
    return _binary_in(driver_dist_dir())


def driver_built() -> bool:
    return driver_binary().is_file() and (driver_dist_dir() / "driver.vrdrivermanifest").is_file()


def _program_data() -> Optional[Path]:
    pd = os.environ.get("ProgramData") or os.environ.get("PROGRAMDATA")
    return Path(pd) if pd else None


def driver_install_dir() -> Path:
    """Where SteamVR loads the driver from.

    Windows: a copy in %ProgramData%\\Nexovara Pose\\SteamVR\\nexovarapose. SteamVR's tools
    break paths with non-Latin letters (C:\\Users\\Пользователь\\... ends up
    garbled in openvrpaths.vrpath and the driver is silently not loaded), and
    SteamVR locks the DLL while it runs (rebuilding in place would fail).
    Other systems / fallback: the build folder itself."""
    if sys.platform.startswith("win"):
        pd = _program_data()
        if pd is not None and str(pd).isascii():
            return pd / "Nexovara Pose" / "SteamVR" / DRIVER_NAME
    return driver_dist_dir()


def _tree_digest(folder: Path) -> Dict[str, str]:
    import hashlib
    out = {}
    if not folder.is_dir():
        return out
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.stat().st_size < 50_000_000:
            out[f.relative_to(folder).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    return out


def installed_copy_current() -> bool:
    """The copy SteamVR uses equals the freshly built driver."""
    inst, dist = driver_install_dir(), driver_dist_dir()
    if inst.resolve() == dist.resolve():
        return driver_built()
    a, b = _tree_digest(dist), _tree_digest(inst)
    return bool(a) and a == b


def _read_vrpaths() -> Tuple[Optional[Path], Optional[dict]]:
    p = _vrpath_file()
    if p is None or not p.is_file() or p.stat().st_size > 200_000:
        return p, None
    try:
        # surrogateescape: entries that other tools wrote in a wrong encoding are
        # kept byte-for-byte when we write the file back
        d = json.loads(p.read_bytes().decode("utf-8", errors="surrogateescape"))
    except (OSError, ValueError):
        return p, None
    return p, d if isinstance(d, dict) else None


def _is_our_entry(entry: str) -> bool:
    # last path component, Windows or POSIX separators (entries may come from Windows)
    if entry.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower() == DRIVER_NAME:
        return True
    try:
        return Path(entry).resolve() in (driver_dist_dir().resolve(),
                                         driver_install_dir().resolve())
    except (OSError, ValueError):
        return False


def _write_vrpaths(edit) -> None:
    """Change openvrpaths.vrpath (SteamVR's list of driver folders) ourselves:
    UTF-8, backup, atomic. Replaces SteamVR's vrpathreg, which garbles non-Latin
    letters in paths."""
    p, data = _read_vrpaths()
    if p is None or data is None:
        raise RuntimeError("SteamVR не найден (нет файла openvrpaths.vrpath) — установите и "
                           "один раз запустите SteamVR")
    edit(data)
    backup = p.with_name(f"openvrpaths.vrpath.nexovarapose-backup-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(p, backup)
    raw = json.dumps(data, indent="\t", ensure_ascii=False).encode("utf-8", errors="surrogateescape")
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".openvrpaths-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def registered_driver_paths() -> List[str]:
    _p, data = _read_vrpaths()
    if not data:
        return []
    ext = data.get("external_drivers") or []
    return [e for e in ext if isinstance(e, str) and _is_our_entry(e)] if isinstance(ext, list) \
        else []


def driver_registered() -> bool:
    """Registered exactly once, at the install folder, and that folder has the driver."""
    regs = registered_driver_paths()
    inst = driver_install_dir()
    if len(regs) != 1:
        return False
    try:
        same = Path(regs[0]).resolve() == inst.resolve()
    except (OSError, ValueError):
        return False
    return same and _binary_in(inst).is_file() and (inst / "driver.vrdrivermanifest").is_file()


def _enable_in_settings(data: dict) -> None:
    """SteamVR switches add-ons off after a crash ("safe mode"): switch ours on again."""
    drv = data.get(DRIVER_SECTION)
    drv = drv if isinstance(drv, dict) else {}
    drv["enable"] = True
    drv.pop("blocked_by_safe_mode", None)
    data[DRIVER_SECTION] = drv


def driver_blocked(settings: Optional[Path] = None) -> bool:
    drv = _settings_section(settings, DRIVER_SECTION)
    return drv.get("enable") is False or bool(drv.get("blocked_by_safe_mode"))


def _settings_section(settings: Optional[Path], name: str) -> dict:
    settings = settings or settings_file()
    try:
        data = json.loads(settings.read_text(encoding="utf-8")) if settings else {}
        sec = data.get(name) if isinstance(data, dict) else None
        return sec if isinstance(sec, dict) else {}
    except (OSError, ValueError):
        return {}


def register_driver() -> Path:
    """Install the built driver for SteamVR (copy to an ASCII-only folder on
    Windows, register it, switch it on). Returns the folder SteamVR loads."""
    if is_running():
        raise RuntimeError("Закройте SteamVR перед установкой драйвера")
    if not driver_built():
        raise RuntimeError("Драйвер ещё не собран — запустите build_driver.bat")
    dist, inst = driver_dist_dir(), driver_install_dir()
    # Keep the configured install location.  Older SteamVR tools could corrupt
    # non-ASCII paths, but the registration code below writes UTF-8 atomically
    # and current SteamVR accepts them.  Rejecting the path here also made the
    # repair/diagnostic flow fail before it could copy and register the driver.
    if inst.resolve() != dist.resolve():
        tmp = inst.with_name(inst.name + ".new")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(dist, tmp)
        old = inst.with_name(inst.name + ".old")
        shutil.rmtree(old, ignore_errors=True)
        if inst.exists():
            os.replace(inst, old)
        os.replace(tmp, inst)
        shutil.rmtree(old, ignore_errors=True)

    def edit(data):
        ext = data.get("external_drivers")
        ext = ext if isinstance(ext, list) else []
        # drop every older Nexovara Pose entry (also ones garbled by vrpathreg)
        data["external_drivers"] = [e for e in ext if not (isinstance(e, str)
                                                           and _is_our_entry(e))] + [str(inst)]
    _write_vrpaths(edit)
    if settings_file() is not None:
        _edit_settings(_enable_in_settings)
    log.info("Nexovara Pose driver installed: %s", inst)
    return inst


def unregister_driver() -> None:
    if is_running():
        raise RuntimeError("Закройте SteamVR перед удалением драйвера")

    def edit(data):
        ext = data.get("external_drivers")
        if isinstance(ext, list):
            data["external_drivers"] = [e for e in ext if not (isinstance(e, str)
                                                               and _is_our_entry(e))]
    _write_vrpaths(edit)
    inst = driver_install_dir()
    if inst.resolve() != driver_dist_dir().resolve():
        shutil.rmtree(inst, ignore_errors=True)
    if settings_file() is not None and no_hmd_mode_enabled():
        set_no_hmd_mode(False)  # never leave SteamVR waiting for a virtual headset


def no_hmd_mode_enabled(settings: Optional[Path] = None) -> bool:
    return bool(_settings_section(settings, DRIVER_SECTION).get("enableHmd", False))


_RESTORE_KEY = "nexovarapose_restore"
_NOHMD_KEYS = ("requireHmd", "forcedDriver", "activateMultipleDrivers")


def set_no_hmd_mode(enable: bool, settings: Optional[Path] = None,
                    window: Tuple[int, int] = (1600, 900), vertical_fov: int = 90) -> Path:
    """'SteamVR without a headset': the Nexovara Pose driver provides a virtual HMD that
    renders into a window. SteamVR must not require a real headset, must use our
    driver for the HMD and run other drivers (trackers) next to it. The previous
    values are remembered and restored when the mode is switched off."""
    def edit(data):
        drv = data.get(DRIVER_SECTION)
        drv = drv if isinstance(drv, dict) else {}
        sv = data.get("steamvr")
        sv = sv if isinstance(sv, dict) else {}
        drv["enableHmd"] = bool(enable)
        if enable:
            if not isinstance(drv.get(_RESTORE_KEY), dict):
                drv[_RESTORE_KEY] = {k: sv.get(k) for k in _NOHMD_KEYS}
            drv["enable"] = True
            drv.pop("blocked_by_safe_mode", None)
            drv["window_width"], drv["window_height"] = int(window[0]), int(window[1])
            drv["vertical_fov"] = int(vertical_fov)
            sv["requireHmd"] = False
            sv["forcedDriver"] = DRIVER_NAME
            sv["activateMultipleDrivers"] = True
        else:
            prev = drv.pop(_RESTORE_KEY, None)
            prev = prev if isinstance(prev, dict) else {}
            for k in ("requireHmd", "forcedDriver"):
                if k == "forcedDriver" and sv.get(k) not in (None, DRIVER_NAME):
                    continue  # changed by someone else meanwhile: keep it
                old = prev.get(k)
                if old is None or old == DRIVER_NAME:
                    sv.pop(k, None)   # SteamVR's default
                else:
                    sv[k] = old
            sv["activateMultipleDrivers"] = True  # trackers next to a real headset
        data[DRIVER_SECTION] = drv
        data["steamvr"] = sv
    return _edit_settings(edit, settings)


# ------------------------------------------------------------- diagnostics
def steamvr_log_file() -> Optional[Path]:
    _p, data = _read_vrpaths()
    dirs = (data or {}).get("log") or []
    for d in dirs if isinstance(dirs, list) else []:
        if isinstance(d, str):
            f = Path(d) / "vrserver.txt"
            if f.is_file():
                return f
    return None


def driver_log_lines(limit: int = 40, path: Optional[Path] = None) -> List[str]:
    """Nexovara Pose lines from SteamVR's log of the last/current session (vrserver.txt)."""
    path = path or steamvr_log_file()
    if path is None:
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 3_000_000))
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    keys = ("nexovarapose", "requirehmd", "forced driver", "forceddriver")
    out = [ln.strip() for ln in text.splitlines()
           if any(k in ln.lower() for k in keys)]
    return out[-limit:]


@dataclass
class Check:
    title: str
    ok: Optional[bool]          # None = information only
    text: str = ""


def diagnose(target_nohmd: bool) -> List[Check]:
    """Step-by-step state of 'Nexovara Pose -> driver -> SteamVR' for the check dialog."""
    out: List[Check] = []
    paths = find_paths()
    if paths is None:
        out.append(Check("SteamVR установлен", False, "Установите SteamVR в Steam и запустите "
                                                       "его один раз."))
        return out
    out.append(Check("SteamVR установлен", True, str(paths.runtime[0])))
    running = is_running()
    out.append(Check("SteamVR запущен", None, "да" if running else "нет (изменять настройки можно "
                                                                  "только при закрытом SteamVR)"))
    if not driver_built():
        out.append(Check("Драйвер собран", False, "Запустите build_driver.bat в папке программы."))
        return out
    out.append(Check("Драйвер собран", True, str(driver_binary())))
    regs = registered_driver_paths()
    if driver_registered():
        current = installed_copy_current()
        out.append(Check("Драйвер установлен в SteamVR", current,
                         str(driver_install_dir()) + ("" if current else
                                                      " — устарел, нажмите «Исправить»")))
    else:
        bad = [r for r in regs if not str(r).isascii() or not Path(r).is_dir()]
        why = ("в SteamVR записан неверный путь (русские буквы или папка удалена)" if bad else
               "несколько записей" if len(regs) > 1 else "не установлен")
        out.append(Check("Драйвер установлен в SteamVR", False, why + " — нажмите «Исправить»"))
    sf = settings_file()
    if sf is not None:
        blocked = driver_blocked(sf)
        out.append(Check("Драйвер включён в SteamVR", not blocked,
                         "SteamVR отключил дополнение (после сбоя) — «Исправить» включит"
                         if blocked else "да"))
        sv = _settings_section(sf, "steamvr")
        nohmd = no_hmd_mode_enabled(sf)
        if target_nohmd or nohmd:
            good = nohmd and sv.get("requireHmd") is False and sv.get("forcedDriver") == DRIVER_NAME
            out.append(Check("Режим без шлема в SteamVR", good,
                             "включён" if good else "не настроен — «Исправить» настроит"))
        trackers = (json.loads(sf.read_text(encoding="utf-8")).get("trackers") or {}) \
            if sf.is_file() else {}
        ours = {k: v for k, v in trackers.items() if k.startswith(f"/devices/{DRIVER_NAME}/")} \
            if isinstance(trackers, dict) else {}
        out.append(Check("Роли трекеров", bool(ours) or None,
                         ", ".join(f"{k.rsplit('-', 1)[-1]}→{str(v).replace('TrackerRole_', '')}"
                                   for k, v in ours.items()) or
                         "не назначены (кнопка «Назначить роли»)"))
    lines = driver_log_lines(200)
    if lines:
        loaded = any("driver initialised" in ln for ln in lines)
        failed = [ln for ln in lines if ("unable to load" in ln.lower() or "failed" in ln.lower())
                  and "nexovarapose" in ln.lower()]
        recv = [ln for ln in lines if "receiving data" in ln or "no data from" in ln]
        if failed:
            out.append(Check("SteamVR загрузил драйвер", False, failed[-1][-200:]))
        else:
            out.append(Check("SteamVR загрузил драйвер", loaded or None,
                             "да (журнал SteamVR)" if loaded else
                             "в журнале последнего запуска SteamVR драйвера нет"))
        if loaded:
            out.append(Check("Данные от Nexovara Pose доходят до драйвера",
                             bool(recv) and "receiving data" in recv[-1] or None,
                             "да" if recv and "receiving data" in recv[-1] else
                             ("последнее: " + recv[-1][-120:] if recv else
                              "пока нет — нажмите «Старт» в Nexovara Pose")))
    else:
        out.append(Check("Журнал SteamVR", None, "не найден (SteamVR ещё не запускался?)"))
    return out


def fix_all(target_nohmd: bool, window: Tuple[int, int] = (1600, 900),
            vertical_fov: int = 90) -> List[str]:
    """Everything the check dialog can repair (SteamVR must be closed)."""
    if is_running():
        raise RuntimeError("Закройте SteamVR (полностью, вместе с окном статуса) и нажмите ещё раз")
    done = []
    if not (driver_registered() and installed_copy_current()) or driver_blocked():
        inst = register_driver()
        done.append(f"драйвер установлен: {inst}")
    if target_nohmd and settings_file() is not None:
        set_no_hmd_mode(True, window=window, vertical_fov=vertical_fov)
        done.append("режим без шлема включён")
    return done


# ------------------------------------------------------------- VRChat
def vrchat_desktop_running() -> bool:
    """VRChat runs WITHOUT VR (desktop mode): it then ignores body trackers."""
    try:
        import psutil
    except ImportError:
        return False
    try:
        for proc in psutil.process_iter(["name"]):
            if (proc.info.get("name") or "").lower() not in ("vrchat.exe", "vrchat"):
                continue
            try:
                cmd = " ".join(proc.cmdline()).lower()
            except Exception:
                cmd = ""
            return "--no-vr" in cmd or not is_running()
    except Exception:
        pass
    return False
