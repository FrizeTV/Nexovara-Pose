"""Settings: dataclass + strict validation + atomic JSON persistence.

Nothing from the config file is ever executed: values are type-coerced and
clamped to safe ranges; unknown keys are ignored.

Auto-configuration vs. user choices
-----------------------------------
`user_set` lists the settings the user changed by hand. Automatic tuning
(camera probe, PC benchmark, game presets) never overwrites those keys —
see `apply_auto()`. "Reset to automatic" simply clears `user_set`.
"""
from __future__ import annotations

import copy
import json
import logging
import math
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import APP_NAME, LEGACY_APP_NAME
from .model_catalog import clean_selection

log = logging.getLogger(__name__)

SCHEMA_VERSION = 4

# Fixed order. VRChat OSC slot = index + 1, Nexovara Pose driver device id = index.
# Fixed slots keep SteamVR role assignments stable when trackers are toggled.
TRACKER_NAMES = [
    "hip", "left_foot", "right_foot",
    "left_knee", "right_knee", "chest",
    "left_elbow", "right_elbow",
]
TRACKER_LABELS_RU = {
    "hip": "Таз (бёдра)",
    "left_foot": "Левая стопа",
    "right_foot": "Правая стопа",
    "left_knee": "Левое колено",
    "right_knee": "Правое колено",
    "chest": "Грудь",
    "left_elbow": "Левый локоть",
    "right_elbow": "Правый локоть",
}
MODEL_VARIANTS = ("lite", "full", "heavy")
REFINER_CHOICES = ("off", "rtmpose_m", "rtmpose_x")
FOURCC_CHOICES = ("auto", "MJPG", "YUY2")
# unknown (e.g. old "vrchat_quest") -> auto; old "steamvr" (VMT) -> "steamvr_driver" on load
TARGETS = ("auto", "vrchat_osc", "steamvr_driver", "steamvr_nohmd", "test")
FINGER_OUTPUTS = ("preview", "driver")


def project_root() -> Path:
    if getattr(sys, "frozen", False):  # PyInstaller build
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


PORTABLE_MARKER = "portable.txt"


def app_dir() -> Path:
    # Portable mode: if <project>/portable.txt exists, keep config, logs and
    # models inside <project>/data instead of the user profile.
    root = project_root()
    if (root / PORTABLE_MARKER).is_file():
        d = root / "data"
    elif sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        d = base / APP_NAME
    elif sys.platform == "darwin":
        d = Path.home() / "Library" / "Application Support" / APP_NAME
    else:
        d = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    # Preserve settings/models from the previous product name on first launch.
    legacy = d.parent / LEGACY_APP_NAME
    if legacy.is_dir() and not any(d.iterdir()):
        try:
            import shutil
            for item in legacy.iterdir():
                target = d / item.name
                if not target.exists():
                    shutil.copytree(item, target) if item.is_dir() else shutil.copy2(item, target)
        except OSError as e:
            log.warning("legacy settings migration failed: %s", e)
    return d


def config_path() -> Path:
    return app_dir() / "config.json"


def _default_trackers() -> Dict[str, bool]:
    return {n: n in ("hip", "left_foot", "right_foot") for n in TRACKER_NAMES}


# what a profile (game / avatar / person) remembers — see profiles.py
PROFILE_KEYS = (
    "target", "user_height", "scale_multiplier", "yaw_offset", "offset_x", "offset_y",
    "offset_z", "trackers", "fingers", "finger_output", "send_head", "hmd_anchor", "vrchat_hmd",
    "filter_min_cutoff", "filter_beta", "lag_comp", "stabilizer", "leg_swap_fix",
    "posture_smoothing", "prediction_ms", "interp_delay_ms", "root_motion", "calibration",
)
MAX_PROFILES = 20
PROFILE_NAME_MAX = 40

NOHMD_WINDOWS = ("1280x720", "1600x900", "1920x1080", "2560x1440")
KEY_CHOICES = ["off"] + [f"F{i}" for i in range(1, 13)]


@dataclass
class Config:
    schema_version: int = SCHEMA_VERSION
    first_run_done: bool = False
    user_set: List[str] = field(default_factory=list)
    target: str = "auto"
    # --- camera ---
    camera: str = "0"            # camera index or path to a video file
    camera_name: str = ""        # OS name of that camera: finds it again if its index changes
    width: int = 640
    height: int = 480
    fps: int = 30
    fourcc: str = "auto"
    rotate: int = 0              # 0 / 90 / 180 / 270
    mirror: bool = False
    camera_vfov: float = 45.0    # vertical field of view, degrees (for root motion)
    camera_profiles: Dict[str, Any] = field(default_factory=dict)  # probe results per camera
    # --- extra cameras (USB webcams, phones as webcams) ---
    extra_cameras: List[Dict[str, Any]] = field(default_factory=list)  # [{source, name, rotate, mirror, enabled}]
    camera_rig: Dict[str, Any] = field(default_factory=dict)  # source -> {"R": 9 floats, "rms_cm"}
    multicam_fusion: bool = True
    rig_auto: bool = True        # calibrate / re-check the extra cameras in the background
    # --- detector ---
    model: str = "full"
    use_gpu: bool = False
    min_detection: float = 0.5
    min_presence: float = 0.5
    min_tracking: float = 0.5
    pc_profile: Optional[Dict[str, Any]] = None  # benchmark result (numbers only)
    # second network that sharpens the joints in the picture (rtmpose.py), off by default
    refiner: str = "off"
    refiner_gpu: bool = True
    refiner_strength: float = 0.7
    ensemble_mode: str = "legacy"
    ensemble_models: List[str] = field(default_factory=list)
    # --- outputs ---
    output_vrchat: bool = True       # always to this PC (127.0.0.1)
    vrchat_port: int = 9000
    send_head: bool = True       # head POSITION for VRChat's alignment (yaw: one-shot only)
    vrchat_hmd: bool = False     # headset pose FROM VRChat (OSCQuery): glue the body to it
    hmd_anchor: bool = True      # glue the body to the real SteamVR headset (driver mode)
    output_driver: bool = False  # own SteamVR driver (driver/ folder, C++)
    driver_port: int = 39580
    driver_hmd: bool = False     # "SteamVR without a headset": virtual HMD + hands from the camera
    nohmd_window: str = "1600x900"   # size of the virtual headset's window on the monitor
    nohmd_fov: int = 90              # its vertical field of view, degrees
    kb_controls: bool = True         # controller buttons from keyboard/mouse (no controllers)
    key_menu: str = "F8"             # -> B on the left hand (VRChat Quick Menu)
    key_triggers: str = "F9"         # -> both triggers (VRChat: confirm FBT calibration)
    key_pointer: str = "F10"         # toggle: right hand follows the mouse, click = trigger
    hmd_head_rotation: bool = False  # virtual HMD turns with your head (off: level, follows body)
    root_motion: bool = False    # estimate translation from the 2D image
    send_rate: int = 90          # packets/sec (with smoothing: fixed output rate)
    # --- person in view ---
    person_lock: bool = True     # follow only the chosen person
    max_people: int = 3          # how many people the detector looks for
    relock_single_s: float = 3.0  # 0 = never auto-switch to a lone stranger
    # --- smoothness & realism ---
    output_smoothing: bool = True  # interpolate between camera frames
    interp_delay_ms: int = 0     # 0 = automatic (≈ one camera frame)
    prediction_ms: int = 0       # extrapolate ahead to hide latency
    rotation_smoothing: bool = True
    joint_limits: bool = True    # knees bend only forwards
    leg_estimation: bool = True  # estimate legs the camera can't see
    pose_states: bool = True     # standing / sitting / lying with smooth transitions
    auto_floor: bool = True
    # --- trackers ---
    trackers: Dict[str, bool] = field(default_factory=_default_trackers)
    # --- fingers (off by default) ---
    fingers: bool = False
    finger_output: str = "preview"
    hand_every: int = 2          # run the hand model every N frames
    # --- accuracy ---
    bone_constraints: bool = True
    foot_lock: bool = True
    floor_clamp: bool = True
    # --- filtering ---
    lag_comp: bool = True        # smoothing with lag compensation (less delay, no overshoot)
    leg_swap_fix: bool = True    # undo left/right leg mix-ups of the model
    stabilizer: bool = True      # soft dead-band: trackers stand still when you do
    posture_smoothing: bool = True  # sitting / lying: calmer legs (more smoothing)
    filter_min_cutoff: float = 1.2
    filter_beta: float = 1.5
    root_min_cutoff: float = 0.4
    finger_min_cutoff: float = 2.0
    visibility_threshold: float = 0.5
    max_speed: float = 8.0       # m/s, larger jumps are clamped
    hold_time: float = 0.6       # seconds to keep last pose when tracking is lost
    lost_hold_s: float = 20.0    # ... then keep SENDING it (trackers stay in the game) this long
    steamvr_prediction: bool = True  # driver: trackers' velocity -> SteamVR predicts them
    # --- body / calibration ---
    user_height: float = 1.70
    scale_multiplier: float = 1.0
    yaw_offset: float = 0.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    offset_z: float = 0.0
    calibration: Optional[Dict[str, Any]] = None
    # --- profiles (games / avatars / people) ---
    profiles: Dict[str, Any] = field(default_factory=dict)   # name -> {key: value, "user_set": []}
    active_profile: str = ""
    # --- system ---
    high_priority: bool = True   # "above normal" process priority on Windows
    # --- ui ---
    show_preview: bool = True
    show_3d: bool = True

    # ------------------------------------------------------------------
    def validate(self) -> "Config":
        c = self
        c.schema_version = SCHEMA_VERSION
        c.first_run_done = bool(c.first_run_done)
        names = {f.name for f in fields(Config)}
        us = c.user_set if isinstance(c.user_set, (list, tuple)) else []
        c.user_set = sorted({str(k) for k in us if str(k) in names})[:200]
        c.target = c.target if c.target in TARGETS else "auto"
        c.camera = str(c.camera).strip()[:512] or "0"
        c.camera_name = _clean_cam_name(c.camera_name) if c.camera.isdigit() else ""
        c.width = _int(c.width, 160, 3840, 640)
        c.height = _int(c.height, 120, 2160, 480)
        c.fps = _int(c.fps, 5, 240, 30)
        c.fourcc = c.fourcc if c.fourcc in FOURCC_CHOICES else "auto"
        r = _int(c.rotate, 0, 270, 0)
        c.rotate = r if r in (0, 90, 180, 270) else 0
        c.mirror = bool(c.mirror)
        c.camera_vfov = _float(c.camera_vfov, 20.0, 120.0, 45.0)
        c.camera_profiles = _clean_profiles(c.camera_profiles)
        c.extra_cameras = _clean_extra_cameras(c.extra_cameras, c.camera)
        c.camera_rig = _clean_profiles(c.camera_rig)
        c.multicam_fusion = bool(c.multicam_fusion)
        c.rig_auto = bool(c.rig_auto)
        c.model = c.model if c.model in MODEL_VARIANTS else "full"
        c.use_gpu = bool(c.use_gpu)
        c.min_detection = _float(c.min_detection, 0.1, 0.95, 0.5)
        c.min_presence = _float(c.min_presence, 0.1, 0.95, 0.5)
        c.min_tracking = _float(c.min_tracking, 0.1, 0.95, 0.5)
        c.pc_profile = _clean_json(c.pc_profile) if isinstance(c.pc_profile, dict) else None
        c.refiner = c.refiner if c.refiner in REFINER_CHOICES else "off"
        c.ensemble_mode = (c.ensemble_mode if c.ensemble_mode in ("legacy", "manual", "auto")
                           else "legacy")
        c.ensemble_models = clean_selection(c.ensemble_models)
        c.refiner_gpu = bool(c.refiner_gpu)
        c.refiner_strength = _float(c.refiner_strength, 0.0, 1.0, 0.7)
        c.output_vrchat = bool(c.output_vrchat)
        c.vrchat_port = _int(c.vrchat_port, 1, 65535, 9000)
        c.send_head = bool(c.send_head)
        c.vrchat_hmd = bool(c.vrchat_hmd)
        c.hmd_anchor = bool(c.hmd_anchor)
        c.output_driver = bool(c.output_driver)
        c.driver_port = _int(c.driver_port, 1024, 65535, 39580)
        c.driver_hmd = bool(c.driver_hmd)
        c.hmd_head_rotation = bool(c.hmd_head_rotation)
        c.nohmd_window = c.nohmd_window if c.nohmd_window in NOHMD_WINDOWS else "1600x900"
        c.nohmd_fov = _int(c.nohmd_fov, 50, 120, 90)
        c.kb_controls = bool(c.kb_controls)
        keys = []
        for name, default in (("key_menu", "F8"), ("key_triggers", "F9"), ("key_pointer", "F10")):
            v = getattr(c, name)
            v = v if v in KEY_CHOICES else default
            if v != "off" and v in keys:  # the same key twice -> back to the default
                v = default if default not in keys else "off"
            keys.append(v)
            setattr(c, name, v)
        c.root_motion = bool(c.root_motion)
        c.send_rate = _int(c.send_rate, 10, 240, 90)
        c.person_lock = bool(c.person_lock)
        c.max_people = _int(c.max_people, 1, 5, 3)
        c.relock_single_s = _float(c.relock_single_s, 0.0, 30.0, 3.0)
        c.output_smoothing = bool(c.output_smoothing)
        c.interp_delay_ms = _int(c.interp_delay_ms, 0, 100, 0)
        c.prediction_ms = _int(c.prediction_ms, 0, 60, 0)
        c.rotation_smoothing = bool(c.rotation_smoothing)
        c.joint_limits = bool(c.joint_limits)
        c.leg_estimation = bool(c.leg_estimation)
        c.pose_states = bool(c.pose_states)
        c.auto_floor = bool(c.auto_floor)
        tr = c.trackers if isinstance(c.trackers, dict) else {}
        c.trackers = {n: bool(tr.get(n, False)) for n in TRACKER_NAMES}
        c.fingers = bool(c.fingers)
        c.finger_output = c.finger_output if c.finger_output in FINGER_OUTPUTS else "preview"
        c.hand_every = _int(c.hand_every, 1, 6, 2)
        c.bone_constraints = bool(c.bone_constraints)
        c.foot_lock = bool(c.foot_lock)
        c.floor_clamp = bool(c.floor_clamp)
        c.lag_comp = bool(c.lag_comp)
        c.leg_swap_fix = bool(c.leg_swap_fix)
        c.stabilizer = bool(c.stabilizer)
        c.posture_smoothing = bool(c.posture_smoothing)
        c.filter_min_cutoff = _float(c.filter_min_cutoff, 0.01, 20.0, 1.2)
        c.filter_beta = _float(c.filter_beta, 0.0, 50.0, 1.5)
        c.root_min_cutoff = _float(c.root_min_cutoff, 0.01, 10.0, 0.4)
        c.finger_min_cutoff = _float(c.finger_min_cutoff, 0.05, 20.0, 2.0)
        c.visibility_threshold = _float(c.visibility_threshold, 0.0, 0.99, 0.5)
        c.max_speed = _float(c.max_speed, 0.5, 50.0, 8.0)
        c.hold_time = _float(c.hold_time, 0.0, 10.0, 0.6)
        c.lost_hold_s = _float(c.lost_hold_s, 0.0, 120.0, 20.0)
        c.steamvr_prediction = bool(c.steamvr_prediction)
        c.user_height = _float(c.user_height, 0.8, 2.5, 1.70)
        c.scale_multiplier = _float(c.scale_multiplier, 0.3, 3.0, 1.0)
        c.yaw_offset = _float(c.yaw_offset, -180.0, 180.0, 0.0)
        c.offset_x = _float(c.offset_x, -10.0, 10.0, 0.0)
        c.offset_y = _float(c.offset_y, -10.0, 10.0, 0.0)
        c.offset_z = _float(c.offset_z, -10.0, 10.0, 0.0)
        if c.calibration is not None and not isinstance(c.calibration, dict):
            c.calibration = None
        c.profiles = _clean_named_profiles(c.profiles)
        c.active_profile = (clean_profile_name(c.active_profile)
                            if c.active_profile in c.profiles else "")
        c.high_priority = bool(c.high_priority)
        c.show_preview = bool(c.show_preview)
        c.show_3d = bool(c.show_3d)
        return c

    def enabled_trackers(self) -> List[str]:
        return [n for n in TRACKER_NAMES if self.trackers.get(n)]

    def is_user_set(self, key: str) -> bool:
        return key in self.user_set

    def mark_user_set(self, keys: Iterable[str]) -> None:
        names = {f.name for f in fields(Config)}
        s = set(self.user_set)
        s.update(k for k in keys if k in names)
        self.user_set = sorted(s)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_auto(cfg: Config, values: Dict[str, Any], force: bool = False) -> List[str]:
    """Apply automatically chosen values, skipping keys the user set by hand.

    Returns the keys that actually changed. With force=True (the user accepted
    a recommendation explicitly) user-set flags for those keys are cleared.
    """
    names = {f.name for f in fields(Config)}
    changed = []
    for k, v in values.items():
        if k not in names:
            continue
        if not force and k in cfg.user_set:
            continue
        if getattr(cfg, k) != v:
            setattr(cfg, k, v)
            changed.append(k)
        if force and k in cfg.user_set:
            cfg.user_set = [x for x in cfg.user_set if x != k]
    cfg.validate()
    return changed


# ---------------------------------------------------------------- helpers
def _float(v, lo, hi, default) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return min(max(f, lo), hi)


def _int(v, lo, hi, default) -> int:
    try:
        i = int(float(v))
    except (TypeError, ValueError, OverflowError):
        return default
    return min(max(i, lo), hi)


def _clean_json(v, depth: int = 0):
    """Keep only plain JSON scalars/containers, bounded in size and depth."""
    if depth > 5:
        return None
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return v if math.isfinite(float(v)) else None
    if isinstance(v, str):
        return v[:200]
    if isinstance(v, list):
        return [_clean_json(x, depth + 1) for x in v[:100]]
    if isinstance(v, dict):
        return {str(k)[:64]: _clean_json(x, depth + 1) for k, x in list(v.items())[:100]}
    return None


def _clean_cam_name(v) -> str:
    return "".join(ch for ch in str(v or "") if ch.isprintable()).strip()[:120]


def remap_camera_sources(cfg: "Config", mapping: Dict[str, str]) -> None:
    """Camera indices changed (devices plugged in/out): move per-camera data
    (probe profiles, rig calibration) to the new indices in one step, so that
    swapped indices (0<->1) don't overwrite each other."""
    mapping = {str(k): str(v) for k, v in mapping.items() if str(k) != str(v)}
    if not mapping:
        return
    for attr in ("camera_profiles", "camera_rig"):
        old = getattr(cfg, attr)
        new = {k: v for k, v in old.items() if k not in mapping and k not in mapping.values()}
        for src, dst in mapping.items():
            if src in old:
                new[dst] = old[src]
        setattr(cfg, attr, new)


def _clean_extra_cameras(v, main_source: str) -> List[Dict[str, Any]]:
    out, seen = [], {str(main_source)}
    if not isinstance(v, list):
        return out
    for item in v[:8]:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source", "")).strip()[:512]
        if not src or src in seen:
            continue
        seen.add(src)
        rot = _int(item.get("rotate", 0), 0, 270, 0)
        out.append({"source": src,
                    "name": _clean_cam_name(item.get("name", "")) if src.isdigit() else "",
                    "rotate": rot if rot in (0, 90, 180, 270) else 0,
                    "mirror": bool(item.get("mirror", False)),
                    "enabled": bool(item.get("enabled", True))})
        if len(out) >= 3:
            break
    return out


def _clean_profiles(v) -> Dict[str, Any]:
    if not isinstance(v, dict):
        return {}
    out = {}
    for k, p in list(v.items())[:16]:
        if isinstance(p, dict):
            out[str(k)[:512]] = _clean_json(p)
    return out


def clean_profile_name(v) -> str:
    return "".join(ch for ch in str(v or "") if ch.isprintable()).strip()[:PROFILE_NAME_MAX]


def _clean_named_profiles(v) -> Dict[str, Any]:
    """Profiles go through the same validation as the settings themselves: the
    values are put into a fresh Config, validated and read back."""
    if not isinstance(v, dict):
        return {}
    out: Dict[str, Any] = {}
    for name, prof in list(v.items())[:MAX_PROFILES]:
        name = clean_profile_name(name)
        if not name or not isinstance(prof, dict) or name in out:
            continue
        tmp = Config()
        keys = [k for k in PROFILE_KEYS if k in prof]
        for k in keys:
            setattr(tmp, k, prof[k])
        tmp.profiles = {}            # no nesting
        tmp.validate()
        clean = {k: copy.deepcopy(getattr(tmp, k)) for k in keys}
        if "calibration" in clean and clean["calibration"] is not None:
            clean["calibration"] = _clean_json(clean["calibration"])
        us = prof.get("user_set") if isinstance(prof.get("user_set"), list) else []
        clean["user_set"] = sorted({str(k) for k in us if str(k) in keys})
        out[name] = clean
    return out


def _migrate_vmt(cfg: "Config", data: dict) -> None:
    """v4.3: VMT support was removed — the Nexovara Pose driver does everything VMT did."""
    if data.get("target") == "steamvr":
        cfg.target = "steamvr_driver"
    if data.get("output_vmt") is True:
        cfg.output_driver = True
    if data.get("finger_output") == "vmt":
        cfg.finger_output = "driver"


def load_config(path: Optional[Path] = None) -> Config:
    path = path or config_path()
    cfg = Config()
    try:
        if path.is_file() and path.stat().st_size < 2_000_000:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                names = {f.name for f in fields(Config)}
                for k, v in data.items():
                    if k in names:
                        setattr(cfg, k, v)
                _migrate_vmt(cfg, data)
                if int(data.get("schema_version", 1) or 1) < 3 and "send_rate" not in \
                        (data.get("user_set") or []):
                    cfg.send_rate = 90  # v3: smoothed 90 Hz output
                if int(data.get("schema_version", 1) or 1) < 2:
                    # v1 configs: everything the user had was set by hand, but the
                    # camera mode was never probed -> let auto-tuning pick it.
                    cfg.first_run_done = False
    except (OSError, ValueError, TypeError) as e:
        log.warning("Config load failed (%s), using defaults", e)
        cfg = Config()
    return cfg.validate()


def save_config(cfg: Config, path: Optional[Path] = None) -> None:
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(cfg.validate().to_dict(), indent=2, ensure_ascii=False)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)  # atomic: never leaves a half-written config
    except OSError as e:
        log.error("Config save failed: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass
