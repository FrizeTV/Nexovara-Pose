"""Tracking pipeline running in its own thread.

camera (thread) -> newest frame -> MediaPipe (all people) -> person lock ->
(+ hands) -> sanitize -> One Euro filter -> calibration -> knee limits ->
bone lengths -> realism (pose states, leg estimation, auto floor) -> feet ->
(HMD anchor) -> trackers (+ rotation smoothing) -> keyframe ->
sender thread (interpolation, 90 Hz) -> OSC ; snapshot -> GUI

Startup automation (in this thread, so the UI stays responsive):
* target "auto"  -> detect the running game / SteamVR and apply its preset
* unknown camera -> probe its real modes once and use the best one
Both only touch settings the user has not changed by hand, and are reported
to the UI as "auto_applied" events so it can store them.
"""
from __future__ import annotations

import copy
import importlib.util
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import steamvr, system
from .camera import (CameraSource, best_from_profile, camera_label, enumerate_devices,
                     find_device, is_video_file, probe_camera, profile_for)
from .config import Config, apply_auto, remap_camera_sources
from .filters import LegSwapGuard, OneEuroFilter, PoseFilter, Stabilizer
from .guidance import Guide, TrackerConfidence, advice
from .skeleton import HeadModel
from .mathutil import QuatOneEuro, basis_from_right_up, rot_y, yaw_of
from .multicam import (AGREE_LIMIT_M, AutoRig, AuxCameras, RigCalibrator, agreement, fuse,
                       rig_rotation)
from .person import PersonLock
from .hands import FingerRange
from .pose import (ModelManifest, PoseEstimator, PoseResult, ensure_model, make_estimator,
                   prepare_rgb)
from .presets import PRESETS, detect_target
from .recording import (Recorder, ReplayEstimator, ReplaySource, is_recording, load_recording,
                        new_recording_path)
from .realism import Realism, fix_knees
from .hmd import HmdState
from .sender import Keyframe, OutputSender
from .skeleton import (L_HIP, L_SH, R_HIP, R_SH, Calibration, FootRotationHold, FootStabilizer,
                       Tracker,
                       apply_bone_lengths, compute_calibration, compute_hand, compute_head,
                       compute_trackers, estimate_root, mp_to_unity, uncalibrated_transform)

log = logging.getLogger(__name__)

MAX_INFER_WIDTH = 1280     # full frame kept up to this width (hands crop from it)
POSITION_LIMIT_M = 100.0   # sanity bound for anything we send
# a frame is handed to us ~one frame time after the sensor exposed it (exposure,
# USB transfer, decoding); the headset pose matching it is from that moment
CAPTURE_LATENCY_S = 0.03
CAMERA_KEYS = ("width", "height", "fps", "fourcc")
ROT_MIN_CUTOFF, ROT_BETA = 2.0, 0.6
# orientation smoothing per tracker: feet/hips turn slowly but their camera
# rotation is the noisiest (short bones, depth) -> calmer filters
ROT_PARAMS = {"left_foot": (1.0, 0.35), "right_foot": (1.0, 0.35),
              "hip": (1.5, 0.5), "chest": (1.5, 0.5)}
# dead-band size per tracker (x Stabilizer radius / angle)
STAB_SCALE = {"left_foot": 1.4, "right_foot": 1.4, "left_knee": 1.2, "right_knee": 1.2,
              "left_elbow": 1.2, "right_elbow": 1.2}
# sitting / lying: wider dead-band for the legs (up to x1.5)
POSTURE_STAB = ("left_foot", "right_foot", "left_knee", "right_knee", "hip")


_HAS_OPENVR = importlib.util.find_spec("openvr") is not None


def body_is_coherent(pts: np.ndarray, trackers: Dict[str, Tracker], head: Tracker) -> bool:
    """A frame where the model produced a body that can't be human (a tracker
    metres away from the hips — e.g. half the body out of the picture) is
    dropped: the previous pose is held instead of throwing trackers across the room."""
    hip = (pts[L_HIP] + pts[R_HIP]) / 2
    scale = max(float(np.linalg.norm((pts[L_SH] + pts[R_SH]) / 2 - hip)), 0.2)
    lim = max(2.0, 5.0 * scale)
    if float(np.linalg.norm(head.pos - hip)) > max(1.5, 3.0 * scale):
        return False
    return all(float(np.linalg.norm(t.pos - hip)) <= lim for t in trackers.values())


def circular_mean_deg(values: Sequence[float]) -> float:
    a = np.radians(np.asarray(values, dtype=np.float64))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def _wants_hmd(cfg: Config) -> bool:
    """Glue the body to a REAL headset (Nexovara Pose driver next to a real headset).
    Never in "without a headset" mode — the HMD there is our own."""
    return bool(cfg.hmd_anchor and cfg.output_driver and not cfg.driver_hmd)


class Engine:
    def __init__(self, cfg: Config, models_dir: Path, model_search_dirs: Sequence[Path] = (),
                 data_dir: Optional[Path] = None):
        self._cfg = copy.deepcopy(cfg)
        self._cfg_lock = threading.Lock()
        self._cfg_dirty = True
        self.models_dir = Path(models_dir)
        self.model_search_dirs = list(model_search_dirs)
        self.manifest = ModelManifest(Path(data_dir) / "models.json" if data_dir else None)
        self.data_dir = Path(data_dir) if data_dir else None
        self._rec_request: Optional[str] = None     # "start" / "stop"
        self._events: "queue.Queue[tuple]" = queue.Queue()
        self._snap_lock = threading.Lock()
        self._snapshot: Optional[Dict[str, Any]] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.hmd_state = HmdState()           # real headset pose (driver feedback / openvr)
        self.sender = OutputSender(self.hmd_state)
        self._calib_lock = threading.Lock()
        self._calib_needed = 0
        self._calib_samples: List[np.ndarray] = []
        self._calib_yaws: Dict[str, List[float]] = {}
        self._calib_deadline = 0.0
        self._person_lock = PersonLock()
        self._pick_lock = threading.Lock()
        self._align_request = False
        self._pick: Optional[Tuple[float, float]] = None
        self._relock = False
        self._rig_request: Optional[float] = None
        self._rig: Optional[RigCalibrator] = None
        self._auto_rig = AutoRig()

    # ---------------- public API (GUI thread) ----------------
    @property
    def paused(self) -> bool:
        return self.sender.paused

    @paused.setter
    def paused(self, v: bool) -> None:
        self.sender.paused = bool(v)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="tracker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_config(self, cfg: Config) -> None:
        with self._cfg_lock:
            self._cfg = copy.deepcopy(cfg)
            self._cfg_dirty = True

    def request_calibration(self, frames: int = 30, timeout: float = 10.0) -> None:
        with self._calib_lock:
            self._calib_needed = max(5, int(frames))
            self._calib_samples = []
            self._calib_yaws = {}
            self._calib_deadline = time.monotonic() + timeout

    def request_vrchat_align(self) -> None:
        """VRChat OSC: turn VRChat's tracking space so that your body faces where
        it faces now (stand facing the camera, look straight)."""
        self._align_request = True

    def pick_person(self, x: float, y: float) -> None:
        """Lock onto the person at normalized preview coordinates (x, y)."""
        with self._pick_lock:
            self._pick = (float(x), float(y))

    def request_rig_calibration(self, seconds: float = 8.0) -> None:
        """Measure how the extra cameras are rotated relative to the main one."""
        with self._pick_lock:
            self._rig_request = float(seconds)

    def start_recording(self) -> None:
        """Record the numbers of your pose (no video) until stop_recording()."""
        with self._pick_lock:
            self._rec_request = "start"

    def stop_recording(self) -> None:
        with self._pick_lock:
            self._rec_request = "stop"

    def relock_person(self) -> None:
        """Forget who we follow; the biggest/central person will be taken next."""
        with self._pick_lock:
            self._relock = True

    def poll_events(self) -> List[tuple]:
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    def snapshot(self) -> Optional[Dict[str, Any]]:
        with self._snap_lock:
            return self._snapshot

    # ---------------- worker ----------------
    def _emit(self, kind: str, payload: Any = None) -> None:
        self._events.put((kind, payload))

    def _take_cfg(self) -> Optional[Config]:
        with self._cfg_lock:
            if not self._cfg_dirty:
                return None
            self._cfg_dirty = False
            return copy.deepcopy(self._cfg)

    def _run(self) -> None:
        system.enable_high_res_timer()
        try:
            self._run_inner()
        except Exception as e:
            log.exception("Tracking stopped with an error")
            self._emit("error", str(e) or e.__class__.__name__)
        finally:
            self.sender.stop()
            system.disable_high_res_timer()
            self._emit("stopped")

    @staticmethod
    def _prepare(frame: np.ndarray, cfg: Config) -> np.ndarray:
        if cfg.rotate == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif cfg.rotate == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif cfg.rotate == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if cfg.mirror:
            frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        if w > MAX_INFER_WIDTH:
            s = MAX_INFER_WIDTH / w
            frame = cv2.resize(frame, (MAX_INFER_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
        return frame

    def _auto_apply(self, cfg: Config, values: Dict[str, Any], why: str) -> None:
        changed = apply_auto(cfg, values)
        if changed:
            self._emit("auto_applied", {"values": {k: getattr(cfg, k) for k in changed},
                                        "why": why})

    def _resolve_cameras(self, cfg: Config) -> None:
        """Find the configured cameras by NAME in the current OS list.

        Camera numbers change when devices are plugged in/out (the log of
        "Не удалось открыть камеру #3"); names don't. Moved cameras get their
        new number (saved), a missing main camera is replaced by a connected one
        for this run, missing extra cameras are skipped. Learns names of
        cameras configured before names were stored."""
        devs = enumerate_devices(0)
        if not devs:
            # names unknown, or the OS reports no cameras (then opening the camera
            # still gets its chance and reports a clear error itself)
            return
        upd: Dict[str, Any] = {"mapping": {}, "extras": [], "notes": []}
        notes = upd["notes"]
        if not is_video_file(cfg.camera):
            dev, st = find_device(devs, cfg.camera, cfg.camera_name)
            if st in ("ok", "moved"):
                if st == "moved":
                    notes.append(f"камера «{dev.name}» теперь под номером {dev.index}")
                    upd["mapping"][cfg.camera] = str(dev.index)
                upd["camera"], upd["camera_name"], upd["persist"] = str(dev.index), dev.name, True
            elif st == "missing":
                used = {s.get("source") for s in cfg.extra_cameras}
                d = ([d for d in devs if str(d.index) not in used] or devs)[0]
                notes.append(f"камера «{camera_label(cfg.camera, cfg.camera_name)}» не найдена — "
                             f"использую «{d.label()}»")
                log.warning("camera %s (%s) not found, using %s", cfg.camera, cfg.camera_name,
                            d.label())
                # a remembered camera that is just unplugged stays chosen in the settings
                upd["camera"], upd["camera_name"] = str(d.index), d.name
                upd["persist"] = not cfg.camera_name
            if "camera" in upd:
                cfg.camera, cfg.camera_name = upd["camera"], upd["camera_name"]
        keep = []
        for spec in cfg.extra_cameras:
            spec = dict(spec)
            src = str(spec.get("source", ""))
            dev, st = find_device(devs, src, spec.get("name", ""))
            if st == "missing":
                if spec.get("enabled", True):
                    notes.append(f"доп. камера «{camera_label(src, spec.get('name', ''))}» "
                                 "не найдена — пропускаю")
                spec["enabled"] = False      # this run only
                spec["missing"] = True
            elif dev is not None:
                if st == "moved":
                    notes.append(f"доп. камера «{dev.name}» теперь под номером {dev.index}")
                    upd["mapping"][src] = str(dev.index)
                spec["source"], spec["name"] = str(dev.index), dev.name
            if spec.get("enabled", True) and spec["source"] == cfg.camera:
                spec["enabled"] = False      # same device as the main camera
                spec["missing"] = True
            keep.append(spec)
            upd["extras"].append({k: v for k, v in spec.items() if k != "missing"}
                                 if not spec.get("missing") else None)
        cfg.extra_cameras = keep
        if upd["mapping"]:
            remap_camera_sources(cfg, upd["mapping"])
        self._emit("cameras_resolved", upd)
        if notes:
            self._emit("status", "Камеры: " + "; ".join(notes))

    def _startup_automation(self, cfg: Config) -> None:
        # 0) find the cameras by name (numbers change when devices are plugged in/out)
        self._resolve_cameras(cfg)
        # 1) game / app preset
        if cfg.target == "auto":
            tid, reason = detect_target()
            self._emit("target", {"id": tid, "reason": reason})
            self._auto_apply(cfg, PRESETS[tid].values, f"режим «{PRESETS[tid].title}»: {reason}")
        elif cfg.target in PRESETS:
            self._auto_apply(cfg, PRESETS[cfg.target].values, PRESETS[cfg.target].title)
        # 2) extra cameras: measure their modes once (they then use the best one)
        for spec in cfg.extra_cameras:
            src = spec.get("source", "")
            if not spec.get("enabled", True) or is_video_file(src) or \
                    profile_for(cfg.camera_profiles, src, spec.get("name", "")):
                continue
            self._emit("status", f"Проверяю дополнительную камеру "
                                 f"«{camera_label(src, spec.get('name', ''))}» (≈10 с)…")
            try:
                prof = probe_camera(int(src), None, self._stop.is_set)
            except Exception as e:
                log.warning("extra camera %s probe failed: %s", src, e)
                self._emit("status", f"Доп. камера: {e}")
                continue
            if self._stop.is_set():
                return
            cfg.camera_profiles[src] = prof
            self._emit("camera_profile", {"camera": src, "profile": prof})
        # 3) main camera mode
        if is_video_file(cfg.camera):
            return
        if all(k in cfg.user_set for k in CAMERA_KEYS):
            return
        profile = profile_for(cfg.camera_profiles, cfg.camera, cfg.camera_name)
        if profile is None:
            label = camera_label(cfg.camera, cfg.camera_name)
            self._emit("status", f"Определяю возможности камеры «{label}» (≈10 с)…")

            def prog(p, msg):
                self._emit("status", f"Камера «{label}»: {msg} {int(p * 100)}%")
            try:
                profile = probe_camera(int(cfg.camera), prog, self._stop.is_set)
            except Exception as e:
                log.warning("camera probe failed: %s", e)
                return
            if self._stop.is_set():
                return
            cfg.camera_profiles[cfg.camera] = profile
            self._emit("camera_profile", {"camera": cfg.camera, "profile": profile})
        best = best_from_profile(profile, prefer_hd=cfg.fingers)
        if best is not None:
            self._auto_apply(cfg, {"width": best.width, "height": best.height,
                                   "fps": int(round(min(best.fps, 120))),
                                   "fourcc": best.fourcc if best.fourcc in ("MJPG", "YUY2")
                                   else "auto"},
                             f"камера «{camera_label(cfg.camera, cfg.camera_name)}»: "
                             f"{best.label()}")

    def _load_model(self, variant: str):
        last = [-1]

        def progress(p: float) -> None:
            pct = int(p * 100)
            if pct // 5 != last[0] // 5:
                last[0] = pct
                self._emit("status", f"Скачивание модели {variant}: {pct}%")
        return ensure_model(variant, self.models_dir, self.model_search_dirs, progress,
                            self.manifest)

    def _choose_person(self, people, rgb, cfg: Config, now: float):
        with self._pick_lock:
            pick, self._pick = self._pick, None
            relock, self._relock = self._relock, False
        if relock:
            self._person_lock.reset()
        if pick is not None:
            self._person_lock.pick(*pick)
        if not cfg.person_lock:
            return (people[0] if people else None), {"people": len(people), "locked": False,
                                                     "status": "", "others": [
                                                         p.image for p in people[1:]]}
        return self._person_lock.select(people, rgb, now, cfg.relock_single_s)

    def _run_inner(self) -> None:
        cfg = self._take_cfg() or copy.deepcopy(self._cfg)
        system.set_priority(cfg.high_priority)
        self._startup_automation(cfg)
        if self._stop.is_set():
            return

        replay = is_recording(cfg.camera)
        num_poses = cfg.max_people if cfg.person_lock else 1
        if replay:
            # a recorded session: no camera, no neural network — the recorded
            # pose numbers go through exactly the same pipeline
            rec = load_recording(Path(cfg.camera))
            cam = ReplaySource(rec)
            est = ReplayEstimator(cam)
            aux = AuxCameras([], {}, None, 1, cfg.relock_single_s, False)
        else:
            self._emit("status", "Подготовка нейросети…")
            model_path = self._load_model(cfg.model)
            est = make_estimator(model_path, cfg.min_detection, cfg.min_presence,
                                 cfg.min_tracking, cfg.use_gpu, num_poses=num_poses,
                                 cls=PoseEstimator)
            aux = AuxCameras(cfg.extra_cameras, cfg.camera_profiles, model_path, num_poses,
                             cfg.relock_single_s, cfg.person_lock)
            cam = CameraSource(cfg.camera, cfg.width, cfg.height, cfg.fps, cfg.fourcc,
                               name=cfg.camera_name)
        hands = None
        recorder: Optional[Recorder] = None
        refiner = None
        ensemble = None
        hmd: Optional[steamvr.HmdReader] = None
        hmd_box: Optional[list] = None   # result of a background headset connect
        try:
            if not replay:
                if cfg.ensemble_mode == "legacy":
                    refiner = self._make_refiner(cfg)
                else:
                    ensemble = self._make_ensemble(cfg)
            # a busy / just re-plugged camera is waited for instead of stopping with an error
            cam.start() if replay else cam.start(wait=True)
            aux.start()
            self.sender.start()
            self._emit("camera_info", {"size": cam.actual_size, "fourcc": cam.actual_fourcc,
                                       "camera": cfg.camera, "name": cfg.camera_name})
            self._emit("status", f"Трекинг запущен: «{camera_label(cfg.camera, cfg.camera_name)}»"
                                 f" · {est.delegate}, модель {cfg.model}")

            filt = PoseFilter(cfg.filter_min_cutoff, cfg.filter_beta, cfg.lag_comp)
            swap_guard = LegSwapGuard()
            stab = Stabilizer()
            root_filt = OneEuroFilter(cfg.root_min_cutoff, 0.3)
            curl_filt = {s: OneEuroFilter(cfg.finger_min_cutoff, 0.5) for s in ("left", "right")}
            finger_range = {s: FingerRange() for s in ("left", "right")}
            rot_filt: Dict[str, QuatOneEuro] = {}
            feet = FootStabilizer()
            foot_hold = FootRotationHold()
            realism = Realism()
            confidence = TrackerConfidence()
            head_model = HeadModel()
            guide = Guide()
            brightness: Optional[float] = None
            conf_values: Dict[str, float] = {}
            hints: List[str] = []
            calibration: Optional[Calibration] = None
            last_raw: Optional[np.ndarray] = None
            last_ts = 0.0
            last_pts_ts = 0.0
            last_pose_t = 0.0
            last_rots: Dict[str, np.ndarray] = {}
            last_trackers: List[Tracker] = []
            last_head: Optional[Tracker] = None
            hand_state: Dict[str, Dict[str, Any]] = {"left": {}, "right": {}}
            last_seq = 0
            frame_no = 0
            proc_frames, proc_t0, proc_fps = 0, time.monotonic(), 0.0
            hmd_retry_t = time.monotonic()
            warned_rm = False
            rinfo = None
            person_info: Dict[str, Any] = {}
            was_lost = False
            last_capture_t: Optional[float] = None

            pending: Optional[Config] = cfg  # apply the initial settings on the first pass
            while not self._stop.is_set():
                new_cfg = self._take_cfg() or pending
                pending = None
                if new_cfg is not None:
                    keep_profiles = cfg.camera_profiles
                    # cameras in use stay as resolved at start (changing them restarts)
                    running = (cfg.camera, cfg.camera_name, cfg.extra_cameras)
                    cfg = new_cfg
                    cfg.camera_profiles = cfg.camera_profiles or keep_profiles
                    cfg.camera, cfg.camera_name, cfg.extra_cameras = running
                    filt.configure(cfg.filter_min_cutoff, cfg.filter_beta, cfg.lag_comp)
                    root_filt.configure(cfg.root_min_cutoff, 0.3)
                    for f in curl_filt.values():
                        f.configure(cfg.finger_min_cutoff, 0.5)
                    self.sender.configure(cfg)
                    calibration = (Calibration.from_dict(cfg.calibration)
                                   if cfg.calibration else None)
                    if calibration is not None and calibration.root_motion != cfg.root_motion:
                        calibration = None
                        if not warned_rm:
                            warned_rm = True
                            self._emit("status", "Режим перемещения изменён — откалибруйтесь "
                                                 "заново (F6)")
                    feet.reset()
                    realism.reset(keep_floor=True)
                    want_hmd = _wants_hmd(cfg)
                    if want_hmd and hmd is None:
                        hmd, msg = steamvr.try_connect()
                        self._emit("status", msg)
                        hmd_retry_t = time.monotonic()
                    elif not want_hmd and hmd is not None:
                        hmd.close()
                        hmd = None
                    if cfg.fingers and hands is None and not replay:
                        hands = self._make_hands()
                    elif not cfg.fingers and hands is not None:
                        hands.close()
                        hands = None

                item = cam.get_latest(last_seq, timeout=0.5)
                cam_lost = bool(getattr(cam, "lost", False))
                if cam_lost != was_lost:
                    was_lost = cam_lost
                    label = camera_label(cfg.camera, cfg.camera_name)
                    why = getattr(cam, "open_error", None) or "отключена"
                    self._emit("status", f"Камера «{label}»: {why}. Жду её — трекинг продолжится "
                                         "сам" if cam_lost else
                                         f"Камера «{label}» снова работает")
                if item is None:
                    if cam.error:
                        raise RuntimeError(cam.error)
                    if cam_lost:
                        self._publish_lost(cfg, cam, est)
                    continue
                last_seq, frame, ts = item
                if replay:
                    est.seq = last_seq            # the recorded pose of exactly this frame
                else:
                    frame = self._prepare(frame, cfg)
                h, w = frame.shape[:2]
                frame_no += 1

                t0 = time.monotonic()
                rgb = prepare_rgb(frame)          # small RGB copy for the pose model
                people = est.detect(rgb, int(ts * 1000))
                res, person_info = self._choose_person(people, rgb, cfg, ts)
                if refiner is not None and res is not None:
                    res = refiner(res, rgb, cfg.refiner_strength)   # sharper joints (RTMPose)
                if ensemble is not None:
                    camera_fps = cam.measured_fps or cfg.fps
                    budget_ms = 750.0 / max(5.0, min(cfg.fps, camera_fps))
                    if hands is not None:
                        budget_ms *= 0.8
                    res = ensemble.apply(res, rgb, ts, cfg.refiner_strength, budget_ms,
                                         (time.monotonic() - t0) * 1000)
                res, multi_info = self._multicam(cfg, res, aux, ts,
                                                 int(person_info.get("people", 1) or 1))
                recorder = self._recording_step(recorder, res, ts, w / max(h, 1), cfg, cam,
                                                replay)
                hand_res = None
                if (hands is not None and res is not None and res.image is not None
                        and frame_no % cfg.hand_every == 0):
                    hand_res = hands.process(frame, res.image, res.visibility, int(ts * 1000))
                now = time.monotonic()
                infer_ms = (now - t0) * 1000.0

                pts = None
                if res is not None:
                    raw = mp_to_unity(res.world)
                    vis = res.visibility
                    if cfg.leg_swap_fix:
                        raw, vis = swap_guard(raw, vis)   # left/right legs mixed up?
                    if cfg.root_motion and res.image is not None:
                        root = estimate_root(res.image, res.world, w / h, cfg.camera_vfov)
                        root_f = root_filt(root, ts) if root is not None else root_filt.value
                        if root_f is not None:
                            raw = raw + root_f
                    if last_raw is not None:
                        # occluded joints: hold the last good position
                        low = vis < cfg.visibility_threshold
                        raw[low] = last_raw[low]
                        # reject teleporting joints (limit speed)
                        dt = max(ts - last_ts, 1e-3)
                        delta = raw - last_raw
                        dist = np.linalg.norm(delta, axis=1)
                        maxd = cfg.max_speed * dt
                        over = dist > maxd
                        if np.any(over):
                            raw[over] = last_raw[over] + delta[over] * (maxd / dist[over])[:, None]
                    last_raw, last_ts = raw, ts
                    # sitting / lying (from the previous frame): calmer legs
                    if cfg.posture_smoothing and rinfo is not None:
                        filt.set_posture(rinfo.weights.get("sit", 0.0),
                                         rinfo.weights.get("lie", 0.0))
                    else:
                        filt.set_posture(0.0, 0.0)
                    smooth = filt(raw, ts, vis)

                    new_cal = self._calibration_step(smooth, cfg, hmd, ts - CAPTURE_LATENCY_S)
                    if new_cal is not None:
                        calibration = new_cal
                        self._person_lock.resign()  # follow the person who calibrated
                        realism.reset()
                        feet.reset()

                    dt_pts = ts - last_pts_ts if last_pts_ts else 1 / 30
                    last_pts_ts = ts
                    if calibration is not None:
                        pts = calibration.apply(smooth)
                        if cfg.joint_limits:
                            pts = fix_knees(pts)
                        if cfg.bone_constraints and calibration.lengths:
                            pts = apply_bone_lengths(pts, calibration.lengths)
                        if calibration.lengths and (cfg.pose_states or cfg.leg_estimation
                                                    or cfg.auto_floor):
                            pts, rinfo = realism.process(
                                pts, vis, dt_pts, calibration.lengths,
                                states=cfg.pose_states, legs=cfg.leg_estimation,
                                auto_floor=cfg.auto_floor)
                        else:
                            rinfo = None
                        standing = rinfo is None or (rinfo.weights["stand"] > 0.7
                                                     and max(rinfo.legs_estimated) < 0.5)
                        pts = feet.apply(pts, dt_pts, cfg.foot_lock and standing, cfg.floor_clamp)
                    else:
                        pts = uncalibrated_transform(smooth)
                        rinfo = None
                    pts = cfg.scale_multiplier * (pts @ rot_y(cfg.yaw_offset).T)
                    pts = pts + np.array([cfg.offset_x, cfg.offset_y, cfg.offset_z])

                    # gluing to a real headset happens in the output thread (90 Hz,
                    # newest headset pose, filtered offset) — see hmd.BodyAnchor

                    tr = compute_trackers(pts, cfg.enabled_trackers())
                    head = head_model(compute_head(pts), pts, vis,
                                      res.image if res is not None else None, dt_pts)
                    ok = bool(np.all(np.isfinite(pts))) and float(np.abs(pts).max()) < POSITION_LIMIT_M
                    ok = ok and body_is_coherent(pts, tr, head)
                    if ok:
                        all_t = [*tr.values(), head]
                        if cfg.fingers or (cfg.output_driver and cfg.driver_hmd):
                            for side in ("left", "right"):
                                ht = compute_hand(pts, side)
                                hand_state[side]["tracker"] = ht
                                all_t.append(ht)
                        else:  # hands not wanted any more -> never send stale ones
                            for side in ("left", "right"):
                                hand_state[side].pop("tracker", None)
                        for t in all_t:
                            if t.rot is None:
                                t.rot = last_rots.get(t.name)
                            else:
                                if cfg.rotation_smoothing:
                                    f = rot_filt.get(t.name)
                                    if f is None:
                                        f = rot_filt[t.name] = QuatOneEuro(
                                            *ROT_PARAMS.get(t.name, (ROT_MIN_CUTOFF, ROT_BETA)))
                                    t.rot = f(t.rot, ts)
                                last_rots[t.name] = t.rot
                            if (cfg.foot_lock and calibration is not None
                                    and t.name in ("left_foot", "right_foot")):
                                t.rot = foot_hold(t.name, t.rot,
                                                  feet.locked[0 if t.name == "left_foot" else 1],
                                                  dt_pts)
                            if cfg.stabilizer and not t.name.endswith("_hand"):
                                k = STAB_SCALE.get(t.name, 1.0)
                                if cfg.posture_smoothing and rinfo is not None and \
                                        t.name in POSTURE_STAB:
                                    k *= 1.0 + 0.5 * (1.0 - rinfo.weights.get("stand", 1.0))
                                t.pos = stab.position(t.name, t.pos, k)
                                if t.rot is not None:
                                    t.rot = stab.rotation(t.name, t.rot, k)
                        last_trackers = [tr[n] for n in cfg.enabled_trackers() if n in tr]
                        if self._align_request:
                            self._align_request = False
                            hipR = basis_from_right_up(pts[R_HIP] - pts[L_HIP],
                                                       (pts[L_SH] + pts[R_SH]) / 2
                                                       - (pts[L_HIP] + pts[R_HIP]) / 2)
                            self.sender.request_yaw_align(
                                yaw_of(hipR[:, 2]) if hipR is not None else 0.0)
                        last_head = head
                        last_pose_t = now
                        last_capture_t = ts - CAPTURE_LATENCY_S
                    else:
                        pts = None

                # finger curls: filter, hold for hold_time, then relax
                if hand_res is not None:
                    for side, hr in hand_res.items():
                        st = hand_state[side]
                        if hr is not None:
                            st["curls"] = curl_filt[side](finger_range[side](hr.curls, now),
                                                          ts)
                            st["image"] = hr.image
                            st["t"] = now
                for side in ("left", "right"):
                    st = hand_state[side]
                    if st.get("t") and now - st["t"] > cfg.hold_time:
                        st.pop("curls", None)
                        st.pop("image", None)
                        st.pop("t", None)
                        curl_filt[side].reset()

                lost = (now - last_pose_t) > cfg.hold_time
                if lost and last_raw is not None and res is None:
                    filt.reset()
                    swap_guard.reset()
                    stab.reset()
                    root_filt.reset()
                    feet.reset()
                    foot_hold.reset()
                    realism.reset(keep_floor=True)
                    for f in rot_filt.values():
                        f.reset()
                    last_raw = None

                if pts is not None and last_trackers:
                    hand_out = {}  # each output decides whether it uses hands
                    for side in ("left", "right"):
                        st = hand_state[side]
                        if "tracker" in st:
                            hand_out[side] = (st["tracker"], st.get("curls"))
                    self.sender.push(Keyframe(now, list(last_trackers), last_head, hand_out,
                                              calibration.hmd_yaw if calibration else None,
                                              calibration.vrc_yaw if calibration else None,
                                              last_capture_t))
                # openvr fallback for the headset pose (the driver reports it itself)
                if hmd is not None:
                    pose = hmd.get_pose()
                    if pose is not None:
                        self.hmd_state.update(pose[0], pose[1], "openvr")

                # SteamVR started after Nexovara Pose: connect to the headset later. Checked in
                # a helper thread — looking through the process list takes a few dozen
                # ms on Windows and must not stall tracking.
                if hmd_box:
                    hmd, msg = hmd_box.pop()
                    hmd_box = None
                    if hmd is not None and not _wants_hmd(cfg):
                        hmd.close()
                        hmd = None
                    elif hmd is not None:
                        self._emit("status", msg)
                if (hmd is None and hmd_box is None and _wants_hmd(cfg)
                        and now - hmd_retry_t > 10.0 and _HAS_OPENVR):
                    hmd_retry_t = now
                    hmd_box = []

                    def _try(box=hmd_box):
                        box.append(steamvr.try_connect() if steamvr.is_running()
                                   else (None, ""))
                    threading.Thread(target=_try, daemon=True, name="hmd-connect").start()

                if frame_no % 15 == 0:
                    if ensemble is not None:
                        self.refiner_info = ensemble.describe()
                    elif refiner is not None:
                        self.refiner_info = refiner.describe()
                # how sure we are about each tracker + advice how to stand
                if frame_no % 10 == 1 and not replay:     # replay frames are a plain backdrop
                    brightness = float(rgb[::8, ::8].mean())
                conf_names = list(cfg.enabled_trackers()) + ["head"]
                if any("tracker" in hand_state[s_] for s_ in ("left", "right")):
                    conf_names += ["left_hand", "right_hand"]
                if res is not None:
                    conf_values = confidence.update(
                        conf_names, res.visibility, res.image,
                        rinfo.statuses if rinfo is not None else None,
                        int(multi_info.get("fused") or 1), ts)
                elif lost:
                    confidence.reset()
                    conf_values = {}
                hints = guide.update(advice(
                    res.image if res is not None else None,
                    res.visibility if res is not None else None, w / max(h, 1), brightness,
                    cam.measured_fps, not is_video_file(cfg.camera), calibration is not None,
                    conf_values), now)

                proc_frames += 1
                if now - proc_t0 >= 1.0:
                    proc_fps = proc_frames / (now - proc_t0)
                    proc_frames, proc_t0 = 0, now

                self._publish(cfg, rgb, w, h, res, pts, last_trackers, last_head, lost,
                              calibration, hand_state, last_seq, ts, cam, est, proc_fps,
                              infer_ms, hmd, rinfo, person_info, multi_info,
                              {"confidence": conf_values, "hints": hints})
        finally:
            if recorder is not None:          # stopped while recording: keep what we have
                self._finish_recording(recorder)
            if refiner is not None:
                refiner.close()
            if ensemble is not None:
                ensemble.close()
            self.refiner_info = ""
            cam.stop()
            aux.stop()
            est.close()
            if hands is not None:
                hands.close()
            self.sender.stop()
            if hmd is not None:
                hmd.close()
            for late, _msg in (hmd_box or []):   # a connect that finished just now
                if late is not None:
                    late.close()

    # ---------------- second network (RTMPose), optional
    refiner_info: str = ""

    def _make_refiner(self, cfg: Config):
        """RTMPose add-on; any problem -> a message and tracking WITHOUT it."""
        if cfg.refiner == "off":
            return None
        from . import rtmpose
        if not rtmpose.onnxruntime_available():
            self._emit("status", "Уточнение RTMPose недоступно: не установлен onnxruntime "
                                 "(запустите setup.bat) — работаю только с MediaPipe")
            return None
        try:
            path = rtmpose.ensure_refiner_model(cfg.refiner, self.models_dir,
                                                self.model_search_dirs,
                                                manifest=self.manifest, download=False)
            ref = rtmpose.Refiner(cfg.refiner, path, cfg.refiner_gpu)
        except Exception as e:
            log.warning("RTMPose unavailable: %s", e)
            self._emit("status", f"Уточнение RTMPose недоступно ({e}) — работаю только с "
                                 "MediaPipe")
            return None
        self._emit("status", f"RTMPose включён: {ref.device}")
        return ref

    def _make_ensemble(self, cfg: Config):
        import os
        from . import rtmpose
        from .ensemble import Member, PoseEnsemble
        from .model_catalog import AUTO_ORDER, EXTRA_MODELS, MAX_EXTRA_MODELS, clean_selection
        from .onnx_pose import OnnxPose, ensure_extra, find_extra

        automatic = cfg.ensemble_mode == "auto"
        dirs = [*self.model_search_dirs, self.models_dir]
        models = ([m for m in AUTO_ORDER if find_extra(m, dirs)] if automatic
                  else clean_selection(cfg.ensemble_models))
        if not models:
            return None
        if not rtmpose.onnxruntime_available():
            self._emit("status", "ONNX Runtime не установлен. Трекинг с MediaPipe.")
            return None
        limit = MAX_EXTRA_MODELS
        if automatic:
            from .autotune import system_info
            ram = system_info()["ram_gb"]
            if ram and ram < 8:
                limit = 1
            elif ram and ram < 16:
                limit = 2
            # Benchmark one variant per family.
            families = {"movenet_thunder": "movenet_lightning", "rtmpose_x": "rtmpose_m"}
            unique = {}
            for m in models:
                unique.setdefault(families.get(m, m), m)
            models = list(unique.values())

        members = []
        threads = max(1, min(4, (os.cpu_count() or 2) // 2))
        for m in models:
            if self._stop.is_set() or len(members) >= limit:
                break
            try:
                self._emit("status", f"Подготовка {EXTRA_MODELS[m]}…")
                path = ensure_extra(m, self.models_dir, self.model_search_dirs,
                                    manifest=self.manifest, download=False)
                if m in rtmpose.REFINERS:
                    adapter = rtmpose.RtmPose(path, rtmpose.REFINERS[m].input_wh,
                                              cfg.refiner_gpu, threads)
                else:
                    adapter = OnnxPose(m, path, cfg.refiner_gpu, threads)
                members.append(Member(m, adapter))
            except Exception as e:
                log.warning("Ensemble member %s unavailable: %s", m, e)
                self._emit("status", f"{EXTRA_MODELS[m]} пропущена: {e}")
        return PoseEnsemble(members, automatic) if members else None

    # ---------------- session recording (pose numbers only)
    rec_seconds: Optional[float] = None       # shown in the UI while recording

    def _recording_step(self, recorder, res, ts: float, aspect: float, cfg: Config, cam,
                        replay: bool):
        with self._pick_lock:
            req, self._rec_request = self._rec_request, None
        if req == "start" and recorder is None:
            if replay:
                self._emit("status", "Запись недоступна при воспроизведении записи")
            elif self.data_dir is None:
                self._emit("status", "Запись недоступна: нет папки данных")
            else:
                recorder = Recorder(new_recording_path(self.data_dir),
                                    {"aspect": round(aspect, 4), "vfov": cfg.camera_vfov,
                                     "model": cfg.model, "camera_fps": round(cam.measured_fps, 1),
                                     "created": time.strftime("%Y-%m-%d %H:%M")})
                self._emit("recording", {"state": "started"})
        if recorder is not None and res is not None:
            recorder.add(ts, res)
        if recorder is not None and (req == "stop" or recorder.full):
            self._finish_recording(recorder)
            recorder = None
        self.rec_seconds = recorder.seconds if recorder is not None else None
        return recorder

    def _finish_recording(self, recorder) -> None:
        self.rec_seconds = None
        try:
            path = recorder.save()
        except Exception as e:
            log.exception("saving the recording failed")
            self._emit("recording", {"state": "failed", "error": str(e)})
            return
        self._emit("recording", {"state": "saved" if path else "empty",
                                 "path": str(path) if path else "",
                                 "seconds": round(recorder.seconds, 1),
                                 "full": recorder.full})

    def _make_hands(self):
        try:
            from .hands import HandTracker
            self._emit("status", "Загрузка модели рук…")
            return HandTracker(self._load_model("hand"))
        except Exception as e:
            log.warning("finger tracking unavailable: %s", e)
            self._emit("status", f"Пальцы недоступны: {e}")
            return None

    def _multicam(self, cfg: Config, res: Optional[PoseResult], aux: AuxCameras, ts: float,
                  people: int = 1):
        """Rig calibration + fusion of the extra cameras into the main result."""
        info: Dict[str, Any] = {"cameras": [], "fused": 0, "rig_progress": None}
        if not aux.workers:
            return res, info
        fresh = aux.fresh(ts)
        if cfg.rig_auto and self._rig is None:
            # background calibration from ordinary movements (see multicam.AutoRig)
            for wk in aux.workers:
                item = fresh.get(wk.source)
                R_old = (rig_rotation(cfg.camera_rig.get(wk.source))
                         if wk.source in cfg.camera_rig else None)
                if res is not None and item is not None:
                    self._auto_rig.add(wk.source, res, item[1],
                                       people > 1 or wk.state.people > 1)
                    if R_old is not None:
                        self._auto_rig.note_agreement(wk.source, agreement(R_old, item[1], res),
                                                      ts)
                entry = self._auto_rig.step(wk.source, R_old, ts)
                if entry is not None:
                    cfg.camera_rig[wk.source] = entry
                    self._emit("rig_calibrated", {wk.source: entry})
        with self._pick_lock:
            req, self._rig_request = self._rig_request, None
        if req is not None:
            self._rig = RigCalibrator([w.source for w in aux.workers], req)
        if self._rig is not None:
            if res is not None:
                self._rig.add(res, {s: r for s, (_t, r) in fresh.items()})
            info["rig_progress"] = self._rig.progress
            if self._rig.done():
                result = self._rig.solve()
                self._rig = None
                for src, entry in result.items():
                    if "R" in entry:
                        cfg.camera_rig[src] = entry
                self._emit("rig_calibrated", result)
        views = [(np.eye(3), res, 1.0)] if res is not None else []
        for wk in aux.workers:
            st = {"source": wk.source, "name": getattr(wk, "name", ""), "fps": wk.state.fps,
                  "found": wk.state.found,
                  "people": wk.state.people, "error": wk.state.error,
                  "rms_cm": (cfg.camera_rig.get(wk.source) or {}).get("rms_cm"),
                  "used": False}
            R = rig_rotation(cfg.camera_rig.get(wk.source)) if wk.source in cfg.camera_rig else None
            item = fresh.get(wk.source)
            if R is not None and item is not None and cfg.multicam_fusion:
                t_aux, r_aux = item
                if res is None or agreement(R, r_aux, res) <= AGREE_LIMIT_M:
                    views.append((R, r_aux, float(np.exp(-max(0.0, ts - t_aux) / 0.1))))
                    st["used"] = True
            info["cameras"].append(st)
        if len(views) >= 2 or (res is None and views):
            world, vis = fuse(views)
            res = PoseResult(world, res.image if res is not None else None, vis)
            info["fused"] = len(views)
        return res, info

    def _publish(self, cfg, rgb, w, h, res, pts, trackers, head, lost, calibration,
                 hand_state, seq, ts, cam, est, proc_fps, infer_ms, hmd, rinfo, person_info,
                 multi_info=None, extra=None):
        # the preview reuses the small RGB image already made for the model: no extra resize
        preview = rgb if cfg.show_preview else None
        with self._calib_lock:
            calib_progress = (len(self._calib_samples) / self._calib_needed
                              if self._calib_needed else None)
        hands_snap = {s: {"image": st.get("image"), "curls": st.get("curls")}
                      for s, st in hand_state.items() if st.get("image") is not None}
        outs = []
        if cfg.output_vrchat:
            outs.append("VRChatOutput")
        if cfg.output_driver:
            outs.append("DriverOutput")
        snap = {
            "seq": seq,
            "frame": preview,
            "camera_lost": False,
            "image_lm": res.image if res is not None else None,
            "multicam": multi_info or {},
            "visibility": res.visibility if res is not None else None,
            "others": person_info.get("others", []),
            "person": {k: v for k, v in person_info.items() if k != "others"},
            "pts": pts,
            "trackers": list(trackers) if not lost else [],
            "head": head if not lost else None,
            "hands": hands_snap,
            "tracking": res is not None,
            "lost": lost,
            "calib_progress": calib_progress,
            "calibrated": calibration is not None,
            "calib_has_hmd_yaw": bool(calibration is not None and calibration.hmd_yaw is not None),
            "pose_state": rinfo.state if rinfo is not None else None,
            "statuses": rinfo.statuses if rinfo is not None else {},
            "legs_estimated": rinfo.legs_estimated if rinfo is not None else [0.0, 0.0],
            "controls": self.sender.controls_state.describe(),
            "confidence": (extra or {}).get("confidence", {}),
            "hints": (extra or {}).get("hints", []),
            "stats": {
                "cam_fps": cam.measured_fps,
                "cam_size": cam.actual_size,
                "proc_fps": proc_fps,
                "infer_ms": infer_ms,
                "latency_ms": (time.monotonic() - ts) * 1000.0,
                "delegate": est.delegate,
                "model": cfg.model,
                "send_rate": self.sender.rate,
                "interp_ms": self.sender.delay_ms if cfg.output_smoothing else 0.0,
                "paused": self.paused,
                "hmd": bool(self.sender.anchored),
                "vrc_hmd": bool(self.sender.vrc_anchored),
                "holding": bool(self.sender.holding),
                "oscquery": self.sender.oscquery_status,
                "recording": self.rec_seconds,
                "refiner": self.refiner_info,
                "replay": is_recording(cfg.camera),
                "outputs": outs,
            },
        }
        with self._snap_lock:
            self._snapshot = snap

    def _publish_lost(self, cfg, cam, est) -> None:
        """Camera unplugged: tell the window (no frames arrive meanwhile)."""
        with self._snap_lock:
            snap = dict(self._snapshot) if self._snapshot else None
        if snap is None:
            return
        snap["camera_lost"] = True
        snap["tracking"] = False
        snap["lost"] = True
        snap["trackers"], snap["head"] = [], None
        snap["seq"] = -2                 # the window redraws once (cards: no body)
        snap["stats"] = dict(snap["stats"], cam_fps=0.0)
        with self._snap_lock:
            self._snapshot = snap

    def _calibration_step(self, smooth: np.ndarray, cfg: Config,
                          hmd: Optional[steamvr.HmdReader],
                          capture_t: Optional[float] = None) -> Optional[Calibration]:
        with self._calib_lock:
            if not self._calib_needed:
                return None
            if time.monotonic() > self._calib_deadline:
                self._calib_needed = 0
                self._calib_samples = []
                self._emit("calib_failed", "Калибровка не удалась: тело не было видно целиком")
                return None
            self._calib_samples.append(smooth.copy())
            # headset headings DURING the calibration (at each frame's capture time):
            # their average is far steadier than one value at the very end
            t_cap = time.monotonic() if capture_t is None else capture_t
            for state, key in ((self.hmd_state, "steamvr"), (self.sender.vrc_hmd_state, "vrchat")):
                p = state.at(t_cap, 0.5)
                if p is not None and not p.virtual:
                    self._calib_yaws.setdefault(key, []).append(yaw_of(p.rot[:, 2]))
            if len(self._calib_samples) < self._calib_needed:
                return None
            samples = self._calib_samples
            yaws, self._calib_yaws = self._calib_yaws, {}
            self._calib_needed = 0
            self._calib_samples = []
        try:
            cal = compute_calibration(samples, cfg.user_height, cfg.root_motion)
        except Exception as e:
            self._emit("calib_failed", f"Калибровка не удалась: {e}")
            return None
        # you looked straight at the camera: that heading = the camera's forward
        if _wants_hmd(cfg) and len(yaws.get("steamvr", [])) >= len(samples) // 2:
            cal.hmd_yaw = circular_mean_deg(yaws["steamvr"])
        if cfg.vrchat_hmd and cfg.output_vrchat and \
                len(yaws.get("vrchat", [])) >= len(samples) // 2:
            cal.vrc_yaw = circular_mean_deg(yaws["vrchat"])   # VRChat's own headset pose
        if cfg.output_vrchat:
            # VRChat: align its heading ONCE (calibrated space faces +Z -> yaw 0);
            # ignored while the body is glued to VRChat's headset pose
            self.sender.request_yaw_align(0.0)
        self._emit("calibrated", cal.to_dict())
        return cal
