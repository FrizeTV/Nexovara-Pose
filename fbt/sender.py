"""Output sender thread with motion interpolation.

The camera delivers 30–60 poses per second, VR runs at 90+ Hz. Sending each
camera pose once makes trackers move in visible steps. This thread sends at a
fixed rate and renders the pose for "now − delay" by interpolating between the
two surrounding camera poses (positions: linear, rotations: slerp; finger
curls: linear). Optional prediction extrapolates a few milliseconds ahead to
win back latency. With smoothing off, every new pose is sent immediately.

All OSC output happens on this thread only, so the output objects never need
locking.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .controls import ControlState, KeyboardControls
from .mathutil import mat_to_quat, quat_slerp, quat_to_mat, yaw_of
from .hmd import BodyAnchor, HmdState
from .outputs import DriverOutput, VRChatOutput
from .skeleton import Tracker
from .system import precise_sleep

log = logging.getLogger(__name__)

MAX_EXTRAPOLATION_S = 0.05
VRC_HMD_MAX_AGE = 1.0      # s: a VRChat headset pose older than this is not used


@dataclass
class Keyframe:
    t: float                                   # arrival time (time.monotonic)
    trackers: List[Tracker]
    head: Optional[Tracker] = None
    hands: Dict[str, Tuple[Tracker, Optional[np.ndarray]]] = field(default_factory=dict)
    yaw: Optional[float] = None                # headset heading at calibration (driver mode)
    vrc_yaw: Optional[float] = None            # same, VRChat's headset pose (OSCQuery)
    capture_t: Optional[float] = None          # when the camera took the picture (monotonic)


def _lerp_tracker(a: Tracker, b: Tracker, u: float) -> Tracker:
    pos = a.pos + (b.pos - a.pos) * u
    if a.rot is None or b.rot is None:
        rot = b.rot if b.rot is not None else a.rot
    else:
        rot = quat_to_mat(quat_slerp(mat_to_quat(a.rot), mat_to_quat(b.rot), u))
    return Tracker(b.name, pos, rot)


def interpolate(k0: Keyframe, k1: Keyframe, t: float) -> Keyframe:
    span = k1.t - k0.t
    u = 1.0 if span <= 1e-6 else (t - k0.t) / span
    u = min(max(u, 0.0), 1.0 + MAX_EXTRAPOLATION_S / max(span, 1e-3))
    a = {x.name: x for x in k0.trackers}
    trackers = [_lerp_tracker(a[x.name], x, u) if x.name in a else x for x in k1.trackers]
    head = (_lerp_tracker(k0.head, k1.head, u) if k0.head is not None and k1.head is not None
            else k1.head)
    hands = {}
    for side, (t1, c1) in k1.hands.items():
        if side in k0.hands:
            t0, c0 = k0.hands[side]
            c = c1 if (c0 is None or c1 is None) else np.clip(c0 + (c1 - c0) * min(u, 1.0), 0, 1)
            hands[side] = (_lerp_tracker(t0, t1, u), c)
        else:
            hands[side] = (t1, c1)
    cap = None
    if k0.capture_t is not None and k1.capture_t is not None:
        cap = k0.capture_t + (k1.capture_t - k0.capture_t) * u
    return Keyframe(t, trackers, head, hands, k1.yaw, k1.vrc_yaw, cap)


class OutputSender:
    def __init__(self, hmd_state: Optional[HmdState] = None):
        self.hmd_state = hmd_state or HmdState()
        self._anchor = BodyAnchor()
        self._yaw_align: Optional[float] = None
        self.anchored = False     # the body is glued to a real headset right now
        # VRChat OSC: the headset pose from VRChat itself (OSCQuery, optional)
        self.vrc_hmd_state = HmdState()
        self._vrc_anchor = BodyAnchor()
        self.vrc_anchored = False
        self.oscquery = None      # oscquery.OscQueryService while enabled
        self._lock = threading.Condition()
        self._frames: Deque[Keyframe] = deque(maxlen=6)
        self._new = False
        self._cfg: Optional[Config] = None
        self._pending_cfg: Optional[Config] = None
        self._outputs: list = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.paused = False
        self.holding = False      # tracking lost: the last pose is being held
        self.rate = 0.0          # packets per second actually sent
        self.delay_ms = 0.0      # current interpolation delay
        self._delay = 0.05       # smoothed delay (s) — changes slowly, never jumps
        self._last_render = -1e9  # render time never goes backwards
        self._controls: Optional[KeyboardControls] = None
        self.controls_state = ControlState()   # last keyboard/mouse state (for the UI)

    # ------------------------------------------------------------ control
    def configure(self, cfg: Config) -> None:
        with self._lock:
            self._pending_cfg = copy.deepcopy(cfg)
            self._lock.notify_all()

    def request_yaw_align(self, yaw_deg: float) -> None:
        """VRChat OSC: align VRChat's heading once (after calibration / on request)."""
        with self._lock:
            self._yaw_align = float(yaw_deg)

    def push(self, kf: Keyframe) -> None:
        with self._lock:
            self._frames.append(kf)
            self._new = True
            self._lock.notify_all()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="sender", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------ worker
    @staticmethod
    def _wants_controls(cfg: Config) -> bool:
        return bool(cfg.output_driver and cfg.kb_controls
                    and (cfg.driver_hmd or cfg.finger_output == "driver"))

    @staticmethod
    def _build(cfg: Config) -> list:
        outs = []
        if cfg.output_vrchat:
            outs.append(VRChatOutput(cfg.vrchat_port, cfg.send_head))
        if cfg.output_driver:
            outs.append(DriverOutput(cfg.driver_port, cfg.driver_hmd, cfg.hmd_head_rotation,
                                     predict=cfg.steamvr_prediction))
        return outs

    def _send(self, kf: Keyframe, controls_only: bool = False, moving: bool = True) -> None:
        """controls_only: tracking is lost, but keys/mouse are in use — send just
        the hands/HMD to the driver so buttons still work (no stale trackers)."""
        cfg = self._cfg
        ctl = self.controls_state if self._controls is not None else None
        with self._lock:
            yaw_align, self._yaw_align = self._yaw_align, None
        for o in self._outputs:
            if isinstance(o, VRChatOutput):
                if yaw_align is not None:
                    o.request_yaw_align(yaw_align)
                if controls_only:
                    continue
                hmd = (self.vrc_hmd_state.at(self._hmd_time(kf), VRC_HMD_MAX_AGE)
                       if cfg.vrchat_hmd and self.oscquery is not None else None)
                if hmd is not None and kf.head is not None:
                    trackers, head, _ = self._vrc_anchor.apply(kf.trackers, kf.head, {}, hmd,
                                                               kf.vrc_yaw, kf.t)
                    # VRChat compares this with its own headset (eyes): send exactly that
                    real = (hmd.pos.copy(), yaw_of(hmd.rot[:, 2]),
                            self._vrc_anchor.last_yaw or 0.0)
                    o.send(trackers, head, kf.t, real_head=real)
                    self.vrc_anchored = True
                else:
                    if self.vrc_anchored:
                        self._vrc_anchor.reset()
                    o.send(kf.trackers, kf.head, kf.t)
                    self.vrc_anchored = False
            else:  # own driver: hands as controllers in no-headset mode or when chosen
                use_hands = cfg.driver_hmd or cfg.finger_output == "driver"
                trackers, head, hands = kf.trackers, kf.head, (kf.hands if use_hands else {})
                anchored = False
                if cfg.hmd_anchor and not cfg.driver_hmd:
                    # the headset where it was when the camera took this picture
                    hmd = self.hmd_state.at(self._hmd_time(kf), 0.3)
                    if hmd is not None and not hmd.virtual:
                        trackers, head, hands = self._anchor.apply(
                            trackers, head, hands or {}, hmd, kf.yaw, kf.t)
                        anchored = True
                self.anchored = anchored
                o.send([] if controls_only else trackers, head, hands or None, ctl,
                       t=kf.t, moving=moving and not controls_only)

    @staticmethod
    def _hmd_time(kf: Keyframe) -> float:
        return kf.capture_t if kf.capture_t is not None else time.monotonic()

    def _auto_delay(self) -> float:
        fr = list(self._frames)
        if len(fr) < 3:
            return 0.03
        gaps = np.diff([f.t for f in fr])
        return float(np.clip(np.median(gaps) * 1.15, 0.008, 0.08))

    def _run(self) -> None:
        sent, t_rate = 0, time.monotonic()
        next_tick = time.monotonic()
        try:
            while not self._stop.is_set():
                with self._lock:
                    if self._pending_cfg is not None:
                        for o in self._outputs:
                            o.shutdown()
                        self._cfg = self._pending_cfg
                        self._pending_cfg = None
                        self._outputs = self._build(self._cfg)
                        c = self._cfg
                        self._controls = (KeyboardControls(c.key_menu, c.key_triggers,
                                                           c.key_pointer)
                                          if self._wants_controls(c) else None)
                        if self._controls is not None and not self._controls.available:
                            self._controls = None
                        self._manage_oscquery(c)
                    cfg = self._cfg
                    if cfg is None or not cfg.output_smoothing:
                        self._lock.wait_for(lambda: self._new or self._stop.is_set()
                                            or self._pending_cfg is not None, 0.1)
                    frames = list(self._frames)
                    new = self._new
                    self._new = False
                if cfg is None or not frames:
                    time.sleep(0.005)
                    continue
                now = time.monotonic()
                age = now - frames[-1].t
                fresh = age <= cfg.hold_time
                # you're out of view for a moment (too close to the camera, turned
                # away, someone walked by): the last pose is HELD and keeps being
                # sent — SteamVR/VRChat keep the trackers instead of dropping them
                # (which lost the in-game calibration and parked trackers far away)
                holding = not fresh and age <= cfg.lost_hold_s
                self.holding = holding
                if cfg.output_smoothing:
                    period = 1.0 / cfg.send_rate
                    next_tick += period
                    wait = next_tick - time.monotonic()
                    if wait > 0:
                        precise_sleep(wait, self._stop)  # Event.wait is ~15 ms coarse on Windows
                    else:
                        next_tick = time.monotonic()
                    now = time.monotonic()
                    target = (cfg.interp_delay_ms / 1000.0) if cfg.interp_delay_ms \
                        else self._auto_delay()
                    # adapt the delay slowly so the rendered time stays smooth
                    self._delay += (target - self._delay) * min(1.0, period / 0.5)
                    self.delay_ms = self._delay * 1000
                    t = now - self._delay + cfg.prediction_ms / 1000.0
                    t = max(t, self._last_render)
                    self._last_render = t
                    kf = self._render(frames, t)
                elif new:
                    kf = frames[-1]
                else:
                    kf = None
                if self._controls is not None:
                    self.controls_state = self._controls.poll()
                else:
                    self.controls_state = ControlState()
                for o in self._outputs:
                    if isinstance(o, DriverOutput):
                        o.poll_feedback(self.hmd_state)   # the real headset's pose
                if kf is None and self._controls is not None and self.controls_state.active:
                    kf = frames[-1]
                if holding:
                    # frozen last pose, glued to where the headset is NOW
                    kf = dataclasses.replace(frames[-1], t=now, capture_t=None)
                if kf is not None and not self.paused and self._outputs:
                    if fresh or holding:
                        self._send(kf, moving=not holding)
                        sent += 1
                    elif self.controls_state.active:
                        self._send(kf, controls_only=True)
                if now - t_rate >= 1.0:
                    self.rate = sent / (now - t_rate)
                    sent, t_rate = 0, now
        except Exception:
            log.exception("sender crashed")
        finally:
            for o in self._outputs:
                try:
                    o.shutdown()
                except Exception:
                    pass
            if self.oscquery is not None:
                self.oscquery.stop()
                self.oscquery = None

    def _manage_oscquery(self, cfg: Config) -> None:
        """VRChat's headset pose (OSCQuery): runs only while wanted."""
        want = bool(cfg.output_vrchat and cfg.vrchat_hmd)
        if want and self.oscquery is None:
            from .oscquery import OscQueryService
            svc = OscQueryService(self.vrc_hmd_state)
            try:
                svc.start()
                self.oscquery = svc
            except OSError as e:
                log.warning("OSCQuery service failed to start: %s", e)
                svc.stop()
        elif not want and self.oscquery is not None:
            self.oscquery.stop()
            self.oscquery = None
            self.vrc_hmd_state.clear()
            self.vrc_anchored = False

    @property
    def oscquery_status(self) -> str:
        return self.oscquery.status if self.oscquery is not None else ""

    @staticmethod
    def _render(frames: List[Keyframe], t: float) -> Keyframe:
        if len(frames) == 1 or t <= frames[0].t:
            return frames[0] if t <= frames[0].t else frames[-1]
        for k0, k1 in zip(frames, frames[1:]):
            if k0.t <= t <= k1.t:
                return interpolate(k0, k1, t)
        return interpolate(frames[-2], frames[-1], t)  # short, clamped extrapolation
