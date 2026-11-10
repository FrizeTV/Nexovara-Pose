"""Output backends.

* VRChatOutput — native VRChat OSC trackers (VRChat on this PC).
  /tracking/trackers/{1..8}/position|rotation  (Unity space, euler degrees)
  /tracking/trackers/head/position             (per-frame position alignment)
  /tracking/trackers/head/rotation             (ONLY as a one-shot yaw alignment:
  VRChat slowly turns its whole tracking space towards every head rotation it
  receives, so a continuously sent camera head yaw made the body drift and
  look crooked). VRChat has no OSC input for fingers.
* DriverOutput — Nexovara Pose's own SteamVR driver (driver/ folder, C++). One UDP
  datagram per frame to 127.0.0.1 with every device (see driver/src/protocol.h):
  body trackers, hands as controllers with finger curls, optional virtual HMD.
  The driver answers with the pose of the REAL headset (if there is one), used
  to glue the body to the headset (see hmd.py).

Tracker slots are fixed per body part (see config.TRACKER_NAMES), so SteamVR
role assignments stay valid when individual trackers are switched on or off.
"""
from __future__ import annotations

import logging
import math
import select
import struct
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import TRACKER_NAMES
from .mathutil import mat_to_quat, mat_to_unity_euler, quat_slerp, quat_to_mat
from .osc import OscClient
from .skeleton import Tracker

log = logging.getLogger(__name__)


def slot_of(name: str) -> int:
    """0-based fixed slot of a body tracker."""
    return TRACKER_NAMES.index(name)


class VRChatOutput:
    MAX_TRACKERS = 8
    # head position for VRChat's per-frame alignment: an extra, speed-adaptive
    # filter — VRChat shifts EVERY tracker by whatever the head does, so head
    # noise would shake the whole body; walking still follows with little lag
    HEAD_MIN_CUTOFF, HEAD_BETA = 0.7, 2.5

    def __init__(self, port: int, send_head: bool = True):
        from .filters import OneEuroFilter
        self.client = OscClient(port)
        self.send_head = send_head
        self._head_filt = OneEuroFilter(self.HEAD_MIN_CUTOFF, self.HEAD_BETA)
        self._yaw_pending: Optional[float] = None
        self._last_real: Optional[Tuple[float, float]] = None
        self._eye_q: Optional[np.ndarray] = None
        self._eye_t: Optional[float] = None
        self.yaw_sent = 0

    def request_yaw_align(self, yaw_deg: float) -> None:
        """One-shot yaw alignment: VRChat snaps its tracking space to this head
        yaw if no head rotation arrived for >300 ms (we never send it otherwise)."""
        self._yaw_pending = float(yaw_deg)

    def send(self, trackers: List[Tracker], head: Optional[Tracker], t: float = 0.0,
             real_head: Optional[Tuple[np.ndarray, float, float]] = None) -> None:
        """real_head: (position, heading°, anchor heading°) of the REAL headset in
        VRChat's space (OSCQuery, see oscquery.py) when the body is glued to it.
        Then VRChat gets the exact head position and heading every frame (no camera
        noise -> its alignment never shakes or turns the body)."""
        for tr in trackers:
            i = slot_of(tr.name) + 1
            if i > self.MAX_TRACKERS:
                continue
            self.client.send(f"/tracking/trackers/{i}/position", *map(float, tr.pos))
            if tr.rot is not None:
                self.client.send(f"/tracking/trackers/{i}/rotation",
                                 *map(float, mat_to_unity_euler(tr.rot)))
        if real_head is not None:
            pos, yaw, anchor_yaw = real_head
            self.client.send("/tracking/trackers/head/position", *map(float, pos))
            self.client.send("/tracking/trackers/head/rotation", 0.0, float(yaw % 360.0), 0.0)
            self._yaw_pending = None
            self._last_real = (float(yaw), float(anchor_yaw))
            self._head_filt.reset()
            return
        if self._last_real is not None:
            # the headset pose stopped coming: keep the body facing the same way
            # (one-shot alignment equal to the last anchored heading)
            yaw, anchor_yaw = self._last_real
            self._last_real = None
            self._yaw_pending = yaw - anchor_yaw
        if self.send_head and head is not None:
            # VRChat lines this up with ITS headset position = the eyes. We used to
            # send the middle of the head (between the ears), ~9 cm behind the eyes:
            # VRChat then moved every tracker 9 cm forward — chest in front, head
            # pulled back. Now the eyes, exactly like the virtual headset of the driver.
            rot = head.rot
            if rot is not None and t:
                # the camera's head rotation is noisy; it only swings a 9 cm lever
                # here, so it is smoothed (~0.15 s) — no shaking passed on to VRChat
                q = np.array(mat_to_quat(rot))
                if self._eye_q is None or self._eye_t is None or not (0 < t - self._eye_t < 0.5):
                    self._eye_q = q
                else:
                    a = 1.0 - math.exp(-(t - self._eye_t) / 0.15)
                    self._eye_q = quat_slerp(self._eye_q, q, a)
                self._eye_t = t
                rot = quat_to_mat(self._eye_q)
            hp = eyes_of(Tracker(head.name, head.pos, rot))
            hp = self._head_filt(hp, t) if t else hp
            self.client.send("/tracking/trackers/head/position", *map(float, hp))
        if self._yaw_pending is not None:
            self.client.send("/tracking/trackers/head/rotation", 0.0,
                             float(self._yaw_pending % 360.0), 0.0)
            self._yaw_pending = None
            self.yaw_sent += 1

    def send_hands(self, *_a, **_k) -> None:  # not supported by VRChat OSC
        pass

    def shutdown(self) -> None:
        self.client.close()


# ------------------------------------------------------------ own SteamVR driver
DRIVER_MAGIC = b"CFBT"
DRIVER_PROTOCOL = 1
DEV_HMD, DEV_LEFT_HAND, DEV_RIGHT_HAND = 16, 17, 18
FLAG_VALID, FLAG_CURLS, FLAG_BUTTONS, FLAG_VELOCITY = 1, 2, 4, 8
MAX_VELOCITY = 6.0          # m/s sent to the driver (the driver accepts up to 20)


class VelocityTracker:
    """Velocity of every tracker from the app's own smooth output stream, by the
    RENDER time of each frame (not by when packets happen to be sent — that
    timing jitters on Windows and made a velocity guessed by the driver noisy:
    trackers shook in SteamVR). Smoothed (~80 ms: measured to add no visible
    jitter), a soft dead-band makes a
    standing tracker exactly 0, limited to MAX_VELOCITY."""

    TAU = 0.08
    DEADBAND = 0.08

    def __init__(self):
        self._prev: Dict[str, Tuple[float, np.ndarray]] = {}
        self._vel: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._prev.clear()
        self._vel.clear()

    def __call__(self, name: str, pos: np.ndarray, t: float) -> np.ndarray:
        prev = self._prev.get(name)
        self._prev[name] = (t, np.asarray(pos, dtype=np.float64).copy())
        if prev is None or not (0.0 < t - prev[0] <= 0.1):
            self._vel[name] = np.zeros(3)
            return np.zeros(3)
        dt = t - prev[0]
        raw = (pos - prev[1]) / dt
        v = self._vel.get(name, np.zeros(3))
        v = v + (raw - v) * (1.0 - math.exp(-dt / self.TAU))
        self._vel[name] = v
        mag = float(np.linalg.norm(v))
        if not math.isfinite(mag) or mag == 0.0:
            return np.zeros(3)
        k = 1.0 if mag >= self.DEADBAND else (mag / self.DEADBAND) ** 2
        if mag * k > MAX_VELOCITY:
            k = MAX_VELOCITY / mag
        return v * k
_HEADER = struct.Struct("<4sHHIB3x")          # 16 bytes
_RECORD = struct.Struct("<BBH3f4f5f")         # 52 bytes
EYE_FORWARD_M, EYE_UP_M = 0.09, 0.03          # eyes relative to the ear midpoint


def eyes_of(head: Tracker) -> np.ndarray:
    """Eye position (where a headset is) from the camera's head (ear midpoint)."""
    if head.rot is None:
        return np.asarray(head.pos, dtype=np.float64)
    return head.pos + head.rot[:, 2] * EYE_FORWARD_M + head.rot[:, 1] * EYE_UP_M


def unity_to_openvr(pos, R) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    """Unity (left-handed, +Z forward) -> OpenVR (right-handed, -Z forward): mirror Z."""
    qx, qy, qz, qw = mat_to_quat(R)
    return (float(pos[0]), float(pos[1]), -float(pos[2])), (-qx, -qy, qz, qw)


def encode_driver_packet(seq: int, devices: List[tuple]) -> bytes:
    """devices: (id, tracker, curls or None[, buttons bits or None[, velocity (Unity) or None]])."""
    records = []
    for dev in devices[:32]:
        dev_id, t, curls = dev[:3]
        buttons = dev[3] if len(dev) > 3 else None
        vel = dev[4] if len(dev) > 4 else None
        p, q = unity_to_openvr(t.pos, t.rot)
        c = tuple(float(x) for x in np.clip(curls, 0.0, 1.0)) if curls is not None else (0.0,) * 5
        if vel is not None and curls is None:
            v = np.clip(np.asarray(vel, dtype=np.float64), -MAX_VELOCITY, MAX_VELOCITY)
            c = (float(v[0]), float(v[1]), -float(v[2]), 0.0, 0.0)    # Unity -> OpenVR
        vals = (*p, *q, *c)
        if not all(math.isfinite(v) for v in vals):
            continue  # never send NaN/inf
        flags = FLAG_VALID | (FLAG_CURLS if curls is not None else 0)
        if vel is not None and curls is None:
            flags |= FLAG_VELOCITY
        if buttons is not None:
            flags |= FLAG_BUTTONS
        records.append(_RECORD.pack(dev_id, flags, int(buttons or 0) & 0x3F, *vals))
    header = _HEADER.pack(DRIVER_MAGIC, DRIVER_PROTOCOL, 0, seq & 0xFFFFFFFF, len(records))
    return header + b"".join(records)


class DriverOutput:
    def __init__(self, port: int, send_hmd: bool = False, head_rotation: bool = False,
                 predict: bool = True):
        self.client = OscClient(port)          # same loopback-only UDP socket
        self.send_hmd = send_hmd
        self.head_rotation = head_rotation
        self.predict = predict
        self.velocity = VelocityTracker()
        self.seq = 0
        self.feedback_seen = False
        self._fb_thread: Optional[threading.Thread] = None
        self._fb_state = None
        self._fb_stop = threading.Event()

    def _hmd(self, head: Tracker, body: List[Tracker]) -> Optional[Tracker]:
        if head is None or head.rot is None:
            return None
        R = head.rot
        if not self.head_rotation:
            # level view that follows the body: you look at a monitor, not around
            hip = next((t for t in body if t.name == "hip" and t.rot is not None), None)
            fwd = (hip.rot if hip is not None else R)[:, 2]
            yaw = math.atan2(fwd[0], fwd[2])
            c, s_ = math.cos(yaw), math.sin(yaw)
            R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        pos = head.pos + R[:, 2] * EYE_FORWARD_M + R[:, 1] * EYE_UP_M
        return Tracker("hmd", pos, R)

    def send(self, trackers: List[Tracker], head: Optional[Tracker],
             hands: Optional[Dict[str, Tuple[Tracker, Optional[np.ndarray]]]] = None,
             controls=None, t: Optional[float] = None, moving: bool = True) -> None:
        """`controls`: ControlState from the keyboard/mouse (buttons, mouse pointer).
        `t`: render time of this frame (for the trackers' velocity); `moving`
        False = a held (frozen) pose: velocity 0."""
        devices: List[tuple] = []
        for tr in trackers:
            if tr.rot is not None:
                vel = None
                if self.predict and t is not None:
                    vel = self.velocity(tr.name, tr.pos, t) if moving else np.zeros(3)
                    if not moving:
                        self.velocity.reset()
                devices.append((slot_of(tr.name), tr, None, None, vel))
        hmd = self._hmd(head, trackers)
        if self.send_hmd and hmd is not None:
            devices.append((DEV_HMD, hmd, None))
        hands = dict(hands or {})
        if controls is not None and controls.pointer is not None and hmd is not None:
            from .controls import pointer_pose
            pos, rot = pointer_pose(hmd.pos, hmd.rot, *controls.pointer)
            prev = hands.get("right")
            hands["right"] = (Tracker("right_hand", pos, rot), prev[1] if prev else None)
        for side, (ht, curls) in hands.items():
            if ht.rot is not None:
                buttons = None
                if controls is not None:
                    buttons = controls.left if side == "left" else controls.right
                devices.append((DEV_LEFT_HAND if side == "left" else DEV_RIGHT_HAND, ht, curls,
                                buttons))
        if devices:
            self.seq += 1
            self.client.send_raw(encode_driver_packet(self.seq, devices))

    def poll_feedback(self, state) -> int:
        """Headset poses the driver sends back -> HmdState. Received by a small
        thread that waits on the socket, so every pose is stamped with the moment
        it ARRIVED (accurate to ~1 ms): the body is matched to the headset by time,
        and a pose stamped up to one sender tick late made that match wobble."""
        if self._fb_thread is None:
            self._fb_state = state
            self._fb_stop = threading.Event()
            self._fb_thread = threading.Thread(target=self._feedback_loop, daemon=True,
                                               name="driver-feedback")
            self._fb_thread.start()
        return 0

    def _feedback_loop(self) -> None:
        from .hmd import FB_VALID, FB_VIRTUAL, parse_feedback
        sock = self.client.socket
        while not self._fb_stop.is_set():
            try:
                ready, _, _ = select.select([sock], [], [], 0.05)
            except (OSError, ValueError):
                return                           # socket closed
            if not ready:
                continue
            t = time.monotonic()
            for data in self.client.receive():
                fb = parse_feedback(data)
                if fb is None:
                    continue
                pos, rot, flags = fb
                if flags & FB_VALID:
                    self._fb_state.update(pos, rot, "driver", virtual=bool(flags & FB_VIRTUAL),
                                          t=t)
                self.feedback_seen = True

    def shutdown(self) -> None:
        if self._fb_thread is not None:
            self._fb_stop.set()
            self._fb_thread.join(timeout=1.0)
            self._fb_thread = None
        self.client.close()
