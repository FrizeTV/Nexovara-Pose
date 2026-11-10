"""Controller buttons from the keyboard and mouse — for playing WITHOUT real
VR controllers ("SteamVR without a headset" with the Nexovara Pose driver).

VRChat in VR mode is driven by controller buttons only: the Quick Menu opens
with B, "Calibrate FBT" is clicked with the laser pointer + trigger and the
calibration is confirmed with BOTH triggers. Without controllers none of that
was possible. Global keys (they work while the game window has the focus):

  menu key     (F8)   -> B on the left hand   (VRChat: Quick Menu)
  triggers key (F9)   -> both triggers        (VRChat: confirm FBT calibration)
  pointer key  (F10)  -> toggle "mouse pointer": the right hand points where
                         the mouse cursor is (centre of the screen = straight
                         ahead), left mouse button = right trigger.

Only the key STATE is read (GetAsyncKeyState), nothing is recorded or stored.
Windows only; elsewhere the controls are simply inactive.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .mathutil import rot_x, rot_y

# button bits — must match driver/src/protocol.h
BTN_A, BTN_B, BTN_SYSTEM, BTN_TRIGGER, BTN_GRIP, BTN_STICK = 1, 2, 4, 8, 16, 32

KEY_CHOICES = ["off"] + [f"F{i}" for i in range(1, 13)]
_VK = {f"F{i}": 0x6F + i for i in range(1, 13)}       # VK_F1 = 0x70
VK_LBUTTON = 0x01

POINTER_YAW_DEG = 55.0      # cursor at the left/right screen edge
POINTER_PITCH_DEG = 40.0    # cursor at the top/bottom edge
# where the pointing hand is held, relative to the (level) head: right, down, forward
POINTER_OFFSET = np.array([0.20, -0.28, 0.32])


@dataclass
class ControlState:
    left: int = 0                                  # button bits per hand
    right: int = 0
    pointer: Optional[Tuple[float, float]] = None  # (yaw, pitch) degrees when the mouse aims

    @property
    def active(self) -> bool:
        return bool(self.left or self.right or self.pointer is not None)

    def describe(self) -> str:
        parts = []
        if self.pointer is not None:
            parts.append("указка мышью")
        if self.left & BTN_B:
            parts.append("меню")
        if (self.left | self.right) & BTN_TRIGGER:
            parts.append("курок")
        return ", ".join(parts)


class _Win32:
    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._u = ctypes.windll.user32
        self._u.GetAsyncKeyState.argtypes = [ctypes.c_int]
        self._u.GetAsyncKeyState.restype = ctypes.c_short
        self._pt = wintypes.POINT()
        self._byref = ctypes.byref

    def down(self, vk: int) -> bool:
        return bool(self._u.GetAsyncKeyState(vk) & 0x8000)

    def cursor(self) -> Optional[Tuple[float, float]]:
        """Cursor position relative to the primary screen: -1..1 (x right, y down)."""
        if not self._u.GetCursorPos(self._byref(self._pt)):
            return None
        w, h = self._u.GetSystemMetrics(0), self._u.GetSystemMetrics(1)
        if w <= 0 or h <= 0:
            return None
        return ((self._pt.x - w / 2) / (w / 2), (self._pt.y - h / 2) / (h / 2))


class KeyboardControls:
    """Polled by the output thread (~90 Hz); cheap: a few API calls."""

    def __init__(self, key_menu: str = "F8", key_triggers: str = "F9", key_pointer: str = "F10",
                 backend=None):
        self.keys = {"menu": key_menu, "triggers": key_triggers, "pointer": key_pointer}
        self._b = backend
        if self._b is None and sys.platform.startswith("win"):
            try:
                self._b = _Win32()
            except Exception:
                self._b = None
        self.pointer_on = False
        self._pointer_was_down = False

    @property
    def available(self) -> bool:
        return self._b is not None

    def _down(self, which: str) -> bool:
        vk = _VK.get(self.keys.get(which, "off"))
        return vk is not None and self._b.down(vk)

    def poll(self) -> ControlState:
        st = ControlState()
        if self._b is None:
            return st
        try:
            if self._down("menu"):
                st.left |= BTN_B
            if self._down("triggers"):
                st.left |= BTN_TRIGGER
                st.right |= BTN_TRIGGER
            p = self._down("pointer")
            if p and not self._pointer_was_down:
                self.pointer_on = not self.pointer_on
            self._pointer_was_down = p
            if self.pointer_on:
                c = self._b.cursor()
                if c is not None:
                    nx = max(-1.0, min(1.0, c[0]))
                    ny = max(-1.0, min(1.0, c[1]))
                    st.pointer = (nx * POINTER_YAW_DEG, ny * POINTER_PITCH_DEG)
                if self._b.down(VK_LBUTTON):
                    st.right |= BTN_TRIGGER
        except Exception:
            return ControlState()
        return st


def level_rotation(R: np.ndarray) -> np.ndarray:
    """Only the yaw (heading) of a rotation — Unity axes, +Z forward, +Y up."""
    fwd = R[:, 2]
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    return rot_y(yaw)


def pointer_pose(head_pos: np.ndarray, head_rot: np.ndarray, yaw_deg: float, pitch_deg: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Right hand held in front of the body, pointing where the mouse aims."""
    base = level_rotation(head_rot)
    pos = np.asarray(head_pos, dtype=np.float64) + base @ POINTER_OFFSET
    rot = base @ rot_y(yaw_deg) @ rot_x(pitch_deg)
    return pos, rot
