"""Notice when Windows adds or removes a device (a camera plugged in / out).

Windows broadcasts WM_DEVICECHANGE to every top-level window. Nexovara Pose then
leaves the cameras alone for a few seconds (camera.notify_device_change): no
camera is listed or opened while Windows is still installing the new device's
driver — touching a half-initialised camera driver is what can crash it.
"""
from __future__ import annotations

import ctypes
import logging
import sys
from typing import Callable

from PySide6.QtCore import QAbstractNativeEventFilter

log = logging.getLogger(__name__)

WM_DEVICECHANGE = 0x0219
DBT_DEVNODES_CHANGED = 0x0007
DBT_DEVICEARRIVAL = 0x8000
DBT_DEVICEREMOVECOMPLETE = 0x8004
_INTERESTING = {DBT_DEVNODES_CHANGED, DBT_DEVICEARRIVAL, DBT_DEVICEREMOVECOMPLETE}


if sys.platform.startswith("win"):
    from ctypes import wintypes

    class _MSG(ctypes.Structure):
        _fields_ = [("hwnd", wintypes.HWND), ("message", wintypes.UINT),
                    ("wParam", wintypes.WPARAM), ("lParam", wintypes.LPARAM),
                    ("time", wintypes.DWORD), ("pt", wintypes.POINT)]
else:
    _MSG = None


class DeviceChangeFilter(QAbstractNativeEventFilter):
    def __init__(self, on_change: Callable[[int], None]):
        super().__init__()
        self._on_change = on_change
        self._failed = False

    def nativeEventFilter(self, event_type, message):  # noqa: N802 (Qt naming)
        if _MSG is None or self._failed:
            return False, 0
        try:
            if bytes(event_type) != b"windows_generic_MSG":
                return False, 0
            msg = _MSG.from_address(int(message))
            if msg.message == WM_DEVICECHANGE and int(msg.wParam) in _INTERESTING:
                self._on_change(int(msg.wParam))
        except Exception as e:         # never break the window's event handling
            self._failed = True
            log.warning("device change watcher disabled: %s", e)
        return False, 0


def install(app, on_change: Callable[[int], None]):
    """Returns the filter (keep a reference!) or None where not supported."""
    if _MSG is None:
        return None
    f = DeviceChangeFilter(on_change)
    app.installNativeEventFilter(f)
    return f
