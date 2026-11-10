"""OS-level performance helpers (all optional, fail silently).

* Windows timers: by default Windows wakes threads in ~15.6 ms steps, which
  makes a 90 Hz sender stutter (it lands on 64 Hz). timeBeginPeriod(1) asks
  for 1 ms resolution while the app runs; precise_sleep() uses time.sleep,
  which on Python 3.11+ uses high-resolution waitable timers on Windows
  (threading.Event.wait does NOT).
* Process priority: "above normal" keeps tracking smooth when a VR game loads
  the CPU, without starving the game (it is not "high"/"realtime").
"""
from __future__ import annotations

import logging
import sys
import time

log = logging.getLogger(__name__)
_timer_enabled = False


def enable_high_res_timer() -> None:
    global _timer_enabled
    if _timer_enabled or not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        if ctypes.windll.winmm.timeBeginPeriod(1) == 0:
            _timer_enabled = True
    except Exception as e:
        log.debug("timeBeginPeriod failed: %s", e)


def disable_high_res_timer() -> None:
    global _timer_enabled
    if not _timer_enabled:
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass
    _timer_enabled = False


def set_priority(raise_it: bool) -> bool:
    try:
        import psutil
        p = psutil.Process()
        if sys.platform.startswith("win"):
            p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS if raise_it else psutil.NORMAL_PRIORITY_CLASS)
        elif raise_it:
            return False  # raising priority needs root on Linux/macOS; skip quietly
        return True
    except Exception as e:
        log.debug("priority change failed: %s", e)
        return False


def precise_sleep(seconds: float, stop=None) -> bool:
    """Sleep with ~1 ms accuracy. Returns True if `stop` (an Event) got set."""
    end = time.perf_counter() + max(0.0, seconds)
    while True:
        if stop is not None and stop.is_set():
            return True
        left = end - time.perf_counter()
        if left <= 0:
            return False
        time.sleep(min(left, 0.02))
