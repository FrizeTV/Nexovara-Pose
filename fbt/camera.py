"""Camera capture + automatic mode probing.

* CameraSource: threaded capture that always hands out the NEWEST frame
  (the reader thread drains the driver buffer, so we never process stale
  frames — the single biggest latency win with webcams).
* probe_camera(): tries the common resolutions/FPS/pixel formats, MEASURES the
  real frame rate of each and picks the best mode for body tracking.
* enumerate_devices()/find_device(): camera NAMES from the OS (without opening
  the cameras) — shown everywhere in the UI and used to find "your" camera
  again when its number changes after plugging devices in or out.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".wmv"}
PROBE_SIZES = [(1920, 1080), (1280, 720), (960, 540), (848, 480), (640, 480)]


def _backends() -> List[int]:
    if sys.platform.startswith("win"):
        # DirectShow opens fast and honours MJPG/FPS requests on most webcams
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def _fourcc_str(v: float) -> str:
    """FOURCC code as text. Some DirectShow drivers report garbage here (the log
    showed "}ë6ä"): only plain ASCII letters/digits/spaces count as a format name."""
    try:
        i = int(v) & 0xFFFFFFFF
    except (TypeError, ValueError, OverflowError):
        return "?"
    s = "".join(chr((i >> (8 * k)) & 0xFF) for k in range(4))
    ok = all(c.isascii() and (c.isalnum() or c == " ") for c in s) and s.strip()
    return s if ok else "?"


def is_video_file(source: str) -> bool:
    return not str(source).strip().isdigit()


# ------------------------------------------------------------ device safety
# All camera DEVICE operations (listing, opening, configuring, closing) go
# through one lock: DirectShow / Media Foundation device lists and capture
# graphs built at the same time from several threads (main camera, extra
# cameras, camera scan) are a known source of instability in camera drivers,
# especially while Windows is adding or removing a USB device.
_device_lock = threading.RLock()
_device_change_t = -1e9
SETTLE_S = 4.0       # after a device change, leave the cameras alone this long


def device_lock() -> threading.RLock:
    return _device_lock


def notify_device_change() -> None:
    """A USB/camera device was plugged in or out (Windows WM_DEVICECHANGE).
    For SETTLE_S seconds no camera is listed or opened: a driver that is still
    being installed is the most fragile moment — touching it then can crash the
    camera driver (and with it the program or even Windows)."""
    global _device_change_t
    _device_change_t = time.monotonic()


def devices_settling(quiet_s: Optional[float] = None) -> bool:
    return time.monotonic() - _device_change_t < (SETTLE_S if quiet_s is None else quiet_s)


def wait_devices_settled(timeout: float = 10.0, stop: Optional[Callable[[], bool]] = None
                         ) -> None:
    end = time.monotonic() + timeout
    while devices_settling() and time.monotonic() < end:
        if stop is not None and stop():
            return
        time.sleep(0.1)


# Which cameras THIS program holds open right now. A new capture of the same
# camera waits until the old one is really closed: after a restart (settings
# changed, Stop -> Start) the previous camera thread may still be finishing its
# last frame, and DirectShow then "opens" the busy camera but never delivers a
# frame ("Камера открылась, но не отдаёт кадры").
_claim_cv = threading.Condition()
_claimed: Dict[str, int] = {}


def _claim_key(index: int, name: str) -> str:
    return name or f"#{index}"


def wait_camera_free(key: str, timeout: float = 8.0,
                     stop: Optional[Callable[[], bool]] = None) -> bool:
    end = time.monotonic() + timeout
    with _claim_cv:
        while _claimed.get(key, 0) > 0:
            left = end - time.monotonic()
            if left <= 0 or (stop is not None and stop()):
                return False
            _claim_cv.wait(min(left, 0.1))
    return True


def _claim(key: str) -> None:
    with _claim_cv:
        _claimed[key] = _claimed.get(key, 0) + 1


def _unclaim(key: Optional[str]) -> None:
    if key is None:
        return
    with _claim_cv:
        n = _claimed.get(key, 0) - 1
        if n > 0:
            _claimed[key] = n
        else:
            _claimed.pop(key, None)
        _claim_cv.notify_all()


def release_capture(cap) -> None:
    if cap is None:
        return
    try:
        with _device_lock:
            cap.release()
    except Exception as e:
        log.debug("camera release failed: %s", e)


# ------------------------------------------------------------ camera names
@dataclass(frozen=True)
class CameraDevice:
    index: int          # index for cv2.VideoCapture with the first backend of _backends()
    name: str           # human-readable name from the OS ("Logitech C920", "DroidCam Source 3")

    def label(self) -> str:
        return f"{self.name} (#{self.index})"


_NAMES_TTL_S = 3.0
_names_lock = threading.Lock()
_names_cache: Tuple[float, Optional[List[CameraDevice]]] = (-1e9, None)


def _clean_name(name, index: int) -> str:
    n = "".join(ch for ch in str(name or "") if ch.isprintable()).strip()[:120]
    return n or f"Камера #{index}"


def _enumerate_os(backend: Optional[int] = None) -> Optional[List[CameraDevice]]:
    """Camera names straight from the OS, without opening the cameras.

    Windows: DirectShow / Media Foundation device lists (the package
    `cv2-enumerate-cameras` returns them in the same order OpenCV uses for the
    camera index). Linux: V4L2 capture devices with their names from sysfs.
    Returns None when names can't be obtained on this system."""
    if sys.platform.startswith("win"):
        try:
            from cv2_enumerate_cameras import enumerate_cameras
        except Exception as e:  # not installed (old .venv) -> fall back to plain numbers
            log.info("camera names unavailable: %s", e)
            return None
        try:
            with _device_lock:
                infos = enumerate_cameras(backend if backend is not None else _backends()[0])
        except Exception as e:
            log.warning("camera enumeration failed: %s", e)
            return None
        return [CameraDevice(int(i.index), _clean_name(i.name, int(i.index))) for i in infos]
    if sys.platform.startswith("linux"):
        return _enumerate_linux()
    return None


def _enumerate_linux() -> Optional[List[CameraDevice]]:
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return None
    out = []
    for dev in os.listdir(base):
        if not dev.startswith("video") or not dev[5:].isdigit():
            continue
        idx = int(dev[5:])
        try:
            # metadata nodes of UVC cameras have index 1 (the capture node has 0)
            with open(os.path.join(base, dev, "index"), encoding="utf-8") as f:
                if f.read().strip() not in ("", "0"):
                    continue
        except OSError:
            pass
        name = ""
        try:
            with open(os.path.join(base, dev, "name"), encoding="utf-8", errors="replace") as f:
                name = f.readline()
        except OSError:
            pass
        out.append(CameraDevice(idx, _clean_name(name, idx)))
    return sorted(out, key=lambda d: d.index)


def enumerate_devices(max_age_s: float = _NAMES_TTL_S) -> Optional[List[CameraDevice]]:
    """Cached OS camera list (cheap; never opens a camera). None = names unknown."""
    global _names_cache
    if devices_settling():
        t, devs = _names_cache
        if t > -1e8:
            # Windows is installing/removing a device right now: use what we know
            return list(devs) if devs is not None else None
        if threading.current_thread() is threading.main_thread():
            return None                     # never block the window
        wait_devices_settled()
    with _names_lock:
        t, devs = _names_cache
        now = time.monotonic()
        if now - t <= max_age_s:
            return list(devs) if devs is not None else None
        devs = _enumerate_os()
        _names_cache = (now, devs)
        return list(devs) if devs is not None else None


def device_name(source: str) -> str:
    """Name of a camera index from the (cached) OS list, "" if unknown."""
    src = str(source).strip()
    if not src.isdigit():
        return ""
    for d in enumerate_devices() or []:
        if d.index == int(src):
            return d.name
    return ""


def camera_label(source: str, name: str = "") -> str:
    """How a camera source is shown to the user everywhere in the app."""
    src = str(source).strip()
    if src.lower().endswith(".npz"):
        return f"Запись: {os.path.basename(src) or src}"
    if is_video_file(src):
        return f"Видео: {os.path.basename(src) or src}"
    name = (name or "").strip()
    return f"{name} (#{src})" if name and name != f"Камера #{src}" else f"Камера #{src}"


def short_label(source: str, name: str = "", limit: int = 34) -> str:
    s = camera_label(source, name)
    return s if len(s) <= limit else s[:limit - 1] + "…"


def find_device(devs: Optional[List[CameraDevice]], source: str, name: str = ""
                ) -> Tuple[Optional[CameraDevice], str]:
    """Locate the configured camera in the current OS list.

    Indices change when cameras are plugged in/out, the name does not.
    Returns (device, status): "ok" — same index (and same name if one was
    remembered); "moved" — the remembered camera now has another index;
    "missing" — not connected; "unknown" — names unavailable / video file."""
    src = str(source).strip()
    if devs is None or not src.isdigit():
        return None, "unknown"
    idx = int(src)
    at = next((d for d in devs if d.index == idx), None)
    name = (name or "").strip()
    if name:
        if at is not None and at.name == name:
            return at, "ok"
        same = [d for d in devs if d.name == name]
        if same:
            return same[0], "moved"
        return None, "missing"
    return (at, "ok") if at is not None else (None, "missing")


def open_capture(index: int) -> Tuple[cv2.VideoCapture, int]:
    wait_devices_settled()
    name = device_name(str(index))      # before the device lock (lock order: names -> device)
    with _device_lock:
        return _open_capture(index, name)


def _open_capture(index: int, name: str) -> Tuple[cv2.VideoCapture, int]:
    backends = _backends()
    for b in backends:
        idx = index
        if name and sys.platform.startswith("win") and b == cv2.CAP_MSMF:
            # Media Foundation may number the cameras differently than DirectShow:
            # open the same device by its name, never a different camera.
            msmf = _enumerate_os(cv2.CAP_MSMF)
            if msmf is not None:
                hit = next((d.index for d in msmf if d.name == name), None)
                if hit is None:
                    continue
                idx = hit
        elif name and b == cv2.CAP_ANY and len(backends) > 2:
            continue  # CAP_ANY could pick another device with this number
        cap = cv2.VideoCapture(idx, b)
        if cap.isOpened():
            return cap, b
        cap.release()
    raise RuntimeError(f"Не удалось открыть камеру «{camera_label(str(index), name)}». "
                       "Проверьте, что она подключена и не занята другой программой "
                       "(Discord, OBS, Zoom, браузер), или выберите другую камеру: "
                       "Настройки → Камера.")


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int,
                      fourcc: str) -> None:
    if fourcc in ("MJPG", "YUY2"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if fourcc in ("MJPG", "YUY2"):
        # some drivers only accept the format after the size was set
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def list_cameras(max_index: int = 6) -> List[int]:
    """Camera indices: from the OS list when available, otherwise by probing."""
    devs = enumerate_devices(0)
    if devs:
        return [d.index for d in devs]
    return _probe_indices(max_index)


def list_camera_devices(max_index: int = 6, allow_open: bool = True
                        ) -> Optional[List[CameraDevice]]:
    """Cameras with names. Without OS names, falls back to opening the indices
    (only if allow_open — not while tracking uses a camera); None = unknown."""
    devs = enumerate_devices(0)
    if devs:
        return devs
    if not allow_open:
        return devs
    # no names, or the OS list is empty (double-check: a camera may still open)
    return [CameraDevice(i, device_name(str(i)) or f"Камера #{i}")
            for i in _probe_indices(max_index)]


def _probe_indices(max_index: int) -> List[int]:
    """Probe camera indices that can actually deliver a frame."""
    found = []
    wait_devices_settled()
    with _device_lock:
        for i in range(max_index):
            cap = None
            try:
                cap = cv2.VideoCapture(i, _backends()[0])
                if cap.isOpened():
                    ok, _ = cap.read()
                    if ok:
                        found.append(i)
            except cv2.error:
                pass
            finally:
                if cap is not None:
                    cap.release()
    return found


# ------------------------------------------------------------------ probing
@dataclass
class CameraMode:
    width: int
    height: int
    fps: float          # measured
    fps_requested: int
    fourcc: str

    def label(self) -> str:
        return f"{self.width}×{self.height} @ {self.fps:.0f} fps ({self.fourcc})"


def _measure(cap, seconds: float, stop) -> Tuple[float, Optional[Tuple[int, int]]]:
    size = None
    for _ in range(4):  # warm-up: drivers often deliver the first frames slowly
        ok, f = cap.read()
        if ok and f is not None:
            size = (f.shape[1], f.shape[0])
    if size is None:
        return 0.0, None
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        if stop is not None and stop():
            break
        ok, f = cap.read()
        if not ok or f is None:
            break
        n += 1
        size = (f.shape[1], f.shape[0])
    dt = time.perf_counter() - t0
    return (n / dt if dt > 0 else 0.0), size


def choose_best_mode(modes: List[CameraMode], prefer_hd: bool = False) -> Optional[CameraMode]:
    """Frame rate matters most for tracking (latency + smoothness); the model
    itself runs at 256 px, so beyond 720p resolution adds decode cost only.
    Among modes within ~10% of the best FPS prefer 1280×720 (or 960×540 when
    fingers are off and the PC is weak), then the closest smaller one."""
    good = [m for m in modes if m.fps >= 5]
    if not good:
        return None
    top = max(min(m.fps, 90.0) for m in good)
    fast = [m for m in good if min(m.fps, 90.0) >= top * 0.9]
    target = 1280 * 720 if prefer_hd else 960 * 540

    def score(m: CameraMode):
        px = m.width * m.height
        over = px > 1280 * 720
        # same size and speed: MJPG — compressed, ~5x less USB bandwidth than YUY2,
        # so a second camera / USB device on the same controller keeps working
        return (over, abs(px - target), -round(min(m.fps, 90.0)), m.fourcc != "MJPG", -m.fps)
    return sorted(fast, key=score)[0]


def probe_camera(index: int, progress: Optional[Callable[[float, str], None]] = None,
                 stop: Optional[Callable[[], bool]] = None, per_mode_s: float = 0.7
                 ) -> Dict:
    """Measure what the camera can really do. Returns a JSON-friendly dict."""
    cap, backend = open_capture(index)
    modes: List[CameraMode] = []
    seen = set()
    try:
        plan = [(w, h, fcc) for (w, h) in PROBE_SIZES for fcc in ("MJPG", "YUY2")]
        for i, (w, h, fcc) in enumerate(plan):
            if stop is not None and stop():
                break
            if progress:
                progress(i / len(plan), f"Проверка {w}×{h} {fcc}…")
            with _device_lock:
                configure_capture(cap, w, h, 60, fcc)
            real_fcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            fps, size = _measure(cap, per_mode_s, stop)
            if size is None:
                continue
            key = (size, real_fcc)
            if key in seen:
                continue
            seen.add(key)
            modes.append(CameraMode(size[0], size[1], round(fps, 1), 60, real_fcc))
            # 60 not honoured? many cams only reach 60 at 30 requested -> also try 30
            if fps < 25:
                with _device_lock:
                    configure_capture(cap, w, h, 30, fcc)
                fps30, size30 = _measure(cap, per_mode_s, stop)
                if size30 and fps30 > fps * 1.2:
                    modes[-1] = CameraMode(size30[0], size30[1], round(fps30, 1), 30, real_fcc)
    finally:
        release_capture(cap)
    if progress:
        progress(1.0, "Готово")
    return {
        "camera": str(index),
        "name": device_name(str(index)),
        "backend": int(backend),
        "modes": [asdict(m) for m in modes],
        "time": int(time.time()),
    }


def profile_for(profiles: Dict, source: str, name: str = "") -> Optional[Dict]:
    """Stored probe result for a camera — only if it was made with THIS camera
    (a profile stored under an index that now belongs to another device is ignored)."""
    prof = (profiles or {}).get(str(source))
    if not isinstance(prof, dict):
        return None
    pname = str(prof.get("name") or "")
    if name and pname and pname != name:
        return None
    return prof


def best_from_profile(profile: Dict, prefer_hd: bool = False) -> Optional[CameraMode]:
    try:
        modes = [CameraMode(**m) for m in profile.get("modes", [])]
    except TypeError:
        return None
    return choose_best_mode(modes, prefer_hd)


# ------------------------------------------------------------------ capture
def _short_path(path: str) -> str:
    """Windows 8.3 short path (ASCII) for a long path, "" if unavailable."""
    if not sys.platform.startswith("win"):
        return ""
    try:
        import ctypes
        from ctypes import wintypes
        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        buf = ctypes.create_unicode_buffer(1024)
        n = fn(path, buf, 1024)
        return buf.value if 0 < n < 1024 else ""
    except Exception:
        return ""


class CameraSource:
    """Threaded capture. The reader thread OWNS the capture object: only it reads,
    and only it releases it. (Before, stop() released the capture while the
    reader could still be inside read() — e.g. blocked on a camera that was just
    unplugged or re-plugged — a use-after-free in the camera backend that could
    crash the program.)

    Live cameras survive being unplugged and plugged in again: after a few
    failed reads the capture is released, and once Windows has finished
    installing the device (device changes settle, see notify_device_change),
    the SAME camera is looked up by its name (its number may have changed) and
    reopened. Meanwhile `lost` is True and no frames arrive."""

    LOST_AFTER_FAILS = 25          # consecutive failed reads -> treat as unplugged
    RETRY_S = 2.0                  # how often to try reopening a lost camera
    STOP_JOIN_S = 3.0

    def __init__(self, source: str, width: int, height: int, fps: int, fourcc: str = "auto",
                 name: str = ""):
        self.source = str(source).strip()
        self.name = str(name or "")
        self.width, self.height, self.fps, self.fourcc = width, height, fps, fourcc
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_file = False
        self._file_fps = 30.0
        self._cond = threading.Condition()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.lost = False              # live camera unplugged / not delivering, reconnecting
        self.open_error: Optional[str] = None   # why the camera isn't delivering (for the UI)
        self._claim: Optional[str] = None
        self.reconnects = 0
        self.measured_fps = 0.0
        self.actual_size: Tuple[int, int] = (0, 0)
        self.actual_fourcc = "?"

    FIRST_FRAME_S = 3.0            # a camera may need a moment after switching modes

    def _open_live(self, index: int) -> cv2.VideoCapture:
        key = _claim_key(index, self.name or device_name(str(index)))
        if not wait_camera_free(key, 8.0, self._stop.is_set):
            raise RuntimeError("Камера ещё занята предыдущим запуском трекинга")
        cap, _ = open_capture(index)
        try:
            fcc = self.fourcc if self.fourcc != "auto" else "MJPG"
            with device_lock():
                configure_capture(cap, self.width, self.height, self.fps, fcc)
            self.actual_fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            frame = None
            end = time.monotonic() + self.FIRST_FRAME_S
            while time.monotonic() < end and not self._stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    break
                frame = None
                time.sleep(0.05)
            if frame is None:
                raise RuntimeError("Камера открылась, но не отдаёт кадры — её занимает другая "
                                   "программа (браузер, Discord, OBS, Zoom, ещё одна копия "
                                   "Nexovara Pose) или драйвер камеры завис")
        except Exception:
            release_capture(cap)
            raise
        _claim(key)
        self._claim = key
        self.actual_size = (frame.shape[1], frame.shape[0])
        return cap

    def _drop_cap(self) -> None:
        """Close the capture (only from the reader thread, or before it started)."""
        cap, self._cap = self._cap, None
        if cap is not None:
            release_capture(cap)
        key, self._claim = self._claim, None
        _unclaim(key)

    def open(self) -> None:
        src = self.source
        if src.isdigit():
            try:
                self._cap = self._open_live(int(src))
            except RuntimeError as e:
                # once more, a moment later: the previous capture may still have been
                # closing, or the camera got another number (a headset / USB device
                # was plugged in) — then it is found again by its name
                log.info("camera %s: %s — trying again", self.name or src, e)
                if self._stop.wait(1.0):
                    raise
                idx = int(src)
                if self.name:
                    dev, _st = find_device(enumerate_devices(0), src, self.name)
                    if dev is not None:
                        idx = dev.index
                self._cap = self._open_live(idx)
        else:
            path = os.path.abspath(src)
            if path.lower().endswith(".npz"):
                raise RuntimeError("Выбрана запись позы, а не камера — для этого действия "
                                   "выберите камеру (Настройки → Камера)")
            if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in VIDEO_EXTS:
                raise RuntimeError("Источник должен быть номером камеры или видеофайлом "
                                   f"({', '.join(sorted(VIDEO_EXTS))})")
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened() and not path.isascii():
                # OpenCV on Windows may not open non-ASCII paths: retry via the 8.3 name
                short = _short_path(path)
                if short and short != path:
                    self._cap.release()
                    self._cap = cv2.VideoCapture(short)
            if not self._cap.isOpened():
                raise RuntimeError("Не удалось открыть видеофайл"
                                   + ("" if path.isascii() else
                                      " (попробуйте путь без русских букв)"))
            self._is_file = True
            f = self._cap.get(cv2.CAP_PROP_FPS)
            self._file_fps = f if f and 1 <= f <= 240 else 30.0
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self._cap.release()
                self._cap = None
                raise RuntimeError("Не удалось прочитать видеофайл")
            self.actual_size = (frame.shape[1], frame.shape[0])
        log.info("Camera opened: %s %dx%d %s", "file" if self._is_file else src,
                 *self.actual_size, self.actual_fourcc)

    def start(self, wait: bool = False) -> None:
        """wait=True (tracking): a live camera that can't deliver frames right now
        (busy, still being released, re-plugged) does NOT stop tracking with an
        error — the camera is waited for and opened as soon as it works."""
        if self._cap is None:
            try:
                self.open()
            except Exception as e:
                if not (wait and self.source.isdigit()):
                    raise
                log.warning("camera %s not available yet: %s", self.name or self.source, e)
                self.open_error = str(e)
                self.lost = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _reconnect(self) -> bool:
        """Lost live camera: wait until device changes settle, find the camera by
        name (index may have changed), reopen. True when frames flow again."""
        while not self._stop.is_set():
            if self._stop.wait(self.RETRY_S):
                return False
            if devices_settling():
                continue                       # Windows is still installing a device
            idx = int(self.source)
            devs = enumerate_devices(0)
            if devs is not None:
                dev, st = find_device(devs, self.source, self.name)
                if dev is None:
                    continue                   # not plugged in (yet)
                idx = dev.index
            try:
                cap = self._open_live(idx)
            except Exception as e:
                log.info("camera %s not back yet: %s", self.source, e)
                self.open_error = str(e)
                continue
            self._cap = cap
            if self._stop.is_set():
                self._drop_cap()
                return False
            self.lost = False
            self.open_error = None
            self.reconnects += 1
            log.info("camera %s reconnected (index %d)", self.name or self.source, idx)
            return True
        return False

    def _loop(self) -> None:
        try:
            if self.lost and not self._reconnect():    # started without a working camera
                return
            self._loop_inner()
        except Exception as e:                 # never let the reader die silently
            log.exception("camera reader crashed")
            self.error = f"Ошибка камеры: {e}"
        finally:
            self._drop_cap()                   # released by the thread that reads it
            with self._cond:
                self._cond.notify_all()

    def _loop_inner(self) -> None:
        fails = 0
        frames, t0 = 0, time.monotonic()
        period = 1.0 / self._file_fps
        next_t = time.monotonic()
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self._is_file:
                    fails += 1
                    if fails > 3:
                        self.error = "Не удалось прочитать видеофайл"
                        break
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    continue
                fails += 1
                if fails >= self.LOST_AFTER_FAILS:
                    # unplugged (or the driver hiccuped): let go of the device at once —
                    # never keep hammering a device Windows is removing/reinstalling
                    log.warning("camera %s stopped delivering frames — reconnecting",
                                self.name or self.source)
                    self.lost = True
                    self.open_error = "камера перестала отдавать кадры (отключена?)"
                    self.measured_fps = 0.0
                    self._drop_cap()
                    with self._cond:
                        self._cond.notify_all()
                    if not self._reconnect():
                        break
                    fails = 0
                    continue
                time.sleep(0.02)
                continue
            fails = 0
            now = time.monotonic()
            with self._cond:
                self._frame = frame
                self._seq += 1
                self._ts = now
                self._cond.notify_all()
            frames += 1
            if now - t0 >= 1.0:
                self.measured_fps = frames / (now - t0)
                frames, t0 = 0, now
            if self._is_file:  # play files at real speed
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_t = time.monotonic()

    def get_latest(self, last_seq: int, timeout: float = 0.5
                   ) -> Optional[Tuple[int, np.ndarray, float]]:
        with self._cond:
            self._cond.wait_for(lambda: self._seq != last_seq or self._stop.is_set()
                                or self.error is not None, timeout)
            if self._seq == last_seq or self._frame is None:
                return None
            return self._seq, self._frame, self._ts

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.STOP_JOIN_S)
            if self._thread.is_alive():
                # the reader is stuck inside the driver (e.g. a device being removed):
                # it releases the capture itself when read() returns — releasing it
                # from here would free memory the driver is still using
                log.warning("camera reader still busy in the driver; it will close itself")
                return
        elif self._cap is not None:            # opened but never started
            self._drop_cap()
