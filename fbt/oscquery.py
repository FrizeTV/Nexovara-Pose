"""The headset pose from VRChat itself, via OSCQuery (optional, off by default).

VRChat (since 2023.3.1) finds OSC apps on this PC through OSCQuery: an app
announces two services over mDNS/Zeroconf — "_oscjson._tcp" (a tiny HTTP server
describing which OSC addresses it wants) and "_osc._udp" (where to send them).
If the app's address tree contains /tracking/vrsystem, VRChat sends it the VR
system's head and wrist poses — after the user accepted an extra notice in
VRChat (VRChat shows a message when it starts sending).
Sources: VRChat OSCQuery wiki (github.com/vrchat-community/osc/wiki/OSCQuery),
vrc-oscquery-lib (Attributes.cs, HostInfo.cs), VrcAdvert (endpoint types),
SlimeVR's VRCOSCHandler (argument order of /tracking/vrsystem/*/pose).

Message: /tracking/vrsystem/head/pose  f f f f f f
         = position x y z (Unity, metres) + rotation x y z (Unity euler, degrees).

With the real headset pose Nexovara Pose glues the body to the head in VRChat OSC
mode as well (see hmd.BodyAnchor, sender.OutputSender) and sends VRChat the
exact head position/heading instead of the camera's noisy one.

Safety:
  * the HTTP server and the UDP receiver listen on 127.0.0.1 only; the HTTP
    server answers GET with two fixed JSON documents and nothing else;
  * UDP datagrams are accepted only from 127.0.0.1, are size-limited and
    strictly parsed (no code, no files);
  * the mDNS announcement carries only the name "Nexovara Pose-xxxx", the service type,
    the ports and the address 127.0.0.1.
"""
from __future__ import annotations

import json
import logging
import math
import secrets
import socket
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import numpy as np

from .hmd import HmdState
from .mathutil import unity_euler_to_mat

log = logging.getLogger(__name__)

HEAD_POSE = "/tracking/vrsystem/head/pose"
WRIST_POSES = ("/tracking/vrsystem/leftwrist/pose", "/tracking/vrsystem/rightwrist/pose")
ENDPOINTS = (HEAD_POSE, *WRIST_POSES)
MAX_DATAGRAM = 2048
ACCESS_WRITE_ONLY = 2


# ---------------------------------------------------------------- OSC parsing
def _read_string(data: bytes, i: int) -> Tuple[str, int]:
    end = data.find(b"\x00", i)
    if end < 0 or end - i > 256:
        raise ValueError("bad OSC string")
    s = data[i:end].decode("ascii")
    i = end + 1
    i += (-i) % 4
    if i > len(data):
        raise ValueError("bad OSC padding")
    return s, i


def parse_message(data: bytes) -> Tuple[str, list]:
    addr, i = _read_string(data, 0)
    if not addr.startswith("/"):
        raise ValueError("bad OSC address")
    tags, i = _read_string(data, i)
    if not tags.startswith(",") or len(tags) > 33:
        raise ValueError("bad OSC type tags")
    args: list = []
    for t in tags[1:]:
        if t == "f":
            if i + 4 > len(data):
                raise ValueError("short OSC message")
            args.append(struct.unpack(">f", data[i:i + 4])[0])
            i += 4
        elif t == "i":
            if i + 4 > len(data):
                raise ValueError("short OSC message")
            args.append(struct.unpack(">i", data[i:i + 4])[0])
            i += 4
        elif t == "s":
            s, i = _read_string(data, i)
            args.append(s)
        elif t in "TF":
            args.append(t == "T")
        else:
            raise ValueError(f"unsupported OSC type {t!r}")
    return addr, args


def parse_packet(data: bytes, depth: int = 0) -> List[Tuple[str, list]]:
    """Messages of an OSC packet (message or bundle; bundles max. 2 levels deep)."""
    if len(data) > MAX_DATAGRAM or len(data) < 4:
        raise ValueError("bad OSC packet size")
    if data.startswith(b"#bundle\x00"):
        if depth >= 2 or len(data) < 16:
            raise ValueError("bad OSC bundle")
        out: List[Tuple[str, list]] = []
        i = 16                                       # "#bundle\0" + time tag
        while i < len(data):
            if i + 4 > len(data):
                raise ValueError("bad OSC bundle element")
            n = struct.unpack(">i", data[i:i + 4])[0]
            i += 4
            if n <= 0 or n % 4 or i + n > len(data):
                raise ValueError("bad OSC bundle element size")
            out.extend(parse_packet(data[i:i + n], depth + 1))
            i += n
            if len(out) > 64:
                raise ValueError("too many OSC messages")
        return out
    return [parse_message(data)]


def pose_from_args(args: list) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """6 floats -> (position, 3x3 rotation), Unity axes; None if not plausible."""
    if len(args) != 6 or not all(isinstance(a, float) for a in args):
        return None
    if not all(math.isfinite(a) for a in args):
        return None
    px, py, pz, rx, ry, rz = args
    if max(abs(px), abs(py), abs(pz)) > 1000.0 or max(abs(rx), abs(ry), abs(rz)) > 1e4:
        return None
    return np.array([px, py, pz], dtype=np.float64), unity_euler_to_mat(rx, ry, rz)


# ---------------------------------------------------------------- OSCQuery documents
def _node_tree(endpoints) -> Dict:
    root = {"DESCRIPTION": "root node", "FULL_PATH": "/", "ACCESS": 0, "CONTENTS": {}}
    for ep in endpoints:
        node = root
        parts = ep.strip("/").split("/")
        for k, part in enumerate(parts):
            path = "/" + "/".join(parts[:k + 1])
            node = node.setdefault("CONTENTS", {}).setdefault(
                part, {"FULL_PATH": path, "ACCESS": 0, "CONTENTS": {}})
        node.pop("CONTENTS", None)
        node.update({"FULL_PATH": ep, "ACCESS": ACCESS_WRITE_ONLY, "TYPE": "ffffff",
                     "DESCRIPTION": "VR system pose (position xyz, rotation xyz)"})
    return root


def find_node(tree: Dict, path: str) -> Optional[Dict]:
    path = "/" + path.strip("/")
    if path == "/":
        return tree
    node = tree
    for part in path.strip("/").split("/"):
        node = (node.get("CONTENTS") or {}).get(part)
        if node is None:
            return None
    return node


def host_info(name: str, osc_port: int) -> Dict:
    return {"NAME": name,
            "EXTENSIONS": {"ACCESS": True, "CLIPMODE": False, "RANGE": True, "TYPE": True,
                           "VALUE": True},
            "OSC_IP": "127.0.0.1", "OSC_PORT": int(osc_port), "OSC_TRANSPORT": "UDP"}


def _make_handler(service: "OscQueryService"):
    class Handler(BaseHTTPRequestHandler):
        server_version = "Nexovara Pose-OSCQuery"
        sys_version = ""
        timeout = 5

        def log_message(self, *_a):          # no request logging
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def do_GET(self):  # noqa: N802 (http.server naming)
            if len(self.path) > 512:
                self._send(414, b"", "text/plain")
                return
            u = urlsplit(self.path)
            service.http_requests += 1
            if "HOST_INFO" in u.query:
                body = json.dumps(service.host_info).encode()
                self._send(200, body, "application/json")
                return
            node = find_node(service.tree, u.path or "/")
            if node is None:
                self._send(404, b"OSC Path not found", "text/plain")
                return
            self._send(200, json.dumps(node).encode(), "application/json")

        do_HEAD = do_GET

        def do_POST(self):  # noqa: N802
            self._send(405, b"", "text/plain")
        do_PUT = do_DELETE = do_POST
    return Handler


class OscQueryService:
    """Receiver of VRChat's /tracking/vrsystem poses + its OSCQuery announcement."""

    def __init__(self, state: HmdState, name: Optional[str] = None, advertise: bool = True):
        self.state = state
        self.name = name or f"Nexovara Pose-{secrets.token_hex(2).upper()}"
        self.advertise = advertise
        self.status = "запуск"
        self.packets = 0
        self.poses = 0
        self.http_requests = 0
        self.last_pose_t = 0.0
        self._stop = threading.Event()
        self._udp: Optional[socket.socket] = None
        self._http: Optional[ThreadingHTTPServer] = None
        self._zc = None
        self._threads: List[threading.Thread] = []
        self.osc_port = 0
        self.http_port = 0
        self.tree = _node_tree(ENDPOINTS)
        self.host_info: Dict = {}

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._udp.bind(("127.0.0.1", 0))
        self._udp.settimeout(0.25)
        self.osc_port = self._udp.getsockname()[1]
        self.host_info = host_info(self.name, self.osc_port)
        self._http = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self._http.daemon_threads = True
        self.http_port = self._http.server_address[1]
        for target, nm in ((self._recv_loop, "vrc-osc-in"), (self._http.serve_forever,
                                                              "vrc-oscquery-http")):
            t = threading.Thread(target=target, name=nm, daemon=True)
            t.start()
            self._threads.append(t)
        self.status = "ожидание VRChat"
        if self.advertise:
            t = threading.Thread(target=self._announce, name="vrc-oscquery-mdns", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        zc, self._zc = self._zc, None
        if zc is not None:
            try:
                zc.unregister_all_services()
                zc.close()
            except Exception:
                pass
        if self._http is not None:
            try:
                self._http.shutdown()
                self._http.server_close()
            except Exception:
                pass
        if self._udp is not None:
            try:
                self._udp.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2.0)

    # ------------------------------------------------------------ mDNS
    def _announce(self) -> None:
        try:
            from zeroconf import IPVersion, ServiceInfo, Zeroconf
        except ImportError:
            self.status = ("нет модуля zeroconf — запустите setup.bat, чтобы VRChat нашёл "
                           "Nexovara Pose")
            log.warning("zeroconf is not installed: VRChat can't discover Nexovara Pose")
            return
        try:
            zc = Zeroconf(ip_version=IPVersion.V4Only)
            host = f"{self.name.lower()}.local."
            addr = [socket.inet_aton("127.0.0.1")]
            infos = [
                ServiceInfo("_oscjson._tcp.local.", f"{self.name}._oscjson._tcp.local.",
                            addresses=addr, port=self.http_port, server=host, properties={}),
                ServiceInfo("_osc._udp.local.", f"{self.name}._osc._udp.local.",
                            addresses=addr, port=self.osc_port, server=host, properties={}),
            ]
            for info in infos:
                if self._stop.is_set():
                    break
                zc.register_service(info)
            if self._stop.is_set():
                zc.unregister_all_services()
                zc.close()
                return
            self._zc = zc
            if self.poses == 0:
                self.status = ("Nexovara Pose объявлен для VRChat — ожидаю позу шлема (в VRChat "
                               "должен быть включён OSC)")
            log.info("OSCQuery: announced %s (http %d, osc %d)", self.name, self.http_port,
                     self.osc_port)
        except Exception as e:
            self.status = f"не удалось объявить Nexovara Pose для VRChat: {e}"
            log.warning("OSCQuery announce failed: %s", e)

    # ------------------------------------------------------------ receiving
    def _recv_loop(self) -> None:
        while not self._stop.is_set():
            try:
                data, src = self._udp.recvfrom(MAX_DATAGRAM + 1)
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                time.sleep(0.05)
                continue
            if src[0] != "127.0.0.1" or len(data) > MAX_DATAGRAM:
                continue
            self.packets += 1
            self.handle(data)

    def handle(self, data: bytes) -> int:
        """Parse one datagram; returns how many head poses were taken."""
        try:
            msgs = parse_packet(data)
        except (ValueError, UnicodeDecodeError, struct.error):
            return 0
        n = 0
        for addr, args in msgs:
            if addr != HEAD_POSE:
                continue
            pose = pose_from_args(args)
            if pose is None:
                continue
            self.state.update(pose[0], pose[1], "vrchat")
            self.poses += 1
            self.last_pose_t = time.monotonic()
            n += 1
        if n:
            self.status = "поза шлема из VRChat получена"
        return n
