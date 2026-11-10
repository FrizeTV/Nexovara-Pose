"""Minimal, dependency-free OSC 1.0 sender over UDP.

Only what we need: int32, float32, string, bool. Writing it ourselves keeps the
attack surface tiny and avoids a third-party dependency. The only input ever
read is the Nexovara Pose driver's fixed-size headset feedback, accepted solely from
the exact address/port we send to (127.0.0.1) and strictly validated.
"""
from __future__ import annotations

import ipaddress
import logging
import math
import socket
import struct
import time
from typing import Iterable, Union

log = logging.getLogger(__name__)

OscArg = Union[int, float, str, bool]


def _pad(b: bytes) -> bytes:
    return b + b"\x00" * (4 - (len(b) % 4))  # always at least one NUL


def _osc_string(s: str) -> bytes:
    return _pad(s.encode("utf-8"))


def encode_message(address: str, args: Iterable[OscArg] = ()) -> bytes:
    if not address.startswith("/"):
        raise ValueError("OSC address must start with '/'")
    tags = [","]
    payload = []
    for a in args:
        if isinstance(a, bool):
            tags.append("T" if a else "F")
        elif isinstance(a, int):
            tags.append("i")
            payload.append(struct.pack(">i", a))
        elif isinstance(a, float):
            if not math.isfinite(a):
                raise ValueError("refusing to send non-finite float")
            tags.append("f")
            payload.append(struct.pack(">f", a))
        elif isinstance(a, str):
            tags.append("s")
            payload.append(_osc_string(a))
        else:
            raise TypeError(f"unsupported OSC arg type {type(a)!r}")
    return _osc_string(address) + _osc_string("".join(tags)) + b"".join(payload)


LOOPBACK = "127.0.0.1"


class OscClient:
    """Fire-and-forget UDP OSC client. PC-only app: the destination is ALWAYS
    this computer (127.0.0.1) — tracking data can never leave the PC."""

    def __init__(self, port: int, ip: str = LOOPBACK):
        addr = ipaddress.ip_address(ip)
        if not addr.is_loopback:
            raise ValueError("Nexovara Pose sends only to this computer (127.0.0.1)")
        if not (1 <= int(port) <= 65535):
            raise ValueError("bad port")
        self._dest = (str(addr), int(port))
        self._sock = socket.socket(socket.AF_INET6 if addr.version == 6 else socket.AF_INET,
                                   socket.SOCK_DGRAM)
        # Send FROM the loopback interface too: the packets stay inside the PC and
        # never touch a network adapter, so VPN clients (TAP/WireGuard/full-tunnel,
        # "kill switch") and proxies can't reroute or block them. Proxies don't
        # apply to UDP anyway.
        try:
            self._sock.bind((str(addr), 0))
        except OSError as e:
            log.warning("could not bind OSC socket to %s: %s", addr, e)
        self._sock.setblocking(False)
        self._last_err = 0.0
        self.sent = 0

    def send(self, address: str, *args: OscArg) -> None:
        try:
            self._sock.sendto(encode_message(address, args), self._dest)
            self.sent += 1
        except (OSError, ValueError) as e:  # never let networking crash tracking
            now = time.monotonic()
            if now - self._last_err > 5.0:
                log.warning("OSC send to %s failed: %s", self._dest, e)
                self._last_err = now

    def send_raw(self, data: bytes) -> None:
        try:
            self._sock.sendto(data, self._dest)
            self.sent += 1
        except OSError as e:
            now = time.monotonic()
            if now - self._last_err > 5.0:
                log.warning("UDP send to %s failed: %s", self._dest, e)
                self._last_err = now

    def receive(self, max_packets: int = 64, max_size: int = 512) -> list:
        """Datagrams waiting on this socket that came from the destination
        (127.0.0.1:port) — e.g. the Nexovara Pose driver's headset feedback. Anything
        from another address/port is dropped. Never blocks."""
        out = []
        for _ in range(max_packets):
            try:
                data, src = self._sock.recvfrom(max_size + 1)
            except (BlockingIOError, InterruptedError):
                break
            except OSError:   # e.g. Windows reports ICMP "port unreachable" here
                continue
            if (src[0], src[1]) == self._dest and len(data) <= max_size:
                out.append(data)
        return out

    @property
    def socket(self):
        return self._sock

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass
