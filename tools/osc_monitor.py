"""OSC monitor: see what Nexovara Pose sends, without VRChat or SteamVR.

  python tools/osc_monitor.py [--port 9000]

Listens ONLY on 127.0.0.1 (this computer). Packets are parsed by a small,
strict, size-limited parser; anything malformed is ignored.
Do not run it while VRChat is open (VRChat uses the same port 9000).
"""
from __future__ import annotations

import argparse
import os
import socket
import struct
import sys
import time

MAX_PACKET = 4096
MAX_DEPTH = 2

TRACKER_NAMES = {  # fixed slots, same order as Nexovara Pose's tracker list
    "1": "1 таз", "2": "2 лев. стопа", "3": "3 прав. стопа", "4": "4 лев. колено",
    "5": "5 прав. колено", "6": "6 грудь", "7": "7 лев. локоть", "8": "8 прав. локоть",
    "head": "голова",
}


def _read_str(b: bytes, i: int):
    end = b.index(b"\x00", i)
    s = b[i:end].decode("utf-8", "replace")
    return s, (end + 4) & ~3


def parse(packet: bytes, depth: int = 0):
    """Yield (address, args) from an OSC message or bundle."""
    if depth > MAX_DEPTH or len(packet) < 4:
        return
    if packet.startswith(b"#bundle\x00"):
        i = 16
        while i + 4 <= len(packet):
            (n,) = struct.unpack(">i", packet[i:i + 4])
            i += 4
            if n <= 0 or i + n > len(packet):
                return
            yield from parse(packet[i:i + n], depth + 1)
            i += n
        return
    addr, i = _read_str(packet, 0)
    if not addr.startswith("/"):
        return
    tags, i = _read_str(packet, i)
    if not tags.startswith(","):
        return
    args = []
    for t in tags[1:]:
        if t == "f":
            args.append(struct.unpack(">f", packet[i:i + 4])[0]); i += 4
        elif t == "i":
            args.append(struct.unpack(">i", packet[i:i + 4])[0]); i += 4
        elif t == "s":
            s, i = _read_str(packet, i); args.append(s)
        elif t in "TF":
            args.append(t == "T")
        else:
            return
    yield addr, args


def main() -> None:
    if sys.platform.startswith("win"):
        os.system("")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    a = ap.parse_args()
    if not 1 <= a.port <= 65535:
        sys.exit("bad port")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", a.port))
    except OSError as e:
        sys.exit(f"Порт {a.port} занят ({e}).\nСкорее всего открыт VRChat — закройте его "
                 "для теста, или укажите в Nexovara Pose другой порт и запустите монитор с --port.")
    sock.settimeout(0.1)
    latest = {}
    counts = {}
    total = 0
    t_rate, rate = time.monotonic(), 0.0
    last_draw = 0.0
    print(f"Слушаю 127.0.0.1:{a.port}… Запустите Nexovara Pose и нажмите «Старт». Ctrl+C — выход.")
    try:
        while True:
            try:
                data = sock.recv(MAX_PACKET)
                try:
                    for addr, args in parse(data):
                        latest[addr] = args
                        total += 1
                        counts[addr] = counts.get(addr, 0) + 1
                except (ValueError, struct.error, IndexError):
                    pass  # malformed packet: ignore
            except socket.timeout:
                pass
            now = time.monotonic()
            if now - t_rate >= 1.0:
                rate = sum(counts.values()) / (now - t_rate)
                counts.clear()
                t_rate = now
            if now - last_draw >= 0.25:
                last_draw = now
                draw(latest, total, rate, a.port)
    except KeyboardInterrupt:
        print("\nВыход.")


def draw(latest, total, rate, port) -> None:
    lines = ["\033[H\033[2J", f"\033[1mNexovara Pose OSC монитор\033[0m — 127.0.0.1:{port}   "
             f"сообщений: {total}   ({rate:.0f}/с)\n"]
    trackers = {}
    other = []
    for addr, args in latest.items():
        p = addr.split("/")
        if len(p) == 5 and p[1] == "tracking" and p[2] == "trackers":
            trackers.setdefault(p[3], {})[p[4]] = args
        else:
            other.append((addr, args))
    if trackers:
        lines.append(f"{'Трекер':<18}{'Позиция X, Y, Z (м)':<30}{'Поворот X, Y, Z (°)'}")
        lines.append("-" * 72)
        for k in sorted(trackers, key=lambda s: (s == "head", s)):
            v = trackers[k]
            pos = ", ".join(f"{x:+.2f}" for x in v.get("position", []))
            rot = ", ".join(f"{x:6.1f}" for x in v.get("rotation", []))
            lines.append(f"{TRACKER_NAMES.get(k, k):<18}{pos:<30}{rot}")
        lines.append("\nY — высота над полом: таз ≈ 0.9–1.0 м, стопы ≈ 0.05–0.1 м, голова ≈ рост−0.1 м.")
    elif not other:
        lines.append("Данных пока нет. Проверьте: трекинг запущен, тело в кадре, "
                     f"в Nexovara Pose включено «Отправлять в VRChat», порт {port}.")
    for addr, args in sorted(other)[:20]:
        lines.append(f"{addr}  {args}")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
