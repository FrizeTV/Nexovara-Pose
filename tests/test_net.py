"""Downloads: direct connection first (bypassing a proxy), then the system
proxy; tracking traffic stays on the loopback interface."""
import io
import socket
import sys
import urllib.error
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fbt import net, pose  # noqa: E402
from fbt.osc import OscClient  # noqa: E402

URL = "https://storage.googleapis.com/mediapipe-models/x/pose_landmarker_lite.task"


def _model_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("m.tflite", bytes(range(256)) * 6000)
    return buf.getvalue()


class _Resp(io.BytesIO):
    def __init__(self, data, url, cut=None):
        super().__init__(data)
        self.url, self.cut, self.headers = url, cut, {"Content-Length": str(len(data))}

    def geturl(self):
        return self.url

    def read(self, n=-1):
        if self.cut is not None and self.tell() >= self.cut:
            raise ConnectionResetError("connection reset by VPN")
        return super().read(n)


def _fake_routes(monkeypatch, behaviour):
    """behaviour: route -> callable(url) returning a response or raising."""
    used = []

    class Opener:
        def __init__(self, route):
            self.route = route

        def open(self, req, timeout=None):
            used.append(self.route)
            return behaviour[self.route](req.full_url)
    monkeypatch.setattr(net, "build_opener", lambda route, check: Opener(route))
    monkeypatch.setattr(net, "system_proxies", lambda: {"https": "http://proxy:3128"})
    return used


def test_direct_route_is_tried_first_and_proxy_is_not_needed(monkeypatch):
    data = b"hello"
    used = _fake_routes(monkeypatch, {"direct": lambda u: _Resp(data, u),
                                      "proxy": lambda u: pytest.fail("proxy used")})
    assert net.fetch(URL, lambda r: r.read(), pose._check_url) == data
    assert used == ["direct"]


def test_falls_back_to_proxy_and_redownloads_whole_file(monkeypatch, tmp_path):
    """Direct connection dies in the middle (e.g. blocked without VPN/proxy):
    the file is downloaded again via the proxy, not glued from two halves."""
    data = _model_bytes()
    used = _fake_routes(monkeypatch, {"direct": lambda u: _Resp(data, u, cut=len(data) // 2),
                                      "proxy": lambda u: _Resp(data, u)})
    dest = tmp_path / "pose_landmarker_lite.task"
    seen = []
    pose._download(URL, dest, seen.append)
    assert used == ["direct", "proxy"]
    assert dest.read_bytes() == data
    assert not list(tmp_path.glob(".dl-*"))


def test_proxy_route_used_when_direct_is_blocked(monkeypatch):
    def refused(u):
        raise urllib.error.URLError(ConnectionRefusedError("blocked"))
    used = _fake_routes(monkeypatch, {"direct": refused, "proxy": lambda u: _Resp(b"ok", u)})
    assert net.fetch(URL, lambda r: r.read(), pose._check_url) == b"ok"
    assert used == ["direct", "proxy"]


def test_both_routes_fail_with_clear_message(monkeypatch):
    def down(u):
        raise socket.timeout("timed out")
    _fake_routes(monkeypatch, {"direct": down, "proxy": down})
    with pytest.raises(RuntimeError, match="storage.googleapis.com.*прокси или VPN"):
        net.fetch(URL, lambda r: r.read(), pose._check_url)


def test_http_errors_and_bad_files_are_not_retried(monkeypatch):
    def missing(u):
        raise urllib.error.HTTPError(u, 404, "Not Found", {}, None)
    used = _fake_routes(monkeypatch, {"direct": missing, "proxy": missing})
    with pytest.raises(urllib.error.HTTPError):
        net.fetch(URL, lambda r: r.read(), pose._check_url)
    assert used == ["direct"]
    for code in (403, 407, 429, 451, 502):     # blocked / proxy login / rate limit / server
        assert net.is_network_error(urllib.error.HTTPError(URL, code, "x", {}, None))
    assert not net.is_network_error(RuntimeError("model file is unexpectedly large"))


def test_redirects_and_final_url_are_checked_on_every_route(monkeypatch):
    _fake_routes(monkeypatch, {"direct": lambda u: _Resp(b"x", "https://evil.example/x"),
                               "proxy": lambda u: _Resp(b"x", u)})
    with pytest.raises(RuntimeError, match="неожиданный адрес"):
        net.fetch(URL, lambda r: r.read(), pose._check_url)
    with pytest.raises(RuntimeError):
        net.fetch("http://storage.googleapis.com/x", lambda r: r.read(), pose._check_url)


def test_proxy_route_only_when_a_proxy_exists(monkeypatch):
    monkeypatch.setattr(net, "system_proxies", lambda: {})
    assert net.routes_to_try() == ("direct",)
    monkeypatch.setattr(net, "system_proxies", lambda: {"https": "http://p:1"})
    assert net.routes_to_try() == ("direct", "proxy")


def test_direct_opener_ignores_proxy_settings(monkeypatch):
    import os
    for k in list(os.environ):
        if k.lower().endswith("_proxy"):
            monkeypatch.delenv(k)
    monkeypatch.setenv("https_proxy", "http://127.0.0.1:9")
    import urllib.request
    direct = net.build_opener("direct", pose._check_url)
    proxied = net.build_opener("proxy", pose._check_url)

    def proxies(op):
        h = [x for x in op.handlers if isinstance(x, urllib.request.ProxyHandler)]
        return h[0].proxies if h else {}
    assert proxies(direct) == {}
    assert proxies(proxied).get("https") == "http://127.0.0.1:9"


def test_osc_packets_stay_on_loopback():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    c = OscClient(rx.getsockname()[1])
    assert c._sock.getsockname()[0] == "127.0.0.1"
    c.send("/tracking/trackers/1/position", 0.0, 1.0, 0.0)
    _data, src = rx.recvfrom(1024)
    assert src[0] == "127.0.0.1"
    rx.close()
