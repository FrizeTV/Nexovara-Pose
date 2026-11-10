"""Downloads that work with and without a proxy / VPN.

Nexovara Pose goes online only to download files (neural-network models from Google
and — only if the RTMPose add-on is switched on — from OpenMMLab, nothing
else), always over HTTPS with certificate verification and an allowlist of
hosts. A system proxy that is broken, slow or blocks those
hosts used to make the download fail, so every download now tries:

  1. a DIRECT connection, bypassing the system proxy (Windows proxy settings,
     HTTP(S)_PROXY variables);
  2. the system proxy, if the direct connection fails (networks that only
     work through a proxy).

Certificate checks and the host allowlist stay the same on both routes, so a
proxy can't substitute files (models are additionally checked by SHA-256).

Tracking data (OSC, driver) never uses this module: it goes over UDP to
127.0.0.1, which no proxy or VPN touches — see osc.py.

Only the Python standard library is used: tools/setup.py imports this before
any packages are installed.
"""
from __future__ import annotations

import http.client
import logging
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, Iterable, Optional, Tuple

log = logging.getLogger(__name__)

ROUTES = ("direct", "proxy")
ROUTE_NAMES = {"direct": "напрямую (в обход прокси)", "proxy": "через системный прокси"}


class _CheckedRedirect(urllib.request.HTTPRedirectHandler):
    """Every redirect target must pass the same allowlist check."""

    def __init__(self, check_url: Callable[[str], None]):
        super().__init__()
        self._check = check_url

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._check(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def system_proxies() -> Dict[str, str]:
    """Proxies from the environment and (on Windows) the system settings."""
    try:
        return {k: v for k, v in urllib.request.getproxies().items()
                if k in ("http", "https", "all")}
    except Exception:
        return {}


def build_opener(route: str, check_url: Callable[[str], None]) -> urllib.request.OpenerDirector:
    ctx = ssl.create_default_context()          # certificate verification ON
    proxies = {} if route == "direct" else system_proxies()
    return urllib.request.build_opener(urllib.request.ProxyHandler(proxies),
                                       urllib.request.HTTPSHandler(context=ctx),
                                       _CheckedRedirect(check_url))


def routes_to_try() -> Tuple[str, ...]:
    """Direct first; the proxy route only if a proxy is actually configured."""
    return ROUTES if system_proxies() else ("direct",)


# connection-level problems -> try the next route. HTTP answers that depend on
# the route also do: 403/451 (blocked by a provider/firewall/filter), 407 (proxy
# wants a login), 429 (rate limit per IP), 5xx. 404 etc. are final.
ROUTE_DEPENDENT_HTTP = {403, 407, 429, 451}
NETWORK_ERRORS = (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                  ssl.SSLError, http.client.HTTPException)


def is_network_error(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return e.code in ROUTE_DEPENDENT_HTTP or e.code >= 500
    return isinstance(e, NETWORK_ERRORS)


def fetch(url: str, consume: Callable[[object], object], check_url: Callable[[str], None],
          headers: Optional[Dict[str, str]] = None, timeout: float = 30.0,
          routes: Optional[Iterable[str]] = None):
    """Open `url` and pass the response to `consume(response)`; on a network
    failure the WHOLE attempt (including reading the body) is repeated on the
    next route. Returns whatever `consume` returns."""
    check_url(url)
    last: Optional[BaseException] = None
    for route in tuple(routes or routes_to_try()):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with build_opener(route, check_url).open(req, timeout=timeout) as r:
                check_url(r.geturl())
                result = consume(r)
            if last is not None:
                log.info("download ok %s: %s", ROUTE_NAMES[route], url)
            return result
        except Exception as e:
            if not is_network_error(e):
                raise
            log.warning("download %s failed (%s): %s", ROUTE_NAMES[route], url, e)
            last = e
    hint = (" Проверьте интернет; если вы за прокси или VPN — включите его или добавьте "
            "программу в исключения (split tunneling).")
    raise RuntimeError(f"не удалось скачать {urllib.parse.urlparse(url).hostname}: {last}."
                       + hint) from last
