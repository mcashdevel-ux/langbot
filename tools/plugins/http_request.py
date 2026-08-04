"""Direct HTTP client — call REST APIs without going through Jina Reader.

Unlike ``fetch_url`` (which fetches human-readable page content via Jina),
this tool makes raw HTTP requests and returns status codes, headers, and body
previews.  Results over 2000 chars are saved to the scratchpad.

SSRF protection
---------------
By default (``web.http_request_block_private_ranges: true``) every request —
and every redirect hop — is checked against a block list of private, loopback,
and link-local ranges before a connection is opened.  Set the config key to
``false`` only if you intentionally need to reach ``localhost`` services
(e.g. your own dev server), consistent with this repo's config-first pattern.

IPv4-mapped / IPv4-compatible normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``_is_private_ip()`` unwraps IPv6 addresses that embed an IPv4 address before
running the block-list check:

* **IPv4-mapped** (``::ffff:a.b.c.d``) — the common form produced by dual-stack
  sockets; ``ipaddress.IPv6Address.ipv4_mapped`` extracts the embedded address.
* **6to4** (``2002:xx:xx::/48``) — ``ipaddress.IPv6Address.sixtofour``.
* **IPv4-compatible** (``::a.b.c.d``, deprecated RFC 4291 §2.5.5.1) — detected
  by checking that the high 96 bits are zero and extracting the low 32 bits.

Without this normalization ``::ffff:127.0.0.1`` would bypass the
``127.0.0.0/8`` entry because Python's ``ipaddress`` does **not** set
``.is_loopback`` for IPv4-mapped addresses and none of the pure-IPv6 block
entries match it.

Known gap — DNS-rebinding TOCTOU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``_check_host()`` resolves the target hostname once (via ``socket.getaddrinfo``)
to decide pass/block; ``httpx`` then performs its **own** independent DNS
resolution when it opens the actual connection.  An attacker who controls DNS
with a short TTL could return a public IP for the pre-flight check and a
private IP moments later for the real connection.  Closing this gap requires
pinning the checked IP for the socket connect (e.g. a custom
``httpx.AsyncHTTPTransport`` that resolves once and passes a literal address to
the OS) — a meaningfully larger transport-layer change tracked as a separate
follow-up, not part of this patch.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

import httpx
from langchain_core.tools import tool

from components.config import config
from components.scratch import save_to_scratch

logger = logging.getLogger(__name__)

_SAFE_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

INLINE_CHARS = 2000
TIMEOUT = 25

# ---------------------------------------------------------------------------
# SSRF block-list
# ---------------------------------------------------------------------------
BLOCK_PRIVATE_RANGES: bool = config.get("web.http_request_block_private_ranges", True)

_BLOCKED_NETWORKS = [
    # IPv4
    ipaddress.ip_network("127.0.0.0/8"),       # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),      # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),     # RFC 1918 private
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / AWS metadata
    ipaddress.ip_network("0.0.0.0/8"),          # "This" network
    ipaddress.ip_network("100.64.0.0/10"),      # CGNAT shared address space
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol Assignments
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1 (documentation)
    ipaddress.ip_network("198.18.0.0/15"),      # Benchmarking
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2 (documentation)
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3 (documentation)
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved (future use)
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("fc00::/7"),           # Unique-local
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("::/128"),             # Unspecified
]


def _is_private_ip(ip_str: str) -> bool:
    """Return True if *ip_str* falls in any private / reserved / loopback range.

    IPv6 addresses that embed an IPv4 address are unwrapped to their IPv4
    equivalent before the check so that addresses like ``::ffff:127.0.0.1``
    are correctly caught by the ``127.0.0.0/8`` block-list entry:

    * IPv4-mapped  ``::ffff:a.b.c.d``  → ``a.b.c.d``  (via ``.ipv4_mapped``)
    * 6to4         ``2002:xx:xx::``    → embedded v4  (via ``.sixtofour``)
    * IPv4-compat  ``::a.b.c.d``       → ``a.b.c.d``  (high-96-bits-zero test)
    """
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    # Normalise IPv6 forms that embed an IPv4 address so the IPv4 block-list
    # entries fire correctly for them.
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            # ::ffff:a.b.c.d  — the common dual-stack / socket form
            addr = addr.ipv4_mapped
        elif addr.sixtofour is not None:
            # 2002:xx:xx::/16 — 6to4 tunnel addresses
            addr = addr.sixtofour
        elif int(addr) >> 32 == 0 and int(addr) != 0:
            # ::a.b.c.d  — deprecated IPv4-compatible form (RFC 4291 §2.5.5.1)
            addr = ipaddress.IPv4Address(int(addr) & 0xFFFF_FFFF)
    return any(addr in net for net in _BLOCKED_NETWORKS)


def _check_host(host: str) -> tuple:
    """Resolve *host* via DNS and check every returned address.

    Returns ``(is_blocked: bool, reason: str)``.  A DNS failure is treated as
    blocked — the tool cannot verify safety so it fails closed.
    """
    if not BLOCK_PRIVATE_RANGES:
        return False, ""
    try:
        results = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        return True, f"DNS resolution failed for '{host}': {e}"

    for _family, _type, _proto, _canonname, sockaddr in results:
        ip = sockaddr[0]
        if _is_private_ip(ip):
            return True, (
                f"Blocked: '{host}' resolves to private/reserved address {ip} "
                f"(SSRF protection). Set web.http_request_block_private_ranges=false "
                f"in config to allow local targets."
            )
    return False, ""


def _ssrf_redirect_hook(response: httpx.Response) -> None:
    """httpx event hook — called for every response including redirect hops.

    Raises ``httpx.InvalidURL`` before httpx follows a redirect into a
    private range, so the blocking happens *before* the connection is opened.
    """
    if not BLOCK_PRIVATE_RANGES:
        return
    if not response.is_redirect:
        return
    location = response.headers.get("location", "")
    if not location:
        return
    try:
        parsed = urlparse(location)
        if not parsed.scheme:
            # Relative redirect — use base host
            base = urlparse(str(response.url))
            host = base.hostname or ""
        else:
            host = parsed.hostname or ""
        if host:
            blocked, reason = _check_host(host)
            if blocked:
                raise httpx.InvalidURL(f"SSRF redirect blocked: {reason}")
    except httpx.InvalidURL:
        raise
    except Exception as e:
        raise httpx.InvalidURL(f"SSRF redirect check error: {e}")


@tool
def http_request(
    method: str,
    url: str,
    headers: str = "",
    body: str = "",
) -> str:
    """Make a direct HTTP request and return the response.

    Use this to call REST APIs, webhooks, or check endpoint health — anything
    that expects a raw HTTP response rather than human-readable page content.

    Requests to private/loopback/link-local addresses (including cloud metadata
    endpoints such as 169.254.169.254) are blocked by default.  Set
    ``web.http_request_block_private_ranges: false`` in the config to allow
    local targets.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS).
        url: Full URL including the scheme (https://...).
        headers: Optional JSON object string for request headers,
                 e.g. ``'{"Authorization": "Bearer ...", "Accept": "application/json"}'``.
        body: Optional request body string (for POST, PUT, PATCH).

    Returns:
        Status line, response headers, and body preview.  Full body is saved to
        the scratchpad when it exceeds {INLINE_CHARS} characters.
    """
    method = (method or "GET").strip().upper()
    if method not in _SAFE_METHODS:
        return (
            f"http_request error: unsupported method '{method}'. "
            f"Allowed: {', '.join(sorted(_SAFE_METHODS))}"
        )

    if not url or not url.startswith(("http://", "https://")):
        url = "https://" + (url or "").lstrip("/")
    if not url.startswith(("http://", "https://")):
        return f"http_request error: invalid URL '{url}'"

    # --- SSRF pre-flight: check the initial target before opening any socket ---
    if BLOCK_PRIVATE_RANGES:
        parsed_url = urlparse(url)
        host = parsed_url.hostname or ""
        if host:
            blocked, reason = _check_host(host)
            if blocked:
                logger.warning("http_request SSRF block: %s", reason)
                return f"http_request error: {reason}"

    # Parse headers
    req_headers = {}
    if headers and headers.strip():
        try:
            import json
            req_headers = json.loads(headers)
            if not isinstance(req_headers, dict):
                return "http_request error: headers must be a JSON object (dict)"
        except json.JSONDecodeError as e:
            return f"http_request error: invalid headers JSON: {e}"

    try:
        with httpx.Client(
            timeout=TIMEOUT,
            follow_redirects=True,
            event_hooks={"response": [_ssrf_redirect_hook]},
        ) as client:
            resp = client.request(method=method, url=url, headers=req_headers, content=body or None)
    except httpx.InvalidURL as e:
        logger.warning("http_request SSRF redirect block: %s", e)
        return f"http_request error: {e}"
    except httpx.TimeoutException:
        return f"http_request error: request to {url} timed out after {TIMEOUT}s"
    except Exception as e:
        return f"http_request error: {type(e).__name__}: {e}"

    # Build response
    resp_headers = "\n".join(f"  {k}: {v}" for k, v in resp.headers.items())
    body_text = resp.text or "(empty body)"
    status_line = f"{resp.status_code} {resp.reason_phrase or ''}"

    if len(body_text) > INLINE_CHARS:
        sid = save_to_scratch(body_text, prefix="http")
        body_preview = body_text[:INLINE_CHARS]
        body_display = (
            f"{body_preview}\n\n... [{len(body_text):,} total chars] "
            f"(full body at scratch:{sid})"
        )
    else:
        body_display = body_text

    return (
        f"{method} {url}\n"
        f"Status: {status_line}\n"
        f"Headers:\n{resp_headers}\n\n"
        f"Body:\n{body_display}"
    )


# Plugin contract — discovered by tools.plugins.discover_plugins()
TOOLS = [http_request]

DESCRIPTIONS = {
    "http_request": (
        "Make a direct HTTP request to a REST API, webhook, or endpoint "
        "and return the status code and response body."
    ),
}

TRIGGERS = {
    "http_request": (
        r"\b(api|http|https|REST|endpoint|webhook|curl|POST|GET request|"
        r"call .*(api|service|endpoint)|check .*(endpoint|health|status code))\b"
    ),
}

# Tool names from this plugin that should always be bound (added to tools.core).
CORE = []
