"""Direct HTTP client — call REST APIs without going through Jina Reader.

Unlike ``fetch_url`` (which fetches human-readable page content via Jina),
this tool makes raw HTTP requests and returns status codes, headers, and body
previews.  Results over 2000 chars are saved to the scratchpad.
"""

import logging
import socket
import ipaddress
import httpx
from langchain_core.tools import tool

from components.scratch import save_to_scratch
from components.config import config

logger = logging.getLogger(__name__)

_SAFE_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}

INLINE_CHARS = 2000
TIMEOUT = 25


def _is_private_ip(ip_str: str) -> bool:
    """Check if an IP string is loopback, private, link-local, or unspecified."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private or
            ip.is_loopback or
            ip.is_link_local or
            ip.is_unspecified
        )
    except ValueError:
        return False


def _is_host_private(host: str) -> bool:
    """Resolve a host name and check if any resolved IP is private/loopback/link-local."""
    if _is_private_ip(host):
        return True
    try:
        infos = socket.getaddrinfo(host, None)
        for family, _, _, _, sockaddr in infos:
            ip = sockaddr[0]
            if _is_private_ip(ip):
                return True
    except socket.gaierror:
        pass
    return False


def on_request(request):
    """Enforce SSRF block on any outgoing request, including redirects."""
    block_private = config.get("web.http_request_block_private_ranges", True)
    if block_private:
        host = request.url.host
        if _is_host_private(host):
            raise ValueError(f"Access to private/loopback/link-local address '{host}' is forbidden")


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
        return f"http_request error: unsupported method '{method}'. Allowed: {', '.join(sorted(_SAFE_METHODS))}"

    if not url or not url.startswith(("http://", "https://")):
        url = "https://" + (url or "").lstrip("/")
    if not url.startswith(("http://", "https://")):
        return f"http_request error: invalid URL '{url}'"

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
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True, event_hooks={"request": [on_request]}) as client:
            resp = client.request(method=method, url=url, headers=req_headers, content=body or None)
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
        body_display = f"{body_preview}\n\n... [{len(body_text):,} total chars] (full body at scratch:{sid})"
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
