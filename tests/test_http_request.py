"""Tests for tools/plugins/http_request.py — SSRF protection (T1).

Covers:
  - Private IPv4 ranges (127.x, 10.x, 172.16.x, 192.168.x, 169.254.x)
  - IPv6 loopback / link-local
  - Blocked redirect targets
  - Opt-out via config flag
  - Normal public requests pass through
  - DNS failure treated as blocked
"""

import importlib
import socket
import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so the module loads without langchain / langbot infra
# ---------------------------------------------------------------------------

def _install_stubs():
    """Insert minimal fakes for langchain_core, components.config, etc."""
    # langchain_core.tools
    lc = types.ModuleType("langchain_core")
    lct = types.ModuleType("langchain_core.tools")
    lct.tool = lambda f: f          # identity decorator
    sys.modules.setdefault("langchain_core", lc)
    sys.modules.setdefault("langchain_core.tools", lct)

    # components.config
    comp = types.ModuleType("components")
    comp_cfg = types.ModuleType("components.config")

    class _Cfg:
        def get(self, key, default=None):
            return default

    comp_cfg.config = _Cfg()
    sys.modules.setdefault("components", comp)
    sys.modules.setdefault("components.config", comp_cfg)

    # components.scratch
    comp_scratch = types.ModuleType("components.scratch")
    comp_scratch.save_to_scratch = lambda text, prefix="": "stub-id"
    sys.modules.setdefault("components.scratch", comp_scratch)


_install_stubs()

# Now import the module under test (force reload so stubs take effect)
if "tools.plugins.http_request" in sys.modules:
    del sys.modules["tools.plugins.http_request"]

# Add the langbot root to the path so the relative import works
import os, pathlib
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import tools.plugins.http_request as hr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_getaddrinfo(ip: str):
    """Return a mock getaddrinfo that resolves any host to *ip*."""
    def _fake(host, port, *a, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0))]
    return _fake


def _make_getaddrinfo_v6(ip: str):
    def _fake(host, port, *a, **kw):
        return [(socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))]
    return _fake


# ---------------------------------------------------------------------------
# _is_private_ip
# ---------------------------------------------------------------------------

class TestIsPrivateIp:
    def test_loopback(self):
        assert hr._is_private_ip("127.0.0.1")

    def test_loopback_range(self):
        assert hr._is_private_ip("127.99.99.99")

    def test_rfc1918_10(self):
        assert hr._is_private_ip("10.0.0.1")

    def test_rfc1918_172(self):
        assert hr._is_private_ip("172.16.0.1")
        assert hr._is_private_ip("172.31.255.255")

    def test_rfc1918_192_168(self):
        assert hr._is_private_ip("192.168.1.1")

    def test_link_local(self):
        assert hr._is_private_ip("169.254.169.254")  # AWS metadata

    def test_ipv6_loopback(self):
        assert hr._is_private_ip("::1")

    def test_ipv6_link_local(self):
        assert hr._is_private_ip("fe80::1")

    def test_ipv6_unique_local(self):
        assert hr._is_private_ip("fc00::1")

    def test_public_ipv4(self):
        assert not hr._is_private_ip("8.8.8.8")
        assert not hr._is_private_ip("1.1.1.1")

    def test_public_ipv6(self):
        assert not hr._is_private_ip("2606:4700:4700::1111")  # Cloudflare

    def test_invalid_string(self):
        assert not hr._is_private_ip("not-an-ip")


# ---------------------------------------------------------------------------
# _check_host
# ---------------------------------------------------------------------------

class TestCheckHost:
    def test_private_ip_blocked(self):
        with patch("socket.getaddrinfo", _make_getaddrinfo("192.168.1.1")):
            blocked, reason = hr._check_host("internal.example.com")
        assert blocked
        assert "192.168.1.1" in reason

    def test_aws_metadata_blocked(self):
        with patch("socket.getaddrinfo", _make_getaddrinfo("169.254.169.254")):
            blocked, reason = hr._check_host("metadata.internal")
        assert blocked
        assert "169.254.169.254" in reason

    def test_public_ip_allowed(self):
        with patch("socket.getaddrinfo", _make_getaddrinfo("93.184.216.34")):
            blocked, reason = hr._check_host("example.com")
        assert not blocked

    def test_dns_failure_blocks(self):
        def _fail(*a, **kw):
            raise socket.gaierror("Name or service not known")
        with patch("socket.getaddrinfo", _fail):
            blocked, reason = hr._check_host("nonexistent.invalid")
        assert blocked
        assert "DNS" in reason

    def test_opt_out_skips_check(self):
        original = hr.BLOCK_PRIVATE_RANGES
        try:
            hr.BLOCK_PRIVATE_RANGES = False
            with patch("socket.getaddrinfo", _make_getaddrinfo("127.0.0.1")):
                blocked, reason = hr._check_host("localhost")
            assert not blocked
        finally:
            hr.BLOCK_PRIVATE_RANGES = original


# ---------------------------------------------------------------------------
# http_request tool — integration-level (httpx mocked)
# ---------------------------------------------------------------------------

def _mock_response(status=200, body="OK", headers=None, is_redirect=False, location=None):
    """Build a minimal httpx.Response-like mock."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    resp.reason_phrase = "OK" if status == 200 else "Found"
    resp.text = body
    resp.headers = httpx.Headers(headers or {})
    if location:
        resp.headers = httpx.Headers({"location": location, **(headers or {})})
    resp.is_redirect = is_redirect
    resp.url = httpx.URL("https://example.com/")
    return resp


class TestHttpRequestSSRF:
    """Tests exercising the full http_request() tool function."""

    def _call(self, url, method="GET"):
        return hr.http_request(method=method, url=url)

    def test_private_target_blocked_before_request(self):
        """Tool must return an error without opening any socket."""
        with patch("socket.getaddrinfo", _make_getaddrinfo("127.0.0.1")):
            with patch("httpx.Client") as mock_client:
                result = self._call("http://localhost/secret")
        assert "error" in result.lower() or "Blocked" in result
        mock_client.assert_not_called()  # no HTTP connection was attempted

    def test_metadata_endpoint_blocked(self):
        with patch("socket.getaddrinfo", _make_getaddrinfo("169.254.169.254")):
            with patch("httpx.Client"):
                result = self._call("http://169.254.169.254/latest/meta-data/")
        assert "error" in result.lower() or "Blocked" in result

    def test_public_target_passes(self):
        """Public targets must not be blocked and should proceed to httpx."""
        mock_resp = _mock_response(body="Hello from the internet")
        with patch("socket.getaddrinfo", _make_getaddrinfo("93.184.216.34")):
            with patch("httpx.Client") as MockClient:
                instance = MockClient.return_value.__enter__.return_value
                instance.request.return_value = mock_resp
                result = self._call("https://example.com/")
        assert "Hello from the internet" in result

    def test_redirect_to_private_blocked(self):
        """A redirect whose Location header resolves to a private IP must be blocked."""
        # The initial request goes to a public host, which redirects to a
        # private one.  The SSRF hook must intercept before following.
        redirect_resp = _mock_response(
            status=301, is_redirect=True,
            location="http://internal.corp/secret",
        )
        # Simulate: public host resolves fine, but 'internal.corp' → private
        def _selective_getaddrinfo(host, *a, **kw):
            if "internal" in host:
                return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("10.0.0.5", 0))]
            return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0))]

        with patch("socket.getaddrinfo", _selective_getaddrinfo):
            # Manually call the redirect hook to verify it raises
            redirect_resp.url = httpx.URL("https://public.example.com/")
            with pytest.raises(httpx.InvalidURL, match="SSRF redirect blocked"):
                hr._ssrf_redirect_hook(redirect_resp)

    def test_opt_out_allows_localhost(self):
        """When BLOCK_PRIVATE_RANGES=False, private IPs should pass through."""
        original = hr.BLOCK_PRIVATE_RANGES
        try:
            hr.BLOCK_PRIVATE_RANGES = False
            mock_resp = _mock_response(body="local response")
            with patch("socket.getaddrinfo", _make_getaddrinfo("127.0.0.1")):
                with patch("httpx.Client") as MockClient:
                    instance = MockClient.return_value.__enter__.return_value
                    instance.request.return_value = mock_resp
                    result = self._call("http://localhost:8080/")
            assert "local response" in result
        finally:
            hr.BLOCK_PRIVATE_RANGES = original

    def test_unsupported_method_rejected(self):
        result = hr.http_request(method="TRACE", url="https://example.com/")
        assert "unsupported method" in result.lower()

    def test_timeout_handled(self):
        with patch("socket.getaddrinfo", _make_getaddrinfo("93.184.216.34")):
            with patch("httpx.Client") as MockClient:
                instance = MockClient.return_value.__enter__.return_value
                instance.request.side_effect = httpx.TimeoutException("timed out")
                result = self._call("https://example.com/slow")
        assert "timed out" in result.lower()
