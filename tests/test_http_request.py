import pytest
import httpx
from tools.plugins.http_request import http_request
from components.config import config


def test_http_request_allowed():
    # Make a public request to an allowed endpoint
    res = http_request.invoke({"method": "GET", "url": "https://httpbin.org/status/200"})
    assert "200" in res or "http_request error" not in res


def test_http_request_blocked_private_ip():
    # Attempting to query localhost
    res = http_request.invoke({"method": "GET", "url": "http://127.0.0.1:8080"})
    assert "forbidden" in res or "forbidden" in res.lower() or "blocked" in res.lower()


def test_http_request_blocked_private_hostname():
    # Attempting to query localhost by hostname
    res = http_request.invoke({"method": "GET", "url": "http://localhost:8080"})
    assert "forbidden" in res or "forbidden" in res.lower() or "blocked" in res.lower()


def test_http_request_blocked_redirect():
    # Attempting to trigger a redirect to 127.0.0.1
    res = http_request.invoke({
        "method": "GET",
        "url": "https://httpbin.org/redirect-to?url=http%3A%2F%2F127.0.0.1%3A8080"
    })
    assert "forbidden" in res or "forbidden" in res.lower() or "blocked" in res.lower()


def test_http_request_bypass_config(monkeypatch):
    # Overriding the block config to False should bypass block check
    original_get = config.get
    def mock_get(key, default, env=None):
        if key == "web.http_request_block_private_ranges":
            return False
        return original_get(key, default, env)
    
    monkeypatch.setattr(config, "get", mock_get)
    res = http_request.invoke({"method": "GET", "url": "http://127.0.0.1:9999"})
    # It should not raise "forbidden" error. Instead, it should let the connection fail normally (e.g., ConnectError).
    assert "forbidden" not in res
    assert "ConnectError" in res or "ConnectTimeout" in res or "refused" in res.lower() or "timed out" in res.lower()
