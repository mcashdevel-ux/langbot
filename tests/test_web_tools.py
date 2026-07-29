"""Unit tests for web_tools.py — search/fetch wrappers.

Network is never touched: ``engines.search_engine`` and ``requests.get`` are
monkeypatched. The shared scratchpad is redirected to a temp directory by the
autouse fixture in ``tests/conftest.py``, and covered by ``test_scratch.py``.
"""

import json
import os

import components.scratch as scratch
import components.web_tools as web_tools


# ---------------------------------------------------------------------------
# search_web
# ---------------------------------------------------------------------------

class TestSearchWeb:
    def test_success_formats_results_and_saves_scratch(self, monkeypatch):
        results = [
            {"title": "First", "url": "http://a", "content": "alpha content"},
            {"title": "Second", "url": "http://b", "content": "beta content"},
        ]
        monkeypatch.setattr(web_tools, "search_engine", lambda *a, **k: results)
        out = web_tools.search_web("query", engine="duckduckgo")
        assert "First" in out and "Second" in out
        assert "scratch:search_" in out

    def test_caps_max_results(self, monkeypatch):
        captured = {}

        def fake_engine(engine, query, max_results):
            captured["max_results"] = max_results
            return []

        monkeypatch.setattr(web_tools, "search_engine", fake_engine)
        web_tools.search_web("q", max_results=999)
        assert captured["max_results"] == web_tools.SEARCH_MAX_RESULTS

    def test_no_results_message(self, monkeypatch):
        monkeypatch.setattr(web_tools, "search_engine", lambda *a, **k: [])
        out = web_tools.search_web("nothing", engine="ddg")
        assert "no results" in out

    def test_engine_error_is_caught(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("engine down")

        monkeypatch.setattr(web_tools, "search_engine", boom)
        out = web_tools.search_web("q", engine="brokenengine")
        assert "search error" in out
        assert "engine down" in out

    def test_missing_fields_handled(self, monkeypatch):
        monkeypatch.setattr(web_tools, "search_engine", lambda *a, **k: [{}])
        out = web_tools.search_web("q")
        assert "(no title)" in out

    def test_snippet_truncated(self, monkeypatch):
        long_content = "z" * 500
        monkeypatch.setattr(
            web_tools, "search_engine",
            lambda *a, **k: [{"title": "T", "url": "u", "content": long_content}],
        )
        out = web_tools.search_web("q")
        # scratch file has the full payload, inline snippet is capped.
        sid = out.split("scratch:")[1].split(")")[0]
        saved = json.loads(open(os.path.join(scratch.SCRATCH_DIR, f"{sid}.txt")).read())
        assert saved[0]["content"] == long_content


# ---------------------------------------------------------------------------
# fetch_url
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, text="", status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class TestFetchUrl:
    def test_adds_scheme(self, monkeypatch):
        seen = {}

        def fake_get(url, **kwargs):
            seen["url"] = url
            return _FakeResp("page text")

        monkeypatch.setattr(web_tools.requests, "get", fake_get)
        web_tools.fetch_url("example.com")
        assert seen["url"] == "https://r.jina.ai/https://example.com"

    def test_preserves_existing_scheme(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            web_tools.requests, "get",
            lambda url, **k: seen.update(url=url) or _FakeResp("t"),
        )
        web_tools.fetch_url("http://example.com")
        assert "http://example.com" in seen["url"]

    def test_success_short_content(self, monkeypatch):
        monkeypatch.setattr(web_tools.requests, "get", lambda url, **k: _FakeResp("short page"))
        out = web_tools.fetch_url("http://x")
        assert "short page" in out
        assert "truncated" not in out

    def test_fetch_save_cap_still_applies(self, monkeypatch):
        # The on-disk cap for fetched pages is unchanged by the move to
        # scratch.py's larger default.
        big = "q" * (web_tools.FETCH_SAVE_CHARS + 1000)
        monkeypatch.setattr(web_tools.requests, "get", lambda url, **k: _FakeResp(big))
        out = web_tools.fetch_url("http://x")
        sid = out.split("scratch:")[1].split(" ")[0]
        path = os.path.join(scratch.SCRATCH_DIR, f"{sid}.txt")
        assert os.path.getsize(path) == web_tools.FETCH_SAVE_CHARS

    def test_long_content_truncated_and_saved(self, monkeypatch):
        big = "q" * (web_tools.FETCH_INLINE_CHARS + 500)
        monkeypatch.setattr(web_tools.requests, "get", lambda url, **k: _FakeResp(big))
        out = web_tools.fetch_url("http://x")
        assert "truncated" in out
        assert "scratch:fetch_" in out

    def test_empty_content(self, monkeypatch):
        monkeypatch.setattr(web_tools.requests, "get", lambda url, **k: _FakeResp("   "))
        out = web_tools.fetch_url("http://x")
        assert "empty content" in out

    def test_network_error(self, monkeypatch):
        def boom(url, **k):
            raise ConnectionError("no route")

        monkeypatch.setattr(web_tools.requests, "get", boom)
        monkeypatch.setattr(web_tools.time, "sleep", lambda *a: None)
        out = web_tools.fetch_url("http://x")
        assert "fetch error" in out

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def flaky_get(url, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResp("rate limited", status_code=429)
            return _FakeResp("recovered page")

        monkeypatch.setattr(web_tools.requests, "get", flaky_get)
        monkeypatch.setattr(web_tools.time, "sleep", lambda *a: None)
        out = web_tools.fetch_url("http://x")
        assert "recovered page" in out
        assert calls["n"] == 2
