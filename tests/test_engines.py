"""Unit tests for engines.py — the parts that don't require a live SearXNG
install: result extraction, the httpx/requests response wrapper, and the
``search_engine`` orchestration (with the engine + HTTP layers mocked).
"""

import sys
import types

import httpx
import pytest

import engines


# ---------------------------------------------------------------------------
# _extract_results
# ---------------------------------------------------------------------------

class TestExtractResults:
    def test_extracts_plain_dicts(self):
        out = engines._extract_results([
            {"title": "T", "url": "http://u", "content": "body", "engine": "e"},
        ])
        assert out == [{
            "title": "T", "url": "http://u", "content": "body", "engine": "e",
            "thumbnail": "", "img_src": "", "publishedDate": "",
            "template": "", "category": "",
        }]

    def test_infobox_fallback_fields(self):
        out = engines._extract_results([
            {"infobox": "Python", "id": "http://py", "extract": "a language"},
        ])
        assert out[0]["title"] == "Python"
        assert out[0]["url"] == "http://py"
        assert out[0]["content"] == "a language"

    def test_filters_fully_empty_results(self):
        out = engines._extract_results([{"engine": "e"}])  # no title/url/content
        assert out == []

    def test_struct_like_attribute_access(self):
        r = types.SimpleNamespace(
            title="S", url="http://s", content="c", engine="eng",
            thumbnail="", img_src="", publishedDate=None,
            template="", category="",
        )
        out = engines._extract_results([r])
        assert out[0]["title"] == "S"
        assert out[0]["publishedDate"] == ""  # None coerced to ""

    def test_published_date_coerced_to_str(self):
        out = engines._extract_results([
            {"title": "T", "url": "u", "publishedDate": 12345},
        ])
        assert out[0]["publishedDate"] == "12345"


# ---------------------------------------------------------------------------
# _HttpxResponseWrapper
# ---------------------------------------------------------------------------

class _FakeRequestsResp:
    def __init__(self):
        self.text = "body text"
        self.status_code = 200
        self.ok = True
        self.content = b"body text"
        self.headers = {"Content-Type": "text/html"}
        self.url = "http://example.com/path"
        self.encoding = "utf-8"
        self._raised = False

    def raise_for_status(self):
        self._raised = True

    def json(self, **kwargs):
        return {"ok": True}


class TestHttpxResponseWrapper:
    def test_copies_attributes(self):
        resp = _FakeRequestsResp()
        w = engines._HttpxResponseWrapper(resp, {"query": "x"})
        assert w.text == "body text"
        assert w.status_code == 200
        assert w.ok is True
        assert w.content == b"body text"
        assert w.search_params == {"query": "x"}

    def test_url_is_httpx_url_with_host(self):
        w = engines._HttpxResponseWrapper(_FakeRequestsResp(), {})
        assert isinstance(w.url, httpx.URL)
        assert w.url.host == "example.com"

    def test_raise_for_status_delegates(self):
        resp = _FakeRequestsResp()
        engines._HttpxResponseWrapper(resp, {}).raise_for_status()
        assert resp._raised is True

    def test_json_delegates(self):
        w = engines._HttpxResponseWrapper(_FakeRequestsResp(), {})
        assert w.json() == {"ok": True}


# ---------------------------------------------------------------------------
# search_engine orchestration (engine + HTTP layers mocked)
# ---------------------------------------------------------------------------

class _FakeEngine:
    def __init__(self, response_results, set_url=True):
        self._results = response_results
        self._set_url = set_url

    def request(self, query, params):
        if self._set_url:
            params["url"] = "http://engine/search?q=" + query

    def response(self, resp):
        return self._results


class TestSearchEngine:
    @pytest.fixture(autouse=True)
    def stub_build_params(self, monkeypatch):
        # _build_params imports searx.utils; stub it out so search_engine's
        # orchestration can be tested without a SearXNG install.
        monkeypatch.setattr(engines, "_build_params", lambda **kwargs: {"method": "GET"})

    def test_happy_path(self, monkeypatch):
        engine = _FakeEngine([{"title": "R", "url": "http://r", "content": "c"}])
        monkeypatch.setattr(engines, "_load_engine", lambda name: engine)
        monkeypatch.setattr(engines, "_make_http_request", lambda params: _FakeRequestsResp())

        out = engines.search_engine("fake", "query", max_results=5)
        assert out and out[0]["title"] == "R"

    def test_engine_declines_returns_empty(self, monkeypatch):
        engine = _FakeEngine([], set_url=False)  # never sets params["url"]
        monkeypatch.setattr(engines, "_load_engine", lambda name: engine)
        monkeypatch.setattr(engines, "_make_http_request", lambda params: _FakeRequestsResp())

        assert engines.search_engine("fake", "query") == []

    def test_http_failure_raises_runtime_error(self, monkeypatch):
        engine = _FakeEngine([])
        monkeypatch.setattr(engines, "_load_engine", lambda name: engine)

        def boom(params):
            raise ConnectionError("down")

        monkeypatch.setattr(engines, "_make_http_request", boom)
        with pytest.raises(RuntimeError, match="HTTP request failed"):
            engines.search_engine("fake", "query")

    def test_response_parse_error_is_swallowed(self, monkeypatch):
        class BadEngine(_FakeEngine):
            def response(self, resp):
                raise ValueError("bad parse")

        monkeypatch.setattr(engines, "_load_engine", lambda name: BadEngine([]))
        monkeypatch.setattr(engines, "_make_http_request", lambda params: _FakeRequestsResp())
        assert engines.search_engine("fake", "query") == []

    def test_max_results_limit_applied(self, monkeypatch):
        many = [{"title": f"T{i}", "url": f"http://{i}"} for i in range(10)]
        monkeypatch.setattr(engines, "_load_engine", lambda name: _FakeEngine(many))
        monkeypatch.setattr(engines, "_make_http_request", lambda params: _FakeRequestsResp())
        out = engines.search_engine("fake", "query", max_results=3)
        assert len(out) == 3


# ---------------------------------------------------------------------------
# _make_http_request
# ---------------------------------------------------------------------------

class TestMakeHttpRequest:
    def test_requires_url(self):
        with pytest.raises(ValueError, match="No URL"):
            engines._make_http_request({"url": ""})

    def test_get_request(self, monkeypatch):
        calls = {}

        class FakeSession:
            def get(self, url, **kwargs):
                calls["method"] = "GET"
                calls["url"] = url
                return "resp"

            def post(self, url, **kwargs):
                calls["method"] = "POST"
                return "resp"

            def close(self):
                calls["closed"] = True

        monkeypatch.setattr(engines.requests, "Session", FakeSession)
        engines._make_http_request({"url": "http://x", "method": "GET"})
        assert calls["method"] == "GET"
        assert calls["closed"] is True

    def test_post_request_with_data(self, monkeypatch):
        calls = {}

        class FakeSession:
            def get(self, url, **kwargs):
                calls["method"] = "GET"
                return "resp"

            def post(self, url, **kwargs):
                calls["method"] = "POST"
                calls["data"] = kwargs.get("data")
                return "resp"

            def close(self):
                pass

        monkeypatch.setattr(engines.requests, "Session", FakeSession)
        engines._make_http_request({"url": "http://x", "method": "POST", "data": {"k": "v"}})
        assert calls["method"] == "POST"
        assert calls["data"] == {"k": "v"}


# ---------------------------------------------------------------------------
# Engine loader / listing (with a stubbed ``searx`` package)
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_searx(monkeypatch):
    """Install a minimal fake ``searx`` + ``searx.engines`` into sys.modules
    and mark the bootstrap as already done."""
    loaded = {}

    engines_cfg = [
        {"name": "arxiv", "engine": "arxiv", "shortcut": "arx",
         "categories": ["science"]},
        {"name": "google", "engine": "google", "shortcut": "go"},
        {"name": "hidden", "engine": "x", "disabled": True},
    ]

    searx_mod = types.ModuleType("searx")
    searx_mod.settings = {"engines": engines_cfg}

    searx_engines = types.ModuleType("searx.engines")

    def load_engines(cfg):
        for c in cfg:
            loaded[c["name"]] = types.SimpleNamespace(name=c["name"])

    searx_engines.load_engines = load_engines
    searx_engines.engines = loaded
    searx_mod.engines = searx_engines

    monkeypatch.setitem(sys.modules, "searx", searx_mod)
    monkeypatch.setitem(sys.modules, "searx.engines", searx_engines)
    monkeypatch.setattr(engines, "_SEARX_INITIALIZED", True)
    monkeypatch.setattr(engines, "_LOADED_ENGINES", {})
    return searx_mod


class TestEngineLoader:
    def test_load_engine_success(self, fake_searx):
        eng = engines._load_engine("arxiv")
        assert eng.name == "arxiv"

    def test_load_engine_caches(self, fake_searx):
        first = engines._load_engine("arxiv")
        assert engines._load_engine("arxiv") is first

    def test_load_engine_unknown_raises(self, fake_searx):
        with pytest.raises(ValueError, match="not found"):
            engines._load_engine("does-not-exist")

    def test_get_engine_names_excludes_disabled(self, fake_searx):
        names = engines._get_engine_names()
        assert "arxiv" in names and "google" in names
        assert "hidden" not in names

    def test_list_engines_sorted_and_filtered(self, fake_searx):
        listed = engines.list_engines()
        names = [e["name"] for e in listed]
        assert names == ["arxiv", "google"]  # sorted, 'hidden' filtered
        assert listed[0]["shortcut"] == "arx"
