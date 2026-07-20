"""SearXNG Engine Adapter — run SearXNG engines directly, no webapp needed.

Loads any SearXNG engine module and calls request()/response() directly,
using SAGE's HTTP stack. Supports all 227 engines from SearXNG.

Usage:
    from features.engines import search_engine
    
    # Simple interface
    results = search_engine("arxiv", "machine learning")
    for r in results:
        print(r["title"], r["url"])
    
    # With options
    results = search_engine("google", "latest news", pageno=2, lang="en-US")
"""

import sys
import os
import typing as t
import logging
from urllib.parse import urlencode

import httpx
import requests

# ---------------------------------------------------------------------------
# Bootstrap: initialize SearXNG settings + engine loader once
# ---------------------------------------------------------------------------

_SEARX_INITIALIZED = False
_ENGINE_CACHE: dict[str, t.Any] = {}
_LOADED_ENGINES: dict[str, t.Any] = {}

logger = logging.getLogger("sage.engines")


def _ensure_searx_initialized():
    global _SEARX_INITIALIZED
    if _SEARX_INITIALIZED:
        return
    
    # Check several known locations for SearXNG source
    _script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(_script_dir, "searxng-src"),
        os.path.expanduser("~/searxng-src"),
        "/usr/local/searxng/searxng-src",
    ]
    searx_src = None
    for p in candidate_paths:
        if os.path.isdir(os.path.join(p, "searx")):
            searx_src = p
            break
    
    if searx_src is None:
        searx_src = os.path.expanduser("~/searxng-src")
        import subprocess
        logger.info("SearXNG source not found — cloning to %s", searx_src)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/searxng/searxng", searx_src],
                check=True, capture_output=True, timeout=120
            )
            logger.info("SearXNG source cloned successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clone SearXNG source: {e}")
    
    if searx_src not in sys.path:
        sys.path.insert(0, searx_src)
    
    # Point to our settings
    settings_path = os.environ.get("SEARXNG_SETTINGS_PATH", "/etc/searxng/settings.yml")
    if not os.path.exists(settings_path):
        # Fallback: use the default settings from source
        settings_path = os.path.join(searx_src, "searx", "settings.yml")
    os.environ["SEARXNG_SETTINGS_PATH"] = settings_path
    
    import searx  # noqa: F401 — auto-calls init_settings()
    _SEARX_INITIALIZED = True


def _load_engine(name: str):
    """Load a single SearXNG engine by name, caching it."""
    if name in _LOADED_ENGINES:
        return _LOADED_ENGINES[name]
    
    _ensure_searx_initialized()
    
    import searx
    from searx.engines import load_engines, engines
    
    # Find this engine in settings
    engines_cfg = searx.settings.get("engines", [])
    cfg = [e for e in engines_cfg if e.get("name") == name]
    if not cfg:
        raise ValueError(f"Engine '{name}' not found in SearXNG settings ({len(engines_cfg)} engines available)")
    
    load_engines(cfg)
    if name not in engines:
        raise ValueError(f"Engine '{name}' failed to load")
    
    _LOADED_ENGINES[name] = engines[name]
    return engines[name]


def _get_engine_names() -> list[str]:
    """Return list of all available SearXNG engine names."""
    _ensure_searx_initialized()
    import searx
    engines_cfg = searx.settings.get("engines", [])
    return [e["name"] for e in engines_cfg if not e.get("disabled")]


# ---------------------------------------------------------------------------
# HTTP transport: convert between requests.Response and httpx.Response
# ---------------------------------------------------------------------------

class _HttpxResponseWrapper:
    """Wraps a requests.Response to behave like an httpx.Response for SearXNG engine response() functions.
    
    SearXNG engines expect an httpx.Response with: .text, .status_code, .ok, .content, .headers, .url, .search_params
    """
    
    def __init__(self, resp: requests.Response, params: dict):
        self._resp = resp
        self.text = resp.text
        self.status_code = resp.status_code
        self.ok = resp.ok
        self.content = resp.content
        self.headers = resp.headers
        # Some engines (e.g. google) inspect resp.url.host, which only exists
        # on a real httpx.URL — a plain str breaks them with
        # AttributeError: 'str' object has no attribute 'host'.
        self.url = httpx.URL(str(resp.url))
        self.search_params = params
        self.encoding = resp.encoding
    
    def raise_for_status(self):
        self._resp.raise_for_status()
    
    def json(self, **kwargs):
        return self._resp.json(**kwargs)


def _make_http_request(params: dict) -> requests.Response:
    """Execute the HTTP request built by an engine's request() function.
    
    Uses requests.Session (SAGE's stack) with timeout and proper handling.
    """
    method = params.get("method", "GET")
    url = params.get("url", "")
    headers = params.get("headers", {})
    cookies = params.get("cookies", {})
    data = params.get("data") or None  # POST form data
    json_data = params.get("json") or None
    content = params.get("content") or None
    
    if not url:
        raise ValueError("No URL in params (engine declined the query)")
    
    # Map SearXNG param names to requests param names
    req_kwargs = {
        "headers": headers,
        "cookies": cookies,
        "timeout": 30,
        "allow_redirects": params.get("allow_redirects", True),
    }
    
    if data:
        req_kwargs["data"] = data
    elif json_data:
        req_kwargs["json"] = json_data
    elif content:
        req_kwargs["data"] = content
    
    session = requests.Session()
    try:
        if method == "POST":
            resp = session.post(url, **req_kwargs)
        else:
            resp = session.get(url, **req_kwargs)
        return resp
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Param builder
# ---------------------------------------------------------------------------

def _build_params(
    query: str,
    engine: t.Any,
    category: str = "",
    pageno: int = 1,
    safesearch: int = 0,
    time_range: str | None = None,
    lang: str = "en",
    **extra,
) -> dict:
    """Build OnlineParams dict for an engine's request() function."""
    from searx.utils import gen_useragent
    
    return {
        "method": "GET",
        "headers": {"User-Agent": gen_useragent()},
        "data": {},
        "json": {},
        "content": b"",
        "url": "",
        "cookies": {},
        "allow_redirects": True,
        "max_redirects": 5,
        "soft_max_redirects": 3,
        "auth": None,
        "verify": None,
        "raise_for_httperror": False,
        "query": query,
        "category": category or (engine.categories[0] if engine.categories else "general"),
        "pageno": pageno,
        "safesearch": safesearch,
        "time_range": time_range,
        "engine_data": {},
        "searxng_locale": lang,
        "language": lang,
        **extra,
    }


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _extract_results(results_list) -> list[dict]:
    """Convert EngineResults (LegacyResult dicts, Result structs, infoboxes) to plain dicts."""
    extracted = []
    for r in results_list:
        if isinstance(r, dict):
            # Normalise: handle infobox dicts (from wikipedia etc) which use 'infobox'/'id' instead of 'title'/'url'
            title = r.get("title", "") or r.get("infobox", "") or ""
            url = r.get("url", "") or r.get("id", "") or ""
            content = r.get("content", "") or r.get("extract", "") or ""
            item = {
                "title": title,
                "url": url,
                "content": content,
                "engine": r.get("engine", ""),
                "thumbnail": r.get("thumbnail", ""),
                "img_src": r.get("img_src", ""),
                "publishedDate": str(r.get("publishedDate", "") or ""),
                "template": r.get("template", ""),
                "category": r.get("category", ""),
            }
        else:
            # Result (msgspec.Struct) — use attribute access
            item = {
                "title": getattr(r, "title", "") or getattr(r, "infobox", "") or "",
                "url": getattr(r, "url", "") or getattr(r, "id", "") or "",
                "content": getattr(r, "content", "") or getattr(r, "extract", "") or "",
                "engine": getattr(r, "engine", "") or "",
                "thumbnail": getattr(r, "thumbnail", "") or "",
                "img_src": getattr(r, "img_src", "") or "",
                "publishedDate": str(getattr(r, "publishedDate", "") or ""),
                "template": getattr(r, "template", "") or "",
                "category": getattr(r, "category", "") or "",
            }
        # Filter empty results
        if item["title"] or item["url"] or item["content"]:
            extracted.append(item)
    return extracted


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search_engine(
    engine_name: str,
    query: str,
    *,
    pageno: int = 1,
    safesearch: int = 0,
    time_range: str | None = None,
    lang: str = "en",
    max_results: int = 10,
) -> list[dict]:
    """Run a search through the specified SearXNG engine directly.
    
    Args:
        engine_name: Engine name (e.g. 'google', 'wikipedia', 'arxiv', 'github', 'duckduckgo')
        query: Search query string
        pageno: Page number (1-indexed)
        safesearch: 0=off, 1=moderate, 2=strict
        time_range: 'day', 'week', 'month', 'year', or None
        lang: Language/locale (e.g. 'en', 'en-US', 'de')
        max_results: Maximum results to return
    
    Returns:
        List of dicts with keys: title, url, content, engine, thumbnail, publishedDate
    
    Raises:
        ValueError: If engine not found
        RuntimeError: If search fails
    """
    engine = _load_engine(engine_name)
    
    params = _build_params(
        query=query,
        engine=engine,
        pageno=pageno,
        safesearch=safesearch,
        time_range=time_range,
        lang=lang,
    )
    
    # Step 1: Build request
    try:
        engine.request(query, params)
    except Exception as e:
        raise RuntimeError(f"Engine '{engine_name}' failed to build request: {e}") from e
    
    if not params.get("url"):
        # Engine declined the query (e.g. query too long for DDG)
        return []
    
    # Step 2: Make HTTP request
    try:
        resp = _make_http_request(params)
    except Exception as e:
        raise RuntimeError(f"HTTP request failed for engine '{engine_name}': {e}") from e
    
    # Step 3: Wrap response — need to cast to SXNG_Response for some engines
    wrapped = _HttpxResponseWrapper(resp, params)
    
    # Some engines check for SXNG_Response type specifically via raise_for_httperror
    # Let the engine's response() function handle parsing
    try:
        from searx.extended_types import SXNG_Response
        wrapped = t.cast(SXNG_Response, wrapped)
    except (ImportError, TypeError):
        pass
    
    # Step 4: Parse response
    try:
        results = engine.response(wrapped)
    except Exception as e:
        # Some engines (e.g. wikipedia) may return errors for certain queries.
        # We degrade to an empty result set rather than crashing the caller, but
        # log at warning level so a parsing failure isn't indistinguishable from
        # a genuinely empty search.
        logger.warning(
            "Engine '%s' response parsing failed (HTTP %s): %s",
            engine_name, getattr(resp, "status_code", "?"), e,
        )
        results = []
    
    # Step 5: Extract results
    extracted = _extract_results(results)
    
    return extracted[:max_results]


def list_engines() -> list[dict]:
    """List all available engines with their metadata."""
    _ensure_searx_initialized()
    import searx
    from searx.engines import load_engines, engines
    
    # Get from settings rather than loading all
    engine_list = searx.settings.get("engines", [])  # type: ignore[attr-defined]
    result = []
    for cfg in engine_list:
        if cfg.get("disabled") or cfg.get("inactive"):
            continue
        result.append({
            "name": cfg.get("name", ""),
            "engine_type": cfg.get("engine", ""),
            "shortcut": cfg.get("shortcut", ""),
            "categories": cfg.get("categories", ["general"]),
            "timeout": cfg.get("timeout", 3),
            "paging": cfg.get("paging", False),
            "time_range_support": cfg.get("time_range_support", False),
            "safesearch": cfg.get("safesearch", False),
            "about": cfg.get("about", {}),
        })
    return sorted(result, key=lambda x: x["name"])


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # List available
    all_engines = list_engines()
    print(f"Available engines: {len(all_engines)}")
    
    # Test a few
    for name in ["arxiv", "github", "wikipedia", "duckduckgo"]:
        try:
            results = search_engine(name, "python programming", max_results=3)
            print(f"\n{name}: {len(results)} results")
            for r in results:
                print(f"  - {r['title'][:60]}")
        except Exception as e:
            print(f"\n{name}: ERROR — {e}")
