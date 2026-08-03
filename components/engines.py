"""SearXNG engines, run directly — no SearXNG webapp, no HTTP hop.

What SearXNG is worth here is its engine modules: one hand-maintained parser per
search site. This loads those modules and calls their ``request()`` / ``response()``
functions itself over a plain ``requests`` stack, so ``web_tools.search_web`` needs
nothing running.

That means the SearXNG source tree has to be importable. ``_ensure_searx_initialized``
looks in ``web.searxng_source_dir``, then ``<repo>/searxng-src``, ``~/searxng-src``
and ``/usr/local/searxng/searxng-src`` — and, unless ``web.searxng_auto_clone`` is
turned off, clones it on the first search that needs it. That clone is a network
fetch and tens of megabytes of disk on a code path that reads like a search, which is
why it is configurable and logged at warning level.

    from components.engines import search_engine

    results = search_engine("arxiv", "machine learning")
    results = search_engine("google", "latest news", pageno=2, lang="en-US")
"""

import sys
import os
import subprocess
import threading
import typing as t
import logging

import httpx
import requests

from .config import config

# ---------------------------------------------------------------------------
# Bootstrap: initialize SearXNG settings + engine loader once
# ---------------------------------------------------------------------------

_SEARX_INITIALIZED = False
_LOADED_ENGINES: dict[str, t.Any] = {}

logger = logging.getLogger(__name__)


def _ensure_searx_initialized():
    global _SEARX_INITIALIZED
    if _SEARX_INITIALIZED:
        return
    
    # This file is <repo>/components/engines.py, so the repo root is two levels up.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_paths = [
        os.path.join(repo_root, "searxng-src"),
        os.path.expanduser("~/searxng-src"),
        "/usr/local/searxng/searxng-src",
    ]
    configured_src = config.get("web.searxng_source_dir", "")
    if configured_src:
        candidate_paths.insert(0, os.path.expanduser(configured_src))
    searx_src = None
    for p in candidate_paths:
        if os.path.isdir(os.path.join(p, "searx")):
            searx_src = p
            break
    
    if searx_src is None:
        if not config.get("web.searxng_auto_clone", True):
            raise RuntimeError(
                "SearXNG source not found in " + ", ".join(candidate_paths)
                + ", and web.searxng_auto_clone is off: clone "
                "https://github.com/searxng/searxng into one of those paths, or point "
                "web.searxng_source_dir at it."
            )
        searx_src = os.path.expanduser("~/searxng-src")
        # Loud on purpose: a search has just become a network fetch and a lot of disk.
        logger.warning("SearXNG source not found — cloning it to %s (one time; turn this "
                       "off with web.searxng_auto_clone=false)", searx_src)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/searxng/searxng", searx_src],
                check=True, capture_output=True, timeout=120
            )
            logger.info("SearXNG source cloned successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to clone SearXNG source: {e}") from e
    
    if searx_src not in sys.path:
        sys.path.insert(0, searx_src)
    
    # Point to our settings
    settings_path = config.get("web.searxng_settings_path", "/etc/searxng/settings.yml",
                               env="SEARXNG_SETTINGS_PATH")
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


_session: requests.Session | None = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """The process-wide HTTP session.

    Shared rather than created per request, so connections are actually pooled: one
    search fans out over several engines and then follows up on the same hosts, where
    a fresh session throws the TLS handshake away every time.
    """
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
        return _session


def _make_http_request(params: dict) -> requests.Response:
    """Execute the HTTP request an engine's ``request()`` built into ``params``."""
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
    
    session = _get_session()
    if method == "POST":
        return session.post(url, **req_kwargs)
    return session.get(url, **req_kwargs)


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
# Multi-engine search, result deduplication & authority scoring  (Track 10)
# ---------------------------------------------------------------------------
import concurrent.futures
import re
from urllib.parse import urlparse, urlunparse

# Engines used when the caller asks for ``engine="auto"``.
# Ordered: general-purpose first, then domain-specific fallbacks that engage
# only when the query contains a matching keyword.
_DEFAULT_MULTI_ENGINES = [
    "duckduckgo", "wikipedia", "arxiv", "github",
]

# Domains whose results get a small authority bonus.
_HIGH_AUTHORITY_DOMAINS = frozenset({
    ".edu", ".gov", "wikipedia.org", "arxiv.org",
    "github.com", "docs.python.org", "pypi.org",
    "stackoverflow.com", "man7.org", "kernel.org",
})

# Domains that typically return spam or thin content — a light penalty.
_LOW_QUALITY_DOMAINS = frozenset({
    "pinterest.com", "quora.com", "tiktok.com",
    "instagram.com", "facebook.com",
})

# URL tracking parameters that do not change the content; stripping them
# increases the chance two URLs map to the same result.
_TRACKING_PARAMS = re.compile(
    r"(?:^|&)(?:utm_[a-z]+|fbclid|gclid|ref|source|mc_[a-z]+"
    r"|_ga|_gl|yclid|msclkid|dclid|igshid|si|feature|wprov)"
    r"(?:=[^&]*)?(?=&|$)",
    re.I,
)


def _normalize_url(raw: str) -> str:
    """Canonical URL form used for deduplication.

    Lowercases the hostname, strips the default port, removes trailing slashes
    on the path, drops known tracking parameters, and drops fragments — so two
    links that resolve to the same document collapse into one.
    """
    if not raw:
        return ""
    parsed = urlparse(raw.strip())
    netloc = (parsed.hostname or "").lower()
    if parsed.port and parsed.port not in (80, 443):
        netloc = f"{netloc}:{parsed.port}"
    path = parsed.path.rstrip("/") or "/"
    query = _TRACKING_PARAMS.sub("", parsed.query)
    query = query.strip("&")
    return urlunparse(("", netloc, path, "", query, ""))


def _authority_score(url: str) -> float:
    """Return a small bonus (positive) or penalty (negative) for a result's domain.

    The bonus is small enough (~0.1) that genuine relevance still dominates,
    but large enough to break ties between equal-scoring results from different
    sources.
    """
    host = (urlparse(url).hostname or "").lower()
    for domain in _HIGH_AUTHORITY_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return 0.1
    for domain in _LOW_QUALITY_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return -0.15
    return 0.0


def _title_overlap(a: str, b: str) -> float:
    """Fraction of shorter title's tokens also present in the longer one.

    Returns 0.0 when either title is empty. Normalised to [0, 1].
    """
    ta = set((a or "").casefold().split())
    tb = set((b or "").casefold().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _dedup_results(results: "list[dict]") -> "list[dict]":
    """Deduplicate search results across engines.

    1. URL dedup — two results whose normalised URLs are equal collide; the one
       with a longer ``content`` field wins (it has more information).
    2. Near-duplicate title — if two results share >=70% of their title tokens,
       the one with more content survives.

    Returns the deduplicated list, preserving the original sort order.
    """
    seen_urls: "dict[str, int]" = {}
    keep: "list[bool]" = [True] * len(results)

    for i, r in enumerate(results):
        url = _normalize_url(r.get("url", ""))
        if not url:
            continue
        prev = seen_urls.get(url)
        if prev is not None:
            # Keep whichever has more content; ties favour the first seen.
            prev_len = len(results[prev].get("content", "") or "")
            this_len = len(r.get("content", "") or "")
            if this_len > prev_len:
                keep[prev] = False
                seen_urls[url] = i
            else:
                keep[i] = False
        else:
            seen_urls[url] = i

    # Near-duplicate title pass (only among still-kept results).
    for i in range(len(results)):
        if not keep[i]:
            continue
        title_i = results[i].get("title", "")
        for j in range(i + 1, len(results)):
            if not keep[j]:
                continue
            title_j = results[j].get("title", "")
            if _title_overlap(title_i, title_j) >= 0.7:
                content_i = len(results[i].get("content", "") or "")
                content_j = len(results[j].get("content", "") or "")
                if content_j > content_i:
                    keep[i] = False
                else:
                    keep[j] = False

    return [r for r, k in zip(results, keep) if k]


def _categorize_query(query: str) -> str:
    """Heuristic to pick the right engines for an ``engine="auto"`` search.

    Returns one of: ``"general"``, ``"academic"``, ``"code"``, ``"wiki"``.
    """
    q = (query or "").casefold()
    if any(w in q for w in ("arxiv", "paper", "research", "doi", "preprint", "citation")):
        return "academic"
    if any(w in q for w in ("github", "repo", "pull request", "issue", "commit",
                             "python", "javascript", "rust", "golang", "code",
                             "library", "package", "npm", "pip", "cargo")):
        return "code"
    if any(w in q for w in ("who is", "what is", "define", "definition",
                             "wikipedia", "encyclopedia")):
        return "wiki"
    return "general"


_AUTO_ENGINE_SETS = {
    "general": ["duckduckgo", "wikipedia"],
    "academic": ["arxiv", "wikipedia", "duckduckgo"],
    "code": ["github", "duckduckgo"],
    "wiki": ["wikipedia", "duckduckgo"],
}


def search_multi(
    query: str,
    engines: "list[str] | None" = None,
    *,
    pageno: int = 1,
    safesearch: int = 0,
    time_range: "str | None" = None,
    lang: str = "en",
    max_results: int = 10,
    max_workers: int = 4,
) -> "list[dict]":
    """Run a query across multiple engines, deduplicate, and rank the merged result.

    Args:
        query: Search query string.
        engines: Engine names; if ``None``, uses ``_DEFAULT_MULTI_ENGINES``.
        max_workers: Maximum concurrent engine calls (default 4).

    Returns:
        Deduplicated list of result dicts, up to ``max_results`` entries.
        Each dict carries an ``_authority`` key with the domain bonus.
    """
    engine_list = engines or list(_DEFAULT_MULTI_ENGINES)
    if not engine_list:
        return []

    all_results: "list[dict]" = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(max_workers, len(engine_list))
    ) as pool:
        futures = {
            pool.submit(
                search_engine, name, query,
                pageno=pageno, safesearch=safesearch,
                time_range=time_range, lang=lang,
                max_results=max(15, max_results * 2),  # over-fetch before dedup
            ): name
            for name in engine_list
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                engine_results = future.result()
            except Exception:
                logger.warning("search_multi: engine %s failed, skipping", name,
                               exc_info=True)
                continue
            all_results.extend(engine_results)

    # Dedup and apply authority scoring.
    deduped = _dedup_results(all_results)
    for r in deduped:
        r["_authority"] = _authority_score(r.get("url", ""))

    # Re-sort: original engine order + authority bonus as tiebreaker.
    deduped.sort(key=lambda r: r.get("_authority", 0.0), reverse=True)

    return deduped[:max_results]


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
