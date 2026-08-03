"""Web tools for the agent: SearXNG-engine search + Jina Reader fetch,
backed by an on-disk scratchpad so large results don't have to live in the
model's context window.

Design: search/fetch return a SHORT preview for the model, and save the
full payload to a scratch file under a short id. The model can page through
more of it with read_scratch(id, offset) only if it actually needs to.
This is the key lever for keeping a small context window usable with
search/browse tasks: full pages never get force-fed into the chat history.
"""

import re
import json
import time
import requests

from .config import config
from .engines import search_engine, search_multi, _categorize_query, _AUTO_ENGINE_SETS
from .console import search_progress
from .scratch import save_to_scratch, read_scratch  # noqa: F401 (re-exported)

# per-result snippet shown inline to the model
SEARCH_SNIPPET_CHARS = config.get("web.search_snippet_chars", 160)
# hard cap, regardless of what the model asks for
SEARCH_MAX_RESULTS = config.get("web.search_max_results", 5)
# how much of a fetched page goes inline; the whole page always reaches scratch
FETCH_INLINE_CHARS = config.get("web.fetch_inline_chars", 900)
JINA_TIMEOUT = config.get("web.jina_timeout", 25)
# anonymous Jina reader is rate-limited; one retry
JINA_RETRY_ON_429 = config.get("web.jina_retry_on_429", 1)


def search_web(query: str, engine: str = "duckduckgo", max_results: int = 5) -> str:
    """Run a search through engines.py and return a compact, context-cheap
    summary. Full result set (titles, urls, content snippets) is saved to
    scratch for deep-diving via read_scratch.

    Use ``engine="auto"`` to fan out to multiple engines (duckduckgo, wikipedia,
    arxiv, etc.), deduplicate, and rank the merged results automatically.
    """
    max_results = min(int(max_results or 5), SEARCH_MAX_RESULTS)

    if engine == "auto":
        category = _categorize_query(query)
        engines = _AUTO_ENGINE_SETS.get(category, ["duckduckgo"])
        results = search_multi(query, engines=engines, max_results=max_results,
                              progress_callback=search_progress)
        used_engine = f"auto ({category}, {', '.join(engines)})"
    else:
        engines_to_try = [engine] + [e for e in ["duckduckgo", "searxng", "bing", "google"] if e != engine]
        results = None
        used_engine = engine
        last_error = None
        for eng in engines_to_try:
            try:
                res = search_engine(eng, query, max_results=max_results)
                if res:
                    results = res
                    used_engine = eng
                    break
            except Exception as e:
                if eng == engine:
                    last_error = e
                continue

        if not results:
            if last_error:
                return f"search error ({engine}): {last_error}"
            return f"no results found for: {query}"

    sid = save_to_scratch(
        json.dumps(results, indent=2, ensure_ascii=False),
        prefix="search",
    )

    lines = [f"Search results for '{query}' via {used_engine} (full data at scratch:{sid}):"]
    for i, r in enumerate(results, 1):
        snippet = (r.get("content") or "")[:SEARCH_SNIPPET_CHARS].replace("\n", " ")
        authority = r.get("_authority", 0.0)
        authority_note = " [authoritative]" if authority > 0 else ""
        lines.append(f"{i}. {r.get('title', '(no title)')} — {r.get('url', '')}{authority_note}\n   {snippet}")
    return "\n".join(lines)


def fetch_url(url: str) -> str:
    """Fetch a page via Jina Reader (https://r.jina.ai, no API key needed —
    anonymous use is rate-limited) and return a truncated, context-cheap
    preview. Full extracted text is saved to scratch."""
    if not re.match(r"^https?://", url):
        url = "https://" + url
    reader_url = f"https://r.jina.ai/{url}"

    attempts = JINA_RETRY_ON_429 + 1
    resp = None
    for i in range(attempts):
        try:
            resp = requests.get(
                reader_url,
                headers={"Accept": "text/plain", "X-Return-Format": "markdown"},
                timeout=JINA_TIMEOUT,
            )
            if resp.status_code == 429 and i < attempts - 1:
                time.sleep(2)
                continue
            resp.raise_for_status()
            break
        except Exception as e:
            if i == attempts - 1:
                return f"fetch error for {url}: {e}"
            time.sleep(1)

    text = (resp.text or "").strip()
    if not text:
        return f"fetch returned empty content for {url}"

    sid = save_to_scratch(text, prefix="fetch")
    preview = text[:FETCH_INLINE_CHARS]
    truncated = len(text) > FETCH_INLINE_CHARS
    note = f"\n...(truncated, full page saved as scratch:{sid} — use read_scratch to page through it)" if truncated else ""
    return f"Content of {url}:\n{preview}{note}"
