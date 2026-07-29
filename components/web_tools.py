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

from .engines import search_engine  # sibling module in the components package
from .scratch import save_to_scratch, read_scratch  # noqa: F401 (re-exported)

SEARCH_SNIPPET_CHARS = 160     # per-result snippet shown inline to the model
SEARCH_MAX_RESULTS = 5         # hard cap, regardless of what the model asks for
FETCH_INLINE_CHARS = 1800      # how much of a fetched page goes inline
FETCH_SAVE_CHARS = 20000       # how much of a fetched page we keep on disk at all
JINA_TIMEOUT = 25
JINA_RETRY_ON_429 = 1          # anonymous Jina reader is rate-limited; one retry


def search_web(query: str, engine: str = "duckduckgo", max_results: int = 5) -> str:
    """Run a search through engines.py and return a compact, context-cheap
    summary. Full result set (titles, urls, content snippets) is saved to
    scratch for deep-diving via read_scratch. Automatically falls back to
    alternate engines if the primary returns no results."""
    max_results = min(int(max_results or 5), SEARCH_MAX_RESULTS)
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
        max_bytes=FETCH_SAVE_CHARS,
    )

    lines = [f"Search results for '{query}' via {used_engine} (full data at scratch:{sid}):"]
    for i, r in enumerate(results, 1):
        snippet = (r.get("content") or "")[:SEARCH_SNIPPET_CHARS].replace("\n", " ")
        lines.append(f"{i}. {r.get('title', '(no title)')} — {r.get('url', '')}\n   {snippet}")
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

    sid = save_to_scratch(text, prefix="fetch", max_bytes=FETCH_SAVE_CHARS)
    preview = text[:FETCH_INLINE_CHARS]
    truncated = len(text) > FETCH_INLINE_CHARS
    note = f"\n...(truncated, full page saved as scratch:{sid} — use read_scratch to page through it)" if truncated else ""
    return f"Content of {url}:\n{preview}{note}"
