"""Web tools for the agent: SearXNG-engine search + Jina Reader fetch,
backed by an on-disk scratchpad so large results don't have to live in the
model's context window.

Design: search/fetch return a SHORT preview for the model, and save the
full payload to a scratch file under a short id. The model can page through
more of it with read_scratch(id, offset) only if it actually needs to.
This is the key lever for keeping a small context window usable with
search/browse tasks: full pages never get force-fed into the chat history.
"""

import os
import re
import json
import time
import uuid
import requests

from engines import search_engine  # your existing module, same directory

SCRATCH_DIR = os.environ.get("AGENT_SCRATCH_DIR", "/tmp/agent_scratch")
os.makedirs(SCRATCH_DIR, exist_ok=True)

SEARCH_SNIPPET_CHARS = 160     # per-result snippet shown inline to the model
SEARCH_MAX_RESULTS = 5         # hard cap, regardless of what the model asks for
FETCH_INLINE_CHARS = 1800      # how much of a fetched page goes inline
FETCH_SAVE_CHARS = 20000       # how much of a fetched page we keep on disk at all
JINA_TIMEOUT = 25
JINA_RETRY_ON_429 = 1          # anonymous Jina reader is rate-limited; one retry


def _new_scratch_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def save_to_scratch(content: str, prefix: str = "doc") -> str:
    sid = _new_scratch_id(prefix)
    path = os.path.join(SCRATCH_DIR, f"{sid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content[:FETCH_SAVE_CHARS])
    return sid


def read_scratch(scratch_id: str, offset: int = 0, length: int = 1500) -> str:
    """Page through a previously saved search/fetch result."""
    path = os.path.join(SCRATCH_DIR, f"{scratch_id}.txt")
    if not os.path.exists(path):
        return f"(no scratch entry found for id '{scratch_id}')"
    with open(path, "r", encoding="utf-8") as f:
        f.seek(max(0, offset))
        chunk = f.read(length)
    total = os.path.getsize(path)
    end = offset + len(chunk)
    more = end < total
    tail = f"\n...(more available, call read_scratch with offset={end})" if more else ""
    return f"[scratch:{scratch_id} bytes {offset}-{end}/{total}]\n{chunk}{tail}"


def search_web(query: str, engine: str = "duckduckgo", max_results: int = 5) -> str:
    """Run a search through engines.py and return a compact, context-cheap
    summary. Full result set (titles, urls, content snippets) is saved to
    scratch for deep-diving via read_scratch."""
    max_results = min(int(max_results or 5), SEARCH_MAX_RESULTS)
    try:
        results = search_engine(engine, query, max_results=max_results)
    except Exception as e:
        return f"search error ({engine}): {e}"

    if not results:
        return f"no results from '{engine}' for: {query}"

    sid = save_to_scratch(json.dumps(results, indent=2, ensure_ascii=False), prefix="search")

    lines = [f"Search results for '{query}' via {engine} (full data at scratch:{sid}):"]
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

    sid = save_to_scratch(text, prefix="fetch")
    preview = text[:FETCH_INLINE_CHARS]
    truncated = len(text) > FETCH_INLINE_CHARS
    note = f"\n...(truncated, full page saved as scratch:{sid} — use read_scratch to page through it)" if truncated else ""
    return f"Content of {url}:\n{preview}{note}"
