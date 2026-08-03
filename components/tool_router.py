"""Which tools get bound to the model this turn.

Every bound tool costs its JSON schema in the prompt, on every step, for the
whole session. Worse than the tokens: a small model picks worse from a long
menu, so a twenty-tool list hurts accuracy exactly where the budget is
tightest. Core tools are always bound; the rest are bound only when the turn
looks like it needs them, or once they have already been used in it (a
multi-round task must keep the tool it started with).

Two signals decide what "looks like it needs them":

* **Keyword triggers** (always on) — regex patterns matched against the turn's
  text. Fast, zero-inference, works on any word the model is likely to type.
* **Embedding similarity** (opt-out via ``tools.embedding_routing``) — the
  already-loaded MiniLM model embeds the turn text and each tool's description,
  and the cosine distance between them is how "check what's stored for auth"
  binds ``vault`` while "vault credential" already matched the regex.

The two signals are unioned, so embedding routing is additive — it cannot
*unbind* a tool the regex already pulled in.  The regex stays because it is the
fast path and is precise for well-worn tool names, while embedding routing
covers the tail of fuzzy requests.

Set ``tools.dynamic_binding`` to false to bind everything.
"""

import logging
import re
import threading

from .config import config

logger = logging.getLogger(__name__)

DYNAMIC_BINDING = config.get("tools.dynamic_binding", True)

# Always bound. ``recall`` is here despite being a memory tool: the system
# prompt requires a recall before any answer that depends on what we know, so
# leaving it unbound would silently disable that rule.
CORE_TOOLS = config.get("tools.core", [
    "execute_shell_command",
    "read_any_file",
    "write_any_file",
    "patch_file",
    "search_web",
    "recall",
])

_TASKS = (r"background|long[- ]running|server|daemon|watch|tail|"
          r"start .*(?:server|process)|\btask\b")

# Situational tools and the words that pull them in. Keys are tool names.
TRIGGERS = {
    "vault": r"vault|credential|secret|api[ _-]?key|token|password|passphrase|env var",
    "remember": r"remember|memoriz|keep in mind|note that|for future|i prefer|my name",
    "git_diff": r"\bgit\b|\bdiff\b|commit|staged|unstaged|working tree|repo",
    "batch_patch": r"batch|refactor|rename|across .*files|every file|all files",
    "fetch_url": r"https?://|www\.|\.com\b|\.org\b|url|link|web ?page|fetch|article|docs?\b",
    "find_in_files": r"grep|search|find|where is|which file|usage|references|occurrenc|\bTODO\b",
    "read_many_files": r"read .*files|several files|these files|whole (?:dir|folder|module)|overview",
    "glob_list": r"list .*files|\*\.|glob|directory|folder contents|tree\b|what files",
    "task_start": _TASKS,
    "task_list": _TASKS,
    "task_status": _TASKS,
    "task_output": _TASKS,
    "task_kill": _TASKS,
    "read_scratch": r"scratch:|scratch id|full (?:output|file|page|diff)|show me the rest|page (?:through|more)",
}

_COMPILED = {name: re.compile(pattern, re.I) for name, pattern in TRIGGERS.items()}

# ---------------------------------------------------------------------------
# Embedding-based routing
# ---------------------------------------------------------------------------
EMBEDDING_ROUTING = config.get("tools.embedding_routing", True)
EMBEDDING_THRESHOLD = config.get("tools.embedding_threshold", 0.35)

# Cached on first use so importing this module stays cheap.
_embeddings_model = None
_embeddings_lock = threading.Lock()
_desc_vectors: "dict[str, list[float]]" = {}   # tool_name -> normalised vector


# First-sentence descriptions of every tool, for embedding similarity matching.
# These are what the turn text is compared against.  The descriptions are kept
# short (the first line of each tool's docstring in ``langbot.py``) so that the
# embedding focuses on *what the tool does*, not how to use it.
_TOOL_DESCRIPTIONS = {
    "execute_shell_command": "Execute a shell command synchronously and return its output.",
    "read_any_file": "Read any text file; binary files are reported by size.",
    "write_any_file": "Write content to any file, overwrite or append.",
    "patch_file": "Surgically replace text in a file without rewriting it.",
    "batch_patch": "Apply multiple find-and-replace patches across files at once.",
    "git_diff": "Show the git diff for a file or directory.",
    "find_in_files": "Search for a text pattern recursively across files.",
    "read_many_files": "Read multiple files matching a glob pattern at once.",
    "glob_list": "List files matching a glob pattern with their sizes.",
    "task_start": "Start a long-running command as a managed background task.",
    "task_list": "List all background tasks and their statuses.",
    "task_status": "Show the current status of a specific background task.",
    "task_output": "Read the captured output of a background task.",
    "task_kill": "Terminate a running background task.",
    "search_web": "Search the web via search engines for current information.",
    "fetch_url": "Fetch and read the contents of a web page by URL.",
    "read_scratch": "Read a portion of a large result saved to the scratchpad.",
    "remember": "Store a durable fact in long-term memory for future sessions.",
    "recall": "Search long-term memory for facts relevant to a query.",
    "vault": "Store, retrieve, list, or delete encrypted credentials.",
}


def _get_embeddings_model():
    """Return the MiniLM embedding model, loading it (once) under a lock."""
    global _embeddings_model
    with _embeddings_lock:
        if _embeddings_model is not None:
            return _embeddings_model
        # The model is already loaded by memory_store, but tool_router may be
        # called before memory is warmed.  Import it lazily so the module
        # itself stays cheap.
        from .memory_store import get_embeddings

        _embeddings_model = get_embeddings(announce=False)
        return _embeddings_model


def register(tool_descriptions: "dict[str, str]", triggers: "dict[str, str]") -> None:
    """Register external (plugin) tools for routing.

    Called once at startup. Tool descriptions are added to the embedding index;
    triggers are compiled and added to the keyword matcher.
    Only *new* entries are added — existing entries are not overwritten, so
    built-in tools always take precedence over plugins with the same name.

    Args:
        tool_descriptions: ``{tool_name: description}`` for embedding similarity.
        triggers: ``{tool_name: regex_pattern}`` for keyword routing.
    """
    for name, desc in (tool_descriptions or {}).items():
        if name not in _TOOL_DESCRIPTIONS:
            _TOOL_DESCRIPTIONS[name] = desc

    for name, pattern in (triggers or {}).items():
        if name not in _COMPILED:
            _COMPILED[name] = re.compile(pattern, re.I)
    # Invalidate cached description vectors so they are recomputed on the next
    # embedding lookup (which will now include the plugin descriptions).
    _desc_vectors.clear()
    logger.debug(
        "tool_router: registered %d plugin description(s), %d trigger(s)",
        len(tool_descriptions or {}), len(triggers or {}),
    )


def _ensure_desc_vectors() -> None:
    """Pre-compute description embeddings on first use."""
    if _desc_vectors:
        return
    model = _get_embeddings_model()
    names = list(_TOOL_DESCRIPTIONS.keys())
    descs = [_TOOL_DESCRIPTIONS[n] for n in names]
    vectors = model.embed_documents(descs)
    for name, vec in zip(names, vectors):
        _desc_vectors[name] = vec


def _cosine(a, b) -> float:
    """Cosine similarity of two equal-length vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _embedding_tool_names(turn_text: str, threshold: float = EMBEDDING_THRESHOLD) -> "set[str]":
    """Tools the turn text is semantically close to, above *threshold*.

    Returns the empty set (not None) when embedding routing is disabled, the
    model isn't loaded yet, or no description is above the threshold.
    """
    if not EMBEDDING_ROUTING:
        return set()
    if not (turn_text or "").strip():
        return set()
    try:
        model = _get_embeddings_model()
        _ensure_desc_vectors()
        query_vec = model.embed_query(turn_text.strip())
    except Exception:
        logger.debug("tool_router: embedding lookup failed, falling back to regex only",
                     exc_info=True)
        return set()

    selected: "set[str]" = set()
    for name, desc_vec in _desc_vectors.items():
        sim = _cosine(query_vec, desc_vec)
        if sim >= threshold:
            selected.add(name)
            logger.debug("tool_router: embedding bound %s (sim=%.3f)", name, sim)
    return selected


# ---------------------------------------------------------------------------
# Turn text extraction
# ---------------------------------------------------------------------------

def _turn_text(messages) -> str:
    """Text the triggers are matched against: the current turn, or its tail.

    Tool results are included so a tool can pull in its natural follow-up — a
    ``search_web`` result contains urls, which is what binds ``fetch_url``.
    """
    last_human = -1
    for i, msg in enumerate(messages):
        if getattr(msg, "type", None) == "human":
            last_human = i
    window = messages[last_human:] if last_human >= 0 else messages[-4:]
    parts = []
    for msg in window:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content:
            parts.append(content[:2000])
    return "\n".join(parts)


def _already_used(messages) -> "set[str]":
    """Tools called since the last human message."""
    last_human = -1
    for i, msg in enumerate(messages):
        if getattr(msg, "type", None) == "human":
            last_human = i
    used = set()
    for msg in messages[last_human + 1:] if last_human >= 0 else messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                used.add(name)
    return used


def select_tool_names(messages) -> "set[str]":
    """Names of the tools to bind for this step.

    The set is the union of:

    1. Always-bound core tools,
    2. Tools already called this turn,
    3. Tools whose keyword trigger matches the turn text, and
    4. Tools whose description embedding is close to the turn text.

    Step 4 is additive: it can only *add* tools, never remove a regex match.
    If embedding routing is disabled (``tools.embedding_routing: false``), step 4
    returns the empty set and the behaviour is identical to the pre-embedding
    tool_router.
    """
    selected = set(CORE_TOOLS) | _already_used(messages)
    text = _turn_text(messages)

    # Keyword triggers (always on).
    for name, pattern in _COMPILED.items():
        if pattern.search(text):
            selected.add(name)

    # Embedding similarity (configurable, additive).
    selected |= _embedding_tool_names(text)

    return selected


def select_tools(all_tools, messages) -> list:
    """Subset of ``all_tools`` to bind, preserving their declared order."""
    if not DYNAMIC_BINDING:
        return list(all_tools)
    names = select_tool_names(messages)
    chosen = [t for t in all_tools if t.name in names]
    # A model with no tools cannot act; fall back to the full set rather than
    # letting a misconfigured core list strand the agent.
    return chosen or list(all_tools)
