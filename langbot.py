import os

# Silence HuggingFace / transformers / tokenizers progress bars and chatter
# *before* those libraries are imported so the embedding model loads quietly
# (no "Loading weights: 100%|█| 103/103 ..." lines on the console).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import subprocess
import logging
import time
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path

# Load .env (API keys, base URLs, etc.) before anything reads os.environ —
# the config layer, vault bootstrap, and LLM construction all consult it.

# python-dotenv is optional: if it is missing, the agent still runs with
# whatever is already in the environment (and a one-line warning).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.getLogger(__name__).warning(
        "python-dotenv not installed — .env file (if any) is not loaded; "
        "install it via 'pip install python-dotenv' or set env vars directly"
    )

logger = logging.getLogger(__name__)

# Send every component's log records to ./memory/langbot.log before anything can
# emit one: unconfigured logging prints to stderr, i.e. into the middle of the
# REPL's panels and prompt (see components/logging_setup.py).
from components.logging_setup import log_path as _log_path, setup as _setup_logging

_setup_logging()

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ModuleNotFoundError:
    from langgraph.checkpoint.memory import MemorySaver
    SQLITE_AVAILABLE = False
    logger.warning("langgraph-checkpoint-sqlite not installed - conversation history "
                   "will not persist")

from components.web_tools import search_web as _search_web, fetch_url as _fetch_url
from components.scratch import offload as _offload, read_scratch as _read_scratch
from components.utils import MAX_OUTPUT_CHARS, truncate
from components.truncate import maybe_truncate as _maybe_truncate
from components.journal import (
    Event as _JournalEvent,
    Journal as _Journal,
    heuristic_title as _heuristic_title,
    journals_dir as _journals_dir,
    resolve_session_key as _resolve_session_key,
)
# Aliased: `config` is the per-thread graph config in this module's REPL helpers.
from components.config import CONFIG_ENV_VAR, CONFIG_FILENAME, config as app_config
from components.memory_store import (
    count as _memory_count,
    get_collection as _memory_collection,
    get_embeddings as _load_embeddings,
    search_memories as _search_memories,
    store_memory as _store_memory,
)
from components.memory_worker import DistillJob, MemoryWorker, parse_fact_entries
from components.fallback_llm import build as _build_distill_llm
from components.fallback_llm import FallbackLLM, DEFAULT_TIERS, DEFAULT_COOLDOWN as _FALLBACK_COOLDOWN
from components.warmup import Warmup
from components.file_ops import (
    read_file as _read_file,
    write_file as _write_file,
    patch_file as _patch_file,
    batch_patch as _batch_patch,
    git_diff as _git_diff,
)
from components.code_search import (
    find_in_files as _find_in_files,
    read_many_files as _read_many_files,
    glob_list as _glob_list,
)
from components import tasks as _tasks
from components import (
    code_search as _code_search,
    file_ops as _file_ops,
    memory_store as _memory_store,
    memory_worker as _memory_worker_mod,
    scratch as _scratch,
    web_tools as _web_tools,
)
from components import context_budget as _ctx
from components import housekeeping as _housekeeping
from components.tool_router import (
    select_tools as _select_tools,
    register as _register_plugin_tools,
    register_tool_meta as _register_tool_meta,
    tool_safety as _tool_safety,
    tool_tier as _tool_tier,
    max_tier as _max_tier,
    SAFETY_READ as _SAFETY_READ,
    SAFETY_WRITE as _SAFETY_WRITE,
    SAFETY_EXEC as _SAFETY_EXEC,
    TIER_CORE as _TIER_CORE,
    TIER_SERVER as _TIER_SERVER,
)
from components.safety import (
    catastrophic_reason as _catastrophic_reason,
    SafetyGate as _SafetyGate,
)
from tools.plugins import discover_plugins
from components.routing import (
    RECURSION_LIMIT,
    NUDGE_MARKER,
    is_nudge,
    nudge_agent,
    route_agent,
    split_repeated_calls,
)
from components.stuck import StuckDetector as _StuckDetector

from components.reflex import ReflexStore as _ReflexStore
from components.session_tools import (
    journal_search as _journal_search,
    session_list as _session_list,
    session_search as _session_search,
    set_current_journal as _set_current_journal,
)
from components.tool_call_repair import repair_message, stats as _repair_stats
from components.context import (
    KernelContext as _KernelContext,
    context_from_config as _context_from_config,
    stash_context as _stash_context,
)

import components.console as ui
from components.input import read_input, setup_readline
from components.vault import (
    bootstrap as _vault_bootstrap,
    run_action as _vault_run,
    redact as _vault_redact,
    save as _vault_save,
    _vault_list,
    _vault_status,
)

# ------------------------------------------------------------------------------
# Configuration — every value below has a working default, so ./langbot.config.json
# (see components/config.py for the search order) is entirely optional.
# ------------------------------------------------------------------------------
BASE_URL = app_config.get("llm.base_url", "http://127.0.0.1:8080/v1",
                             env="LLM_BASE_URL")
LLM_MODEL = app_config.get("llm.model", "local-model", env="LLM_MODEL")
LLM_API_KEY = app_config.get("llm.api_key", "not-needed", env="LLM_API_KEY")
LLM_TEMPERATURE = app_config.get("llm.temperature", 0.1, env="LLM_TEMPERATURE")
LLM_MAX_RETRIES = app_config.get("llm.max_retries", 10)
THINKING_MODE = app_config.get("llm.thinking_mode", "auto")
SQLITE_DB_PATH = app_config.get("paths.checkpoint_db", "./memory/agent_checkpoints.db")

# ------------------------------------------------------------------------------
# 0. Credential Vault — load stored secrets into the environment before the LLM
#    and tools are constructed, so *_API_KEY values are available to them.
# ------------------------------------------------------------------------------
_VAULT_ENV_LOADED = _vault_bootstrap()

# Procedural reflex store (Track D of docs/neo-port-plan.md) — deterministic
# when-X → do-Y rules matched before the LLM is ever called.  Rules persist
# under ./memory/reflexes.json (see components/reflex.py).
_reflex_store = _ReflexStore()

# ------------------------------------------------------------------------------
# 1. LLM & Embeddings
# ------------------------------------------------------------------------------
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=BASE_URL,
    api_key=LLM_API_KEY,
    temperature=LLM_TEMPERATURE,
    max_retries=LLM_MAX_RETRIES,
)

# The embedding model and the Chroma collection cost seconds to build. Warm them
# on a background thread from main() so the REPL is usable immediately; the memory
# tools do not consult this, since memory_store loads on first use under its own
# lock either way (see components/warmup.py).
# The start-up disk sweep rides along here for the same reason: it must not be on
# the interactive loop, and it is the only other thing that wants to run once per
# start. It goes first so it is done before the checkpointer is busy.
_active_thread_id = None
_sweep_summary = "pending"


def _sweep_disk() -> None:
    global _sweep_summary
    _sweep_summary = _housekeeping.sweep(
        _scratch.SCRATCH_DIR, SQLITE_DB_PATH, _active_thread_id
    )


_warmup = Warmup({
    "housekeeping": _sweep_disk,
    "embeddings": lambda: _load_embeddings(announce=False),
    "memory store": _memory_collection,
})

# ------------------------------------------------------------------------------
# 2. Semantic Memory Store (components/memory_store.py) + background distiller
# ------------------------------------------------------------------------------
# Distillation runs on its own tier chain (hosted free-tier models first, this
# local model last), because small local models are the weakest link at returning
# the strict JSON the distiller needs. See components/fallback_llm.py.
_distill_llm = _build_distill_llm(
    llm,
    validate=lambda text: parse_fact_entries(text) is not None,
)
_memory_worker = MemoryWorker(llm=_distill_llm)

# ------------------------------------------------------------------------------
# 2b. Summarizer LLM (Groq tiers → local) — used for history compaction.
# ------------------------------------------------------------------------------
# Compaction summarises older messages into a rolling summary that the system
# prompt carries forward.  The same tiered-fallback pattern used for distillation
# is applied here: hosted models handle the few sentences of prose that a
# compaction produces, falling back to the local model when Groq is unavailable.
# Tiers, temperature, and cooldown read from the same ``distill.*`` config keys;
# override ``summarize.*`` to split them.
_summarize_llm = FallbackLLM(
    llm,
    tiers=app_config.get("summarize.tiers",
                          app_config.get("distill.tiers", DEFAULT_TIERS)),
    validate=lambda text: bool(text and len(text.strip()) > 20),
    cooldown=app_config.get("summarize.cooldown_seconds",
                             app_config.get("distill.cooldown_seconds",
                                            _FALLBACK_COOLDOWN)),
    temperature=app_config.get("summarize.temperature",
                                app_config.get("distill.temperature", 0.0)),
    timeout=app_config.get("summarize.timeout",
                            app_config.get("distill.timeout", 30.0)),
)

# ------------------------------------------------------------------------------
# 2c. Durable Journal (Track B) — audit/observability layer
# ------------------------------------------------------------------------------
# The journal is an append-only event log per session under ./memory/sessions/.
# It is *not* the source of truth (the LangGraph checkpoint DB is); it exists
# so past sessions can be browsed by title/activity and searched by keyword, even
# after housekeeping prunes the checkpoint rows.  Journaling is best-effort:
# it must never break the agent loop, so every append swallows its exceptions.

_current_journal = None
_turn = 0


def _journal_event(type_: str, source: str = "system", **data) -> None:
    """Append an event to the current session's journal. Best-effort: journaling
    must never break the agent loop, so exceptions are swallowed.."""
    j = _current_journal
    if j is None:
        return
    try:
        j.log.append(_JournalEvent(type=type_, data=data, source=source, turn=_turn))
    except Exception:  # noqa: BLE001 — journaling must never break the loop
        logger.debug("journal: append failed", exc_info=True)


def _journal_set_title(text: str) -> None:
    """Set the session title from the first user message (no LLM call.."""
    j = _current_journal
    if j is None:
        return
    try:
        if j.meta().get("title") is None:
            j.set_title(_heuristic_title(text))
    except Exception:  # noqa: BLE001 — journaling must never break the loop
        logger.debug("journal: set_title failed", exc_info=True)


def _start_journal(thread_id: str) -> None:
    """Create (or replace) the journal for a thread and make it current..
    Called from main() at start and from /new when the thread changes.."""
    global _current_journal
    try:
        key, meta_extra = _resolve_session_key()
        _current_journal = _Journal.create(name=f"session:{thread_id}",
                                            encrypt_key=key,
                                            meta_extra=meta_extra)
    except Exception:  # noqa: BLE001 — journaling must never break the loop
        logger.debug("journal: create failed", exc_info=True)
        _current_journal = None


def _format_sessions(limit: int = 20) -> str:
    """List past sessions, newest by activity — the /sessions slash command."""
    try:
        metas = _Journal.list()
    except Exception as e:  # noqa: BLE001 — listing is user-facing
        return f"[error listing sessions] {e}"
    if not metas:
        return "(no sessions found)"
    lines = []
    for m in metas[:limit]:
        sid = m.get("id", "?")
        title = m.get("title") or "(untitled)"
        created = m.get("created_at", "?")
        name = m.get("name", "?")
        n_events = 0
        finished = False
        try:
            j = _Journal.load(sid)
            evs = j.events()
            n_events = len(evs)
            finished = any(e.type == "finish" for e in evs[-5:])
        except Exception:  # noqa: BLE001 — a corrupt session must not break the list
            pass
        status = "done" if finished else "active/incomplete"
        lines.append(f"  {sid}  {title[:50]:50s}  {created}  "
                     f"{n_events:5d} events  [{status}]  ({name})")
    return (f"Sessions ({len(metas)} total, showing {min(limit, len(metas))}):\n"
            + "\n".join(lines))


def _search_session(session_id: str, query: str, n: int = 10) -> str:
    """Search a past session's events by keyword — the /session slash command..
    Handles encrypted sessions transparently (the passphrase from
    LANGBOT_SESSION_ENCRYPT re-derives the key from the session's stored salt)..
    """
    try:
        key, _ = _resolve_session_key(journal_id=session_id)
        j = _Journal.load(session_id, encrypt_key=key)
        evs = j.events()
    except FileNotFoundError:
        return f"Session {session_id} not found"
    except Exception as e:  # noqa: BLE001 — search errors are user-facing
        return f"[error reading session] {e}"
    query_lower = query.lower()
    matches = []
    for ev in evs:
        d = ev.data if isinstance(ev.data, dict) else {}
        text = " ".join(str(v) for v in d.values()
                        if isinstance(v, (str, int, float)))
        if query_lower in text.lower() or query_lower in ev.type.lower():
            matches.append(ev)
    if not matches:
        return f"No events matching '{query}' in session {session_id}"
    lines = []
    for ev in matches[:n]:
        d = ev.data if isinstance(ev.data, dict) else {}
        seq = ev.seq
        if ev.type == "user_message":
            content = d.get("text", "")[:200]
        elif ev.type == "tool_result":
            content = f"[{d.get('tool', '?')}] {str(d.get('preview', ''))[:200]}"
        elif ev.type == "llm_response":
            content = d.get("content", "")[:200]
            calls = d.get("tool_calls", [])
            if calls:
                names = [c.get("function", {}).get("name", "?") for c in calls]
                content += f" calls={names}"
        elif ev.type == "agent_message":
            content = d.get("text", "")[:200]
        elif ev.type == "condensation":
            kind = d.get("kind", "?")
            summary = d.get("summary", "")
            content = f"[{kind}] {summary[:200]}" if summary else f"[{kind}]"
        else:
            content = str(d)[:200]
        lines.append(f"  [{seq}] {ev.type}: {content}")
    header = f"Session {session_id}: {len(matches)} matches for '{query}'"
    if len(matches) > n:
        header += f" (showing first {n})"
    return header + "\n" + "\n".join(lines)


def _extract_summary_facts(summary_text: str) -> str:
    """Extract key facts from an existing summary before recompacting it.

    When a rolling summary is recompacted (the thread has grown past the budget
    again), the new summarizer sees the old summary plus new older messages.
    Without this pass, facts captured in the first compaction can be silently
    dropped — the summarizer has no way to know which details from a paragraph
    of prose are essential.

    This extraction uses the same tiered summmarizer so that Groq's models handle
    it when available; a failure here is not fatal (compaction proceeds without
    preserved facts).
    """
    prompt = (
        "Extract the essential facts from this session summary as a short "
        "bullet list. Include: decisions made, file paths, commands used, "
        "project names, identifiers, preferences, and anything the assistant "
        "will need to remember in future turns. One fact per line, no preamble, "
        "no markdown formatting.\n\n"
        f"Summary:\n{summary_text}"
    )
    try:
        response = _summarize_llm.invoke(prompt)
        return (response.content or "").strip()
    except Exception:  # noqa: BLE001 — fact extraction is best-effort
        logger.warning("compact: fact extraction failed, recompacting "
                       "without preserved facts")
        return ""

# ------------------------------------------------------------------------------
# 3. Tools (original + memory)
# ------------------------------------------------------------------------------
def _resolve_ctx(config: "RunnableConfig | None" = None) -> "_KernelContext | None":
    """Resolve the per-run ``KernelContext`` from a tool's injected ``config``.

    langgraph auto-injects ``config: RunnableConfig`` into any tool (or node) that
    declares it, stripping it from the OpenAI schema sent to the model.  The
    ``tools_node`` stashes a ``KernelContext`` under
    ``config["configurable"]["kernel_context"]`` before the ToolNode runs; tools
    resolve it from there.  When absent (direct ``.func(...)`` test calls,
    the REPL's slash handlers, or a graph run that predates Track G), ``None``
    is returned and callers fall back to the process-wide singletons — so the
    context is purely additive and a revert is mechanical."""
    if not config:
        return None
    return (config.get("configurable") or {}).get("kernel_context")
@tool
def remember(fact: str, tags: "list[str] | None" = None) -> str:
    """Store one durable fact in long-term memory.

    Args:
        fact: a self-contained statement worth knowing in future sessions
            (e.g. "the langbot repo lives at ~/ai/repos/langbot"). Greetings,
            small talk, and anything true only of this turn do not belong here.
        tags: optional short category words for the fact (e.g. ["preference"],
            ["filesystem", "project"]); searchable later via `recall("#tag")`.

    Facts that duplicate something already stored are not stored twice.
    """
    try:
        before = _memory_count()
        mem_id = _store_memory(fact, tags=tags)
        stored = "Memory stored" if _memory_count() > before else "Already remembered"
        return f"{stored} (id {mem_id}): {truncate(fact, 200)}"
    except Exception as e:  # noqa: BLE001 — memory store errors are user-facing
        logger.debug("langbot: store_memory failed", exc_info=True)
        return f"Failed to store memory: {e}"

@tool
def recall(query: str, n: int = 3) -> str:
    """Search long-term memory for facts relevant to a query.

    Call this before answering anything that depends on what you already know about
    the user or this machine: preferences, project and file locations, credentials'
    names, past decisions, earlier findings.

    Args:
        query: what to look for, in words (e.g. "where does the langbot repo live").
            A "#tag" token (e.g. "#preference") matches facts carrying that tag.
        n: how many facts to return at most (default 3).

    Only facts above a relevance threshold are returned, each with a similarity
    score; an empty result means memory holds nothing relevant, not that the search
    failed. Use `remember` to store a new fact.
    """
    try:
        memories = _search_memories(query, n)
        if not memories:
            return ("No memory is relevant to that query "
                    "(nothing above the relevance threshold).")
        return "Relevant memories:\n" + "\n".join(
            f"- {m.text}  [relevance {m.score:.2f}"
            + (" " + " ".join(f"#{t}" for t in m.tags) if m.tags else "")
            + "]"
            for m in memories
        )
    except Exception as e:  # noqa: BLE001 — recall errors are user-facing
        logger.debug("langbot: recall failed", exc_info=True)
        return f"Failed to recall memories: {e}"

@tool
def execute_shell_command(command: str, cwd: str = "", timeout: int = 120) -> str:
    """Execute a shell command synchronously and return its output.

    Optionally run in ``cwd`` with a custom ``timeout`` (seconds; 0 = no limit).
    For servers, watchers, or anything long-running, use 'task_start' instead so
    the process is tracked and can be inspected or killed.

    Long output is previewed inline, with the whole of it saved to scratch and
    reachable via 'read_scratch'.
    """
    # --- Hard refusal for catastrophic commands (whole-system, irreversible) ---
    _reason = _catastrophic_reason(command)
    if _reason:
        logger.warning("catastrophic_command_blocked: refused %r (%s)", command, _reason)
        return f"Refused: command not executed ({_reason})."

    # --- T4. Blast-radius pattern warning check ---
    blast_patterns = app_config.get(
        "tools.blast_radius_patterns",
        [r"rm\s+-rf\b", r"rm\s+-r\s+-f\b", r"git\s+push\s+.*--force\b", r"git\s+push\s+.*-f\b", r"DROP\s+TABLE\b", r"DROP\s+DATABASE\b"]
    )
    is_destructive = False
    matched_pattern = ""
    for pat in blast_patterns:
        if re.search(pat, command, re.I):
            is_destructive = True
            matched_pattern = pat
            break

    if is_destructive:
        logger.warning("blast_radius: dangerous command execution flagged: %r (matched pattern: %r)", command, matched_pattern)

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout if timeout and timeout > 0 else None,
            cwd=cwd or None,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
        if result.returncode:
            output += f"\n[Exit code: {result.returncode}]"
        
        warning_prefix = "⚠️ destructive command detected\n" if is_destructive else ""

        if not output:
            return warning_prefix + f"Command '{command}' executed successfully."
        
        full_result = _offload(output, prefix="shell", inline_chars=MAX_OUTPUT_CHARS,
                               label="full output")
        return warning_prefix + full_result
    except subprocess.TimeoutExpired:
        warning_prefix = "⚠️ destructive command detected\n" if is_destructive else ""
        return warning_prefix + f"Timeout ({timeout}s): '{command}'"
    except Exception as e:  # noqa: BLE001 — shell errors are user-facing
        logger.debug("langbot: shell command failed: %s", command, exc_info=True)
        warning_prefix = "⚠️ destructive command detected\n" if is_destructive else ""
        return warning_prefix + f"Execution failed: {e}"

@tool
def read_any_file(file_path: str) -> str:
    """Read any text file. Binary files are reported by size, not dumped.

    Large files are truncated inline; call 'read_scratch' with the returned id
    to see the rest.
    """
    return _read_file(file_path)

@tool
def write_any_file(file_path: str, content: str, append: bool = False) -> str:
    """Write content to any file (overwrite, or append=True).

    Overwrites are idempotent (skipped when unchanged). To make a small change to
    an existing file, prefer 'patch_file' over rewriting the whole thing.
    """
    return _write_file(file_path, content, append=append)

@tool
def patch_file(file_path: str, old_text: str, new_text: str) -> str:
    """Surgically replace the first occurrence of old_text with new_text in a file.

    Prefer this over rewriting whole files. For .py files the result is
    syntax-checked and automatically rolled back on error. Idempotent: a no-op
    if the change is already applied.
    """
    return _patch_file(file_path, old_text, new_text)

@tool
def batch_patch(patches: list[dict]) -> str:
    """Apply multiple {file_path, old_text, new_text} patches in one call."""
    return _batch_patch(patches)

@tool
def git_diff(file_path: str = ".", cached: bool = False) -> str:
    """Show the git diff for a file or directory (cached=True for staged)."""
    return _git_diff(file_path, cached=cached)

@tool
def find_in_files(pattern: str, path: str = ".") -> str:
    """Search for a text pattern across source/text files (recursive).

    Result sets over 12 matches are paged via 'read_scratch'; a small set
    whose lines are huge is head+tail truncated with the full output saved to a
    file (see the notice in the result).
    """
    return _find_in_files(pattern, path)

@tool
def read_many_files(pattern: str, max_files: int = 20) -> str:
    """Read multiple files matching a glob pattern (e.g. 'src/**/*.py').

    Large result sets are truncated inline and paged via 'read_scratch'.
    """
    return _read_many_files(pattern, max_files=max_files)

@tool
def glob_list(pattern: str, max_results: int = 100) -> str:
    """List files matching a glob pattern with sizes (does not read contents)."""
    return _glob_list(pattern, max_results=max_results)

@tool
def task_start(command: str, cwd: str = "", config: RunnableConfig = None) -> str:
    """Start a long-running command as a managed background task; returns its id.

    Use for servers, watchers, or anything that should keep running while you
    continue working. Inspect with task_list/task_output; stop with task_kill.
    """
    _reason = _catastrophic_reason(command)
    if _reason:
        logger.warning("catastrophic_command_blocked: refused task_start %r (%s)", command, _reason)
        return f"Refused: command not executed ({_reason})."
    ctx = _resolve_ctx(config)
    if ctx is not None:
        return _tasks.task_start(command, cwd=cwd, manager=ctx.tasks)

    return _tasks.task_start(command, cwd=cwd)

@tool
def task_list(config: RunnableConfig = None) -> str:
    """List background tasks and their status."""
    ctx = _resolve_ctx(config)
    if ctx is not None:
        return _tasks.task_list(manager=ctx.tasks)
    return _tasks.task_list()

@tool
def task_status(task_id: str, config: RunnableConfig = None) -> str:
    """Show the status of one background task."""
    ctx = _resolve_ctx(config)
    if ctx is not None:
        return _tasks.task_status(task_id, manager=ctx.tasks)
    return _tasks.task_status(task_id)

@tool
def task_output(task_id: str, offset: int = 0, config: RunnableConfig = None) -> str:
    """Read a background task's captured output, paged by byte offset."""
    ctx = _resolve_ctx(config)
    if ctx is not None:
        return _tasks.task_output(task_id, offset=offset, manager=ctx.tasks)
    return _tasks.task_output(task_id, offset=offset)

@tool
def task_kill(task_id: str, config: RunnableConfig = None) -> str:
    """Terminate a running background task."""
    ctx = _resolve_ctx(config)
    if ctx is not None:
        return _tasks.task_kill(task_id, manager=ctx.tasks)
    return _tasks.task_kill(task_id)

@tool
def search_web(query: str, engine: str = "duckduckgo", max_results: int = 5) -> str:
    """Search the web via SearXNG engines."""
    return _search_web(query=query, engine=engine, max_results=max_results)

@tool
def fetch_url(url: str) -> str:
    """Fetch page text via Jina Reader."""
    return _fetch_url(url=url)

@tool
def read_scratch(scratch_id: str, offset: int = 0, length: int = 1500) -> str:
    """Read a portion of a saved scratch file."""
    return _read_scratch(scratch_id=scratch_id, offset=offset, length=length)

@tool
def vault(action: str, name: str = "", value: str = "") -> str:
    """Manage encrypted credentials stored in the local vault.

    Actions:
      - 'store':  encrypt and save a credential (needs name + value)
      - 'get':    retrieve a credential value (needs name)
      - 'list':   list stored credential names (no values)
      - 'remove': delete a credential (needs name)
      - 'status': show vault health
    Stored credentials are also exported as environment variables.
    """
    return _vault_run(action, name=name, value=value)


@tool
def reflex_distill(trigger: str, command: str, config: RunnableConfig = None) -> str:
    """Store a procedural reflex rule: when the user's words match ``trigger``,
    run ``command`` directly, without an LLM call.

    Use this after a successful shell command the user is likely to repeat
    (e.g. "check disk space" → "df -h").  The rule fires on keyword overlap
    (≥ 2 trigger words present in the user message), best-first, and is
    disabled via the /reflex slash command.  Returns the rule's id, or an
    error message if either field is empty."""
    ctx = _resolve_ctx(config)
    store = ctx.reflex_store if ctx and ctx.reflex_store else _reflex_store
    rid = store.distill(trigger, command)
    if not rid:
        return ("Reflex not stored: both 'trigger' and 'command' are required "
                "(trigger = the words that should fire it, e.g. 'check disk space').")
    return f"Reflex stored: id={rid} trigger={trigger!r} command={command!r}"

tools = [
    execute_shell_command, read_any_file, write_any_file,
    patch_file, batch_patch, git_diff,
    find_in_files, read_many_files, glob_list,
    task_start, task_list, task_status, task_output, task_kill,
    search_web, fetch_url, read_scratch,
    remember, recall, vault, reflex_distill,
    tool(_session_list), tool(_session_search), tool(_journal_search),
]

# Dynamically loaded plugin tools (tools/plugins/*.py).
# Each plugin exports TOOLS, DESCRIPTIONS, and TRIGGERS.
_plugin_tools, _plugin_descs, _plugin_triggers, _plugin_core = discover_plugins()
if _plugin_tools:
    tools.extend(_plugin_tools)
    _register_plugin_tools(_plugin_descs, _plugin_triggers, _plugin_core)

_TOOL_NAMES = {t.name for t in tools}

# Binding all twenty schemas on every step is the single largest fixed cost in
# the prompt, so the set is chosen per step (see components/tool_router.py).
# Bound models are cached by tool set: bind_tools is cheap but not free, and the
# same handful of sets recurs all session.
_bound_llms: "dict[tuple[str, ...], object]" = {}


_schema_tokens: "dict[tuple[str, ...], int]" = {}


def _bind_tools(selected):
    key = tuple(t.name for t in selected)
    if key not in _bound_llms:
        _bound_llms[key] = llm.bind_tools(selected)
    return _bound_llms[key]


def _tool_schema_tokens(selected) -> int:
    """Tokens the bound tool schemas add to the prompt.

    Counted from the JSON actually sent (`convert_to_openai_tool`), because this is
    the number `context.reserve_tokens` was sized against, back when every tool was
    bound on every step.
    """
    key = tuple(t.name for t in selected)
    if key not in _schema_tokens:
        try:
            payload = json.dumps([convert_to_openai_tool(t) for t in selected])
        except Exception:  # noqa: BLE001 — schema rendering is best-effort
            logger.debug("context: could not render tool schemas", exc_info=True)
            return 0
        _schema_tokens[key] = _ctx.estimate_tokens(payload)
    return _schema_tokens[key]

# Argument names weak models reach for instead of the real ones. Renaming them is
# strictly better than letting the call fail on an unexpected keyword; only
# unambiguous synonyms belong here (nothing that could change what a call means).
_ARG_ALIASES = {
    "recall": {"q": "query", "text": "query", "search": "query", "question": "query",
               "limit": "n", "top_k": "n", "k": "n"},
    "remember": {"text": "fact", "memory": "fact", "content": "fact", "facts": "fact",
                 "tag": "tags", "labels": "tags", "categories": "tags"},
    "reflex_distill": {"keywords": "trigger", "phrase": "trigger", "when": "trigger",
                        "rule": "trigger", "cmd": "command", "shell": "command",
                        "action": "command", "procedure": "command"},
}

# ------------------------------------------------------------------------------
# 4. System Prompt (autonomy + memory)
# ------------------------------------------------------------------------------
# Kept short on purpose: every word here is re-sent on every step, and a small
# model follows a handful of sharp rules plus one worked example better than a
# page of policy prose.
system_prompt = SystemMessage(content=(
    "You are an autonomous assistant with shell, file, web and long-term memory tools "
    "on this machine. You finish tasks yourself.\n"
    "- Never ask for permission and never describe what you would do: call the tool.\n"
    "- Use the tool-calling interface, not code blocks or curl commands that imitate it.\n"
    "- If a call fails or returns little, try other arguments or another tool in the "
    "same turn.\n"
    "- Think inside <thought>...</thought>, then act.\n"
    "- Prefer patch_file over rewriting a file; use task_start for anything long-running.\n"
    "- Call recall before answering anything that depends on what you already know "
    "(preferences, paths, past decisions); remember only durable facts.\n"
    "- After a successful shell command the user is likely to repeat, store a "
    "procedural rule with reflex_distill (trigger = the words that should fire it, "
    "e.g. 'check disk space'; command = the exact shell command).\n"
    "- Long results are saved to a scratch file: read the rest with "
    "read_scratch(scratch_id, offset).\n"
    "Example — 'is the api key set?' is answered by calling vault with "
    '{"action": "list"}, not by saying you will check.'
))


def _thinking_directive() -> str:
    """Return the thinking-mode suffix to append to the system prompt.

    ``/no_think`` suppresses Qwen3-family reasoning blocks; ``/think`` explicitly
    requests them.  ``"auto"`` (the default) leaves the model to decide.  The
    directive is appended as a trailing line so it does not alter the body of the
    prompt.
    """
    mode = THINKING_MODE
    if mode == "off":
        return "\n/no_think"
    if mode == "on":
        return "\n/think"
    return ""


_NO_THINK_SUFFIX = _thinking_directive()

# ------------------------------------------------------------------------------
# 5. Agent Node
# ------------------------------------------------------------------------------
class AgentState(MessagesState):
    """Messages plus the rolling summary of everything compaction dropped."""
    summary: str


# Previous step's prompt, rendered per message, so the next step can measure how
# much of it a prompt-caching server could still reuse (see _ctx.record_step).
_last_prompt: "list[str]" = []


def _record_prompt(messages, selected) -> None:
    """Account for one step's prompt: overhead vs history, and cache reuse."""
    global _last_prompt
    rendered = [f"{getattr(m, 'type', '?')}:{getattr(m, 'content', '')}" for m in messages]
    schemas = _tool_schema_tokens(selected)
    lead = _ctx.message_tokens(messages[0])
    history = _ctx.total_tokens(messages[1:])
    _ctx.record_step(
        history_tokens=history,
        overhead_tokens=lead + schemas,
        schema_tokens=schemas,
        prefix_tokens=_ctx.shared_prefix_tokens(_last_prompt, rendered),
    )
    _last_prompt = rendered


def agent(state: AgentState):
    summary = state.get("summary") or ""
    # One system message, always first: served chat templates commonly reject a
    # second one (llama.cpp: "System message must be at the beginning"), so the
    # rolling summary is folded into the prompt rather than sent beside it.
    if summary:
        lead = SystemMessage(
            content=f"{system_prompt.content}{_NO_THINK_SUFFIX}\n\nEarlier in this session:\n{summary}"
        )
    else:
        lead = SystemMessage(content=system_prompt.content + _NO_THINK_SUFFIX)
    messages = [lead]
    messages += state["messages"]
    selected = _select_tools(tools, state["messages"])
    _record_prompt(messages, selected)
    response = _bind_tools(selected).invoke(messages)
    # Small local models often print the call they meant to make instead of
    # using the tool-calling channel; recover those so they actually execute.
    repair_message(response, _TOOL_NAMES, _ARG_ALIASES)
    # Track tokens the model spent in <think> blocks that get stripped for display.
    _ctx.record_thinking_tokens(response.content if hasattr(response, "content") else "")
    return {"messages": [response]}

# ------------------------------------------------------------------------------
# 6. Automatic Knowledge Distillation Node
# ------------------------------------------------------------------------------
_MEMORY_TOOL_NAMES = {"remember", "recall"}

# Tools whose output is machine state, not knowledge: a task list or a vault
# status describes this minute, so distilling it spends a whole extra LLM call
# (doubling turn latency on a local model) to produce nothing worth keeping.
NON_DISTILLABLE_TOOLS = set(app_config.get("memory.non_distillable_tools", [
    "task_list", "task_status", "task_output", "task_kill", "task_start",
    "vault", "read_scratch", "glob_list",
]))


def distill_knowledge(state: AgentState) -> AgentState:
    """
    Hand the turn's user request, tool results, and answer to the background
    memory worker, which extracts durable facts and stores them off the graph's
    critical path.

    Guard: only distil when the turn actually executed at least one tool call
    that returned a result. If the model only *described* what it would do (no
    tool messages in this turn), there is nothing factual to extract — storing
    the assistant's intentions as facts would poison the memory with
    hallucinations. The memory tools themselves don't count as evidence: a turn
    whose only tool call was ``remember``/``recall`` has no new grounding, and
    treating it as such is how greetings ended up in long-term memory. Tools in
    ``NON_DISTILLABLE_TOOLS`` are excluded for the same reason.
    """
    # Nudges are HumanMessages too (see components/routing.py); distilling one
    # would file the guardrail's own text as the user's request.
    user_msgs = [m for m in state["messages"]
                 if isinstance(m, HumanMessage) and not is_nudge(m)]
    ai_msgs = [m for m in state["messages"] if m.type == "ai" and m.content]

    if not user_msgs or not ai_msgs:
        return state

    # Find the index of the last HumanMessage so we only inspect the current turn.
    last_human_idx = max(
        i for i, m in enumerate(state["messages"])
        if isinstance(m, HumanMessage) and not is_nudge(m)
    )
    turn_msgs = state["messages"][last_human_idx:]

    # --- T8. Recall Compliance Check ---
    last_ai_msg = ai_msgs[-1]
    last_ai_content = str(last_ai_msg.content or "")
    memory_related_phrases = [
        "you said", "you mentioned", "preference", "recalled", 
        "remember", "as requested", "my knowledge", "long-term memory"
    ]
    if any(phrase in last_ai_content.lower() for phrase in memory_related_phrases):
        # Check if 'recall' was called in this turn
        called_tools = []
        for m in turn_msgs:
            for call in (getattr(m, "tool_calls", None) or []):
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    called_tools.append(name)
        if "recall" not in called_tools:
            logger.info(
                "compliance check: final answer contains memory-referencing phrase "
                "but 'recall' was not invoked this turn (prose: %r)", 
                last_ai_content[:150]
            )
    tool_results = [
        m for m in turn_msgs
        if getattr(m, "type", None) == "tool"
        and getattr(m, "name", None) not in _MEMORY_TOOL_NAMES
        and getattr(m, "name", None) not in NON_DISTILLABLE_TOOLS
    ]
    if not tool_results:
        logger.debug("Knowledge distillation skipped: no distillable tool results "
                     "in this turn.")
        return state

    # Build context from actual tool outputs so the distillation model has
    # grounded evidence rather than the assistant's prose descriptions.
    tool_context = "\n".join(
        f"[{getattr(m, 'name', 'tool')}]: "
        f"{(m.content if isinstance(m.content, str) else str(m.content))[:400]}"
        for m in tool_results
    )

    _memory_worker.enqueue(DistillJob(
        user_text=user_msgs[-1].content,
        ai_text=ai_msgs[-1].content,
        tool_context=tool_context,
        enqueued_at=time.time(),
    ))
    return state

# ------------------------------------------------------------------------------
# 7. Build Graph with Distillation & Autonomous Guardrail
# ------------------------------------------------------------------------------
def _summarize(prompt: str) -> str:
    """One cheap completion, with no tools bound, used to compact history.

    Uses the tiered summarizer (Groq models first, local model last) so
    summarization quality does not depend on the local model alone.
    """
    response = _summarize_llm.invoke([HumanMessage(content=prompt)])
    return (response.content or "") if hasattr(response, 'content') else str(response)


# ------------------------------------------------------------------------------
# 7a. Procedural Reflex Node (Track D of docs/neo-port-plan.md)
# ------------------------------------------------------------------------------
_REFLEX_ANSWER_MARKER = "[REFLEX]"


def reflex_node(state: AgentState):
    """Deterministic fast-path: if the user's latest message matches a stored
    reflex rule, run its command via the same ``execute_shell_command`` path
    (catastrophic denylist and blast-radius warning included) and return the
    result as the turn's answer — no LLM call, no tools round-trip.

    The command's output is wrapped in a marker so the REPL renders it as
    a normal answer panel (and downstream routing treats it as a final
    answer, not a tool result).  A match is journaled for audit; ``use()``
    increments the rule's counter.  No match → return {} so the graph falls
    through to ``compact``/``agent`` as usual."""
    messages = state["messages"]
    if not messages:
        return {}
    last = messages[-1]
    if not isinstance(last, HumanMessage) or is_nudge(last):
        return {}
    query = str(getattr(last, "content", "") or "")
    hits = _reflex_store.match(query)
    if not hits:
        return {}
    rule = hits[0]
    _reflex_store.use(rule["id"])
    _journal_event("reflex", rule_id=rule["id"], trigger=rule["trigger"],
                   command=rule["command"], source="system")
    logger.info("reflex: firing rule %s (%r) for %r", rule["id"], rule["trigger"], query[:120])
    output = execute_shell_command.invoke({"command": rule["command"]})
    answer = (f"{_REFLEX_ANSWER_MARKER} {rule['trigger']} → "
                f"`{rule['command']}`\n\n{output}")
    return {"messages": [AIMessage(content=answer)]}


def route_reflex(state: AgentState) -> str:
    """After the reflex node::a match ends the turn (no LLM call);;no match
    falls through to ``compact`` as usual."""
    last = state["messages"][-1]
    if getattr(last, "type", None) == "ai" and _REFLEX_ANSWER_MARKER in str(getattr(last, "content", "")):
        return "distill"
    return "compact"


def compact_context(state: AgentState):
    """Fold the oldest messages into the rolling summary when over budget.

    Runs before every agent step, so a single huge tool result is caught on the
    step that follows it rather than on the next user turn. Messages are dropped
    from the checkpointed state with ``RemoveMessage``, which is what keeps the
    thread from growing forever on disk as well as in the prompt.

    On recompaction (the summary already carries prior compactions), key facts
    are extracted from the old summary and fed into the summarization prompt so
    they survive into the new summary.
    """
    messages = state["messages"]
    summary = state.get("summary") or ""
    if not _ctx.needs_compaction(messages, summary):
        return {}
    preserve_facts = ""
    if summary:
        preserve_facts = _extract_summary_facts(summary)
    dropped, _kept, new_summary = _ctx.compact(
        messages, _summarize, summary,
        keep_last=_ctx.KEEP_LAST_MESSAGES,
        keep_last_tokens=_ctx.KEEP_LAST_TOKENS,
        preserve_facts=preserve_facts)
    removable = [m for m in dropped if getattr(m, "id", None)]
    if not removable:
        return {}
    _ctx.record_compaction(len(removable), _ctx.total_tokens(removable))
    _journal_event("condensation", kind="summary", summary=new_summary,
                    dropped=len(removable), source="system")
    return {
        "messages": [RemoveMessage(id=m.id) for m in removable],
        "summary": new_summary,
    }


# ------------------------------------------------------------------------------
# Track F — token-aware confirmation gate (config ``tools.confirm_mutating``).
# ------------------------------------------------------------------------------
# Default OFF: langbot is an autonomous agent ("Never ask for permission") —
# everything that is not catastrophic runs immediately, exactly as before Track F.
# When ON, gray-zone mutating calls (write/exec tools that aren't trusted
# read-only) are refused with a notice; the user can approve by replying "yes"
# (or ok/approved/go ahead/…), and the model re-issues the exact same call —
# the gate then lets it through once.  Reflex rules are exempt: they are
# user-created procedures (distilled from a successful command), so they count
# as pre-approved (they still pass through ``execute_shell_command``'s own
# catastrophic hard-block).
_CONFIRM_MUTATING = app_config.get("tools.confirm_mutating", False)
_APPROVAL_RE = re.compile(
    r"^\s*(?:yes|y|yeah|ok|okay|sure|go ahead|go for it|approved|"
    r"confirm(?:ed)?|run it|do it|proceed|affirmative)[.!]*\s*$",
    re.I,
)
_safety_gate = _SafetyGate(confirm_mutating=_CONFIRM_MUTATING)

# Pending user-approval state (only meaningful when ``_CONFIRM_MUTATING``).
_pending_confirm: "str | None" = None    # exact signature awaiting approval
_pending_confirm_tool: "str | None" = None  # tool name awaiting approval
_pending_approved: bool = False


def _clear_pending_confirm(ctx: "_KernelContext | None" = None) -> None:
    """Clear the pending gray-zone approval state.

    Operates on ``ctx.pending_*`` when a per-run context is present (concurrent
    sessions must not approve each other's pending calls); otherwise on the
    module-level defaults (the REPL's single-session path, unchanged)."""
    if ctx is not None:
        ctx.pending_confirm = None
        ctx.pending_confirm_tool = None
        ctx.pending_approved = False
        return
    global _pending_confirm, _pending_confirm_tool, _pending_approved
    _pending_confirm = None
    _pending_confirm_tool = None
    _pending_approved = False


def _note_user_input(text: str, ctx: "_KernelContext | None" = None) -> None:
    """Called once per user message (before the turn streams).  A whole-message
    approval ("yes", "ok", "go ahead", …) approves the pending gray-zone call;
    any other message clears the pending state (the moment passed; a new task
    context starts).  No-op when the gate is off."""
    if ctx is not None:
        pending_confirm = ctx.pending_confirm
        pending_confirm_tool = ctx.pending_confirm_tool
    else:
        pending_confirm = _pending_confirm
        pending_confirm_tool = _pending_confirm_tool
    if not _CONFIRM_MUTATING:
        return
    if pending_confirm is None:
        return
    if _APPROVAL_RE.match(text.strip()):
        if ctx is not None:
            ctx.pending_approved = True
        else:
            global _pending_approved
            _pending_approved = True
        _journal_event("confirm_approved", tool=pending_confirm_tool,
                        signature=pending_confirm[:300], source="user")
        logger.info("confirm gate: user approved pending %s %r",
                    pending_confirm_tool, pending_confirm[:120])
    else:
        _journal_event("confirm_cancelled", tool=pending_confirm_tool,
                        signature=pending_confirm[:300], source="user")
        _clear_pending_confirm(ctx)


def _confirm_signature(tool_name: str, call: dict) -> str:
    """Canonical signature for a mutating call, used to match a re-issued
    call against the user-approved one.  Shell tools key on the exact command
    string; non-shell mutating tools key on tool name + sorted args."""
    if tool_name in ("execute_shell_command", "task_start"):
        return str(call.get("args", {}).get("command", "") or "")
    args_json = json.dumps(call.get("args", {}), sort_keys=True)
    return f"{tool_name}({args_json[:200]})"


def _check_confirm_gate(tool_name: str, call: dict, ctx: "_KernelContext | None" = None) -> "tuple[bool, str]":
    """(allowed, reason) for one tool call under the confirmation gate.

    Read tools always pass.  With the gate off (default), mutating tools pass
    too (pre-Track-F behaviour preserved).  With it on: catastrophic → hard
    block; trusted read-only shell → fast pass; gray zone → refuse and remember
    the pending signature — unless the user already approved exactly this signature,
    in which case it runs once and the pending state clears.
    """
    global _pending_confirm, _pending_confirm_tool, _pending_approved
    if ctx is not None:
        pending_confirm = ctx.pending_confirm
        pending_confirm_tool = ctx.pending_confirm_tool
        pending_approved = ctx.pending_approved
    else:
        pending_confirm = _pending_confirm
        pending_confirm_tool = _pending_confirm_tool
        pending_approved = _pending_approved
    if _tool_safety(tool_name) == _SAFETY_READ:
        return True, ""
    if not _CONFIRM_MUTATING:
        return True, ""
    if tool_name in ("execute_shell_command", "task_start"):
        cmd = _confirm_signature(tool_name, call)
        # The gate object is built once at import; sync its confirm flag from the
        # module-level config flag (the single source of truth) so a config
        # change (or a test monkeypatch) takes effect without a rebuild.
        _safety_gate.confirm_mutating = _CONFIRM_MUTATING
        verdict = _safety_gate.check_shell(cmd)
        if not verdict.allowed:
            return False, verdict.reason
        if not verdict.needs_confirm:
            return True, ""
    # Gray zone (mutating, not trusted read-only — or a non-shell mutating tool).
    sig = _confirm_signature(tool_name, call)
    if pending_approved and pending_confirm == sig and pending_confirm_tool == tool_name:
        _clear_pending_confirm(ctx)
        return True, ""
    if ctx is not None:
        ctx.pending_confirm = sig
        ctx.pending_confirm_tool = tool_name
        ctx.pending_approved = False
    else:
        _pending_confirm = sig
        _pending_confirm_tool = tool_name
        _pending_approved = False
    return False, f"requires confirmation: {sig[:200]!r} — ask the user to approve it; once approved, re-issue the exact same call."


def _gate_tool_calls(to_run: list, ctx: "_KernelContext | None" = None) -> "tuple[list, list]":
    """Split ``to_run`` into (runnable, refused-ToolMessages) under the gate.

    Refused calls are answered with a ToolMessage (same invariant as
    ``split_repeated_calls``: every call in the assistant message is answered,
    so the next request stays valid)."""
    runnable = []
    refused = []
    for call in to_run:
        name = call.get("name", "?")
        allowed, reason = _check_confirm_gate(name, call, ctx)
        if allowed:
            runnable.append(call)
        else:
            logger.warning("confirm gate: refused %s (%s)", name, reason)
            _journal_event("confirm_gate", tool=name, reason=reason,
                            signature=_confirm_signature(name, call)[:300], source="system")
            refused.append(ToolMessage(
                content=f"⛔ {reason}",
                name=name,
                tool_call_id=call.get("id") or "",
                status="error",
            ))
    return runnable, refused


_tool_node = ToolNode(tools)
_tool_node = ToolNode(tools)

# Per-thread background-task managers (Track G): two concurrent sessions must
# not clobber each other's task lists.  Keyed by thread_id;a thread's tasks
# survive across its own turns (the manager is created once per thread, not per
# graph invocation), while different threads get isolated managers.  The REPL's
# slash handlers (``/tasks``, ``/kill``) keep using the module-level singleton.

_thread_tasks: "dict[str, object]" = {}


def _ctx_for_run(config: RunnableConfig) -> "_KernelContext":
    """Build the per-run ``KernelContext`` for a graph invocation.



    The context carries:the thread_id, a per-thread task manager, the shared
    reflex store/journal/memory worker (process-wide singletons by design),and
    fresh per-run pending-confirm state (so two sessions can't approve each
    other's pending calls).  It is stashed into ``config["configurable"]`` so
    every tool in the run resolves the same instance via ``_resolve_ctx``."""
    thread_id = (config.get("configurable") or {}).get("thread_id") or "default"
    mgr = _thread_tasks.get(thread_id)
    if mgr is None:
        mgr = _tasks.BackgroundTaskManager()
        _thread_tasks[thread_id] = mgr
    ctx = _KernelContext(
        thread_id=thread_id,
        tasks=mgr,
        reflex_store=_reflex_store,
        journal=_current_journal,
        memory_worker=_memory_worker,
    )
    _stash_context(config, ctx)
    return ctx


def tools_node(state: AgentState, config: RunnableConfig):
    """Run tools, then scrub any stored credential values from their output
    before it re-enters the model's context (see vault.redact).

    A call the model already made verbatim this turn is answered from the transcript
    instead of executed (``routing.split_repeated_calls``) — where a stuck loop stops
    costing shell commands and fetches. Blocked calls still get a ToolMessage, so
    every call in the assistant message is answered and the next request stays valid.
    """
    ctx = _ctx_for_run(config)
    messages = state["messages"]
    to_run, blocked = split_repeated_calls(messages)
    # Track F confirmation gate: gray-zone mutating calls are refused here
    # (when ``tools.confirm_mutating`` is on) before they reach the tool node.



    to_run, gated = _gate_tool_calls(to_run, ctx)
    blocked = blocked + gated
    if not blocked:
        result = _tool_node.invoke(state)
    elif not to_run:
        result = {"messages": []}
    else:
        # The tool node sees a copy of the message carrying only the calls to run; the
        # original stays in state, so call ids still line up with the replies.
        trimmed = messages[-1].model_copy(update={"tool_calls": to_run})
        result = _tool_node.invoke({**state, "messages": list(messages[:-1]) + [trimmed]})

    for msg in result.get("messages", []):
        # Skip the vault tool itself — 'get' is meant to return the value.
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) != "vault" \
                and isinstance(getattr(msg, "content", None), str):
            msg.content = _vault_redact(msg.content)
            # Kernel-level head+tail fallback (Track A): any tool result that did
            # not self-offload to scratch (e.g. a plugin tool, glob_list on a
            # huge dir) is caught here instead of being dumped whole into the
            # message state.  Vault output is exempted above — a credential value
            # must never be written to a plaintext file under ./memory/truncated/.
            msg.content = _maybe_truncate(
                msg.content, prefix=getattr(msg, "name", "tool") or "tool"
            )
    if blocked:
        result = {**result, "messages": list(result.get("messages", [])) + blocked}
    return result
# Stuck-detection constants (Track C of docs/neo-port-plan.md).  The detector
# folds the whole current turn's messages and fires on five patterns (see
# components/stuck.py).  A ``halt`` verdict ends the turn with a stuck
# directive ((mirroring neo's ``(stuck: ...)`` finish);;a ``nudge`` verdict
# injects a softer "change approach" watchdog message,, capped at 3 per turn
# like the existing nudge budget,, then escalates to halt..
STUCK_HALT_MARKER = "[STUCK]"
MAX_STUCK_NUDGES = app_config.get("routing.max_stuck_nudges", 3)
_STUCK_DETECTOR = _StuckDetector()


def _stuck_directive(verdict, ignored: int) -> str:
    """Build the corrective HumanMessage for a stuck verdict."""
    if verdict.severity == "halt":
        text = (f"{verdict.detail} — stop now and answer from what you "
                 f"already have")
    else:
        text = (f"{verdict.detail} — change approach,, explain the blocker,, "
                 f"or finish")
    if ignored >= 1:
        text += (" — you have been told this before; change approach,, "
                 "explain the blocker,, or finish")
    return f"{NUDGE_MARKER} {STUCK_HALT_MARKER} {text}"


def _current_turn_messages(messages) -> list:
    """Messages since the last real user message (nudges excluded) — the
    window stuck detection must see.

    The detector's patterns (monologue, repeating, soft_repeating,
    alternating) count events across whatever list they are given; if they
    saw the whole session, prior turns' chat answers (content-only AI
    messages that legitimately ended those turns) would poison the monologue
    counter and halt a perfectly good tool round in the current turn.  Slicing
    to the current turn mirrors neo's design, where the detector folds the
    journaled event log per turn (see docs/neo-port-plan.md Track C).
    """
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, HumanMessage) and not is_nudge(m):
            return messages[i:]
    return messages


def stuck_node(state: AgentState):
    """Run stuck detection after each tool round ((between ``tools`` and
    ``compact``).  On a verdict,, inject a corrective HumanMessage ((the
    ``stuck`` node's output is rendered like ``nudge``'s,, and the directive
    counts as a nudge for the turn's budget via ``NUDGE_MARKER``).  A ``halt``
    verdict ((or 3 ignored nudges)) escalates to a terminal stuck message
    routed straight to ``distill``/END — no 4th call..
    """
    messages = state["messages"]
    verdict = _STUCK_DETECTOR.check(_current_turn_messages(messages))
    if verdict is None:
        return {}
    ignored = sum(1 for m in messages
                  if is_nudge(m) and STUCK_HALT_MARKER in str(getattr(m, "content", "")))
    if verdict.severity == "halt" or ignored >= MAX_STUCK_NUDGES:
        _journal_event("stuck", pattern=verdict.pattern, detail=verdict.detail,
                       severity="halt", ignored=ignored, source="system")
        return {"messages": [HumanMessage(content=_stuck_directive(verdict, ignored))],
                "stuck_halt": True}
    _journal_event("watchdog", pattern=verdict.pattern, detail=verdict.detail,
                   severity="nudge", ignored=ignored, source="system")
    return {"messages": [HumanMessage(content=_stuck_directive(verdict, ignored))]}


def route_stuck(state: AgentState) -> str:
    """After the stuck node::a terminal halt directive goes to ``distill``
    ((end the turn);;a soft nudge ((or no verdict)) goes back through
    ``compact`` to ``agent``."""
    last = state["messages"][-1]
    if is_nudge(last) and STUCK_HALT_MARKER in str(getattr(last, "content", "")):
        return "distill"
    return "compact"

builder = StateGraph(AgentState)
builder.add_node("reflex", reflex_node)
builder.add_node("compact", compact_context)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)
builder.add_node("nudge", nudge_agent)
builder.add_node("distill", distill_knowledge)

builder.add_edge(START, "reflex")
builder.add_conditional_edges("reflex", route_reflex, ["distill", "compact"])
builder.add_edge("compact", "agent")
builder.add_conditional_edges("agent", route_agent, ["tools", "nudge", "distill"])
builder.add_node("stuck", stuck_node)
builder.add_edge("tools", "stuck")
builder.add_conditional_edges("stuck", route_stuck, ["distill", "compact"])
builder.add_edge("compact", "agent")
builder.add_edge("nudge", "agent")
builder.add_edge("distill", END)

# ------------------------------------------------------------------------------
# 8. Execution Loop
# ------------------------------------------------------------------------------
def _render_message(msg) -> None:
    """Surface a single streamed graph message as a live Rich panel.

    - AI message with content + tool calls  -> intermediate "Thought" panel
    - AI message with content, no tool calls -> final "Answer" panel (Markdown)
    - AI tool calls                          -> "Tool Call" panel(s)
    - Tool message                           -> "Tool Result" panel
    """
    mtype = getattr(msg, "type", None)

    if mtype == "ai":
        tool_calls = getattr(msg, "tool_calls", None) or []
        content = msg.content
        if content:
            text = _vault_redact(content if isinstance(content, str) else str(content))
            # Qwen-family models put reasoning in <think> blocks; same treatment.
            text = text.replace("<think>", "<thought>").replace("</think>", "</thought>")
            if "<thought>" in text and "</thought>" in text:
                parts = text.split("</thought>")
                thought_part = parts[0].replace("<thought>", "").strip()
                ans_part = parts[1].strip() if len(parts) > 1 else ""
                if thought_part:
                    ui.thought_panel(thought_part)
                if ans_part:
                    if tool_calls:
                        ui.thought_panel(ans_part)
                    else:
                        ui.final_answer_panel(ans_part)
            else:
                if tool_calls:
                    ui.thought_panel(text)
                else:
                    ui.final_answer_panel(text)
        for call in tool_calls:
            ui.tool_call_panel(call.get("name", "tool"), call.get("args") or {})

    elif mtype == "tool":
        # tools_node has already redacted stored secrets from non-vault output.
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        is_error = getattr(msg, "status", None) == "error"
        ui.tool_result_panel(getattr(msg, "name", None) or "tool", content, is_error=is_error)


def _stream_turn(app, config, user_input: str) -> None:
    """Stream one agent turn, rendering every node update as it arrives.

    ``stream_mode="updates"`` yields only the *new* messages produced by each
    node, so output appears the instant it is generated (no buffering until the
    end). A spinner covers the wait before the first panel is emitted. The
    ``distill`` node re-emits the whole state, so we only render output from the
    ``agent`` and ``tools`` nodes.

    One human turn yields at most one final-answer panel: a second no-tool-call
    AI message is a duplicate (see routing.route_agent's guard) and is dropped
    instead of rendered. At DEBUG level every chunk's (node, message types) is
    logged so a duplicate can be traced back to its source — a second ``agent``
    invocation with no ``tools``/``nudge`` message in between points at the
    graph, whereas two AI messages in one update points at the LLM server. The
    drop itself is logged at DEBUG because routing.route_agent already warns
    about it; this guard only catches what slips past that one.
    """
    spinner = ui.GradientSpinner("Thinking...")
    spinner.start()
    spinner_running = True
    answered = False
    last_final = ""
    # Point the session tools at this turn's journal (cleared on exit so a
    # stale journal can't leak across turns).
    _set_current_journal(_current_journal)
    try:
        for chunk in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "stream chunk: node=%s messages=%s", node,
                        [getattr(m, "type", None) for m in (update or {}).get("messages", [])],
                    )
                if node not in ("agent", "tools", "nudge", "stuck", "reflex") or not update:
                    continue
                for msg in update.get("messages", []):
                    mtype = getattr(msg, "type", None)
                    # Journal every event the graph produces (best-effort; audit
                    # layer only — never the source of truth).
                    if mtype == "ai":
                        content = getattr(msg, "content", "") or ""
                        calls = getattr(msg, "tool_calls", None) or []
                        if content or calls:
                            _journal_event("llm_response", content=str(content)[:4000],
                                           tool_calls=calls, source="agent")
                        for call in calls:
                            _journal_event("tool_call",
                                           name=call.get("name", "?"),
                                           args=call.get("args") or {},
                                           source="agent")
                    elif mtype == "tool":
                        _journal_event("tool_result",
                                      tool=getattr(msg, "name", "tool"),
                                      preview=str(getattr(msg, "content", ""))[:2000],
                                      call_id=getattr(msg, "tool_call_id", ""),
                                      source="tool")
                    elif mtype == "human" and is_nudge(msg):
                        _journal_event("nudge",
                                      note=str(getattr(msg, "content", ""))[:500],
                                      source="system")
                    is_final_answer = (
                        mtype == "ai"
                        and msg.content
                        and not (getattr(msg, "tool_calls", None) or [])
                    )
                    if is_final_answer and answered:
                        logger.debug(
                            "dropping a duplicate final answer for this turn (preview: %r)",
                            str(msg.content)[:120],
                        )
                        continue
                    if spinner_running:
                        spinner.stop()
                        spinner_running = False
                    _render_message(msg)
                    if is_final_answer:
                        last_final = str(msg.content)
                        answered = True
        if answered:
            _journal_event("finish", message=last_final[:2000], source="agent")
    finally:
        _set_current_journal(None)
        if spinner_running:
            spinner.stop()


def _split_trailing_tags(text: str) -> "tuple[str, list[str]]":
    """Split trailing ``#tag`` tokens off a /save argument."""
    words = text.split()
    tags: "list[str]" = []
    while words and words[-1].startswith("#") and len(words[-1]) > 1:
        tags.insert(0, words.pop()[1:])
    return " ".join(words), tags


_SLASH_HELP = [
    ("Session", "/quit, /exit", "End the session"),
    ("Session", "/new, /clear", "Start a fresh conversation (new memory thread)"),
    ("Session", "/history", "Show conversation history summary"),
    ("Session", "/sessions", "List past sessions (journals), newest by activity)"),
    ("Session", "/session <id> <query>", "Search a past session's events by keyword"),
    ("Memory", "/knowledge <q>", "Search long-term memory (a '#tag' query filters by tag)"),
    ("Memory", "/save <fact> [#tag ...]", "Store a fact in long-term memory, optionally tagged"),
    ("Memory", "/tags", "List distinct memory tags with counts"),
    ("Memory", "/forget <id>", "Delete a memory from the store by id"),
    ("Reflex", "/reflex", "List procedural reflex rules (uses, disabled state)"),
    ("Reflex", "/reflex disable <id>", "Disable a reflex rule so it no longer fires"),
    ("Reflex", "/reflex enable <id>", "Re-enable a disabled reflex rule"),
    ("Vault", "/vault list", "List credentials stored in the vault"),
    ("Vault", "/vault status", "Show vault health dashboard"),
    ("Tasks", "/tasks", "List background tasks and their status"),
    ("Tasks", "/kill <id>", "Terminate a running background task"),
    ("Files", "/ls [dir]", "List files in a directory"),
    ("System", "/help", "Show this help"),
    ("System", "/info", "Show model, tool count, thread, memory size"),
    ("System", "/health", "Show checkpointer, memory, vault, and task status"),
    ("System", "/config", "Show the active config file (or that defaults are in use)"),
    ("System", "/compact", "Force compaction of older message history into summary"),
    ("System", "/log [n]", "Show last n lines of the log file"),
]


def _tail_log(n: int = 30) -> str:
    path = _log_path()
    if not path or not os.path.exists(path):
        return "No log file found."
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            from collections import deque
            lines = deque(f, maxlen=n)
            return "".join(lines).strip()
    except Exception as e:  # noqa: BLE001 — log read errors are user-facing
        return f"Error reading log file: {e}"


def _handle_slash(text: str, config: dict, app: object) -> bool:
    """Handle a /command. Returns True if the session should end.

    These are local REPL commands (advertised by input.py's tab-completer);
    they never reach the LLM.
    """
    parts = text[1:].strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit"):
        return True
    if cmd == "history":
        try:
            state = app.get_state(config)
            messages = state.values.get("messages", []) if state else []
        except Exception as e:  # noqa: BLE001 — state retrieval is best-effort
            ui.warning(f"Failed to retrieve state: {e}")
            return False
        if not messages:
            ui.info("No messages in the current conversation history.")
            return False
        
        ui.header("Conversation History")
        for i, m in enumerate(messages, 1):
            role = getattr(m, "type", m.__class__.__name__.replace("Message", "").lower())
            content = getattr(m, "content", "")
            if not isinstance(content, str):
                content = str(content)
            
            # Clean up content for a compact single line
            content_clean = content.replace("\n", " ").strip()
            content_truncated = content_clean[:80] + "..." if len(content_clean) > 80 else content_clean
            
            info_part = f"#{i:02d} {role.upper()}"
            if role == "tool":
                tool_name = getattr(m, "name", "unknown")
                info_part += f" ({tool_name})"
            elif role == "ai" and getattr(m, "tool_calls", None):
                tool_names = ", ".join(tc.get("name", "") for tc in m.tool_calls)
                info_part += f" [calls: {tool_names}]"
                
            ui.kv(info_part, content_truncated or "(empty)")
        return False
    if cmd == "help":
        groups = {}
        for grp, name, desc in _SLASH_HELP:
            groups.setdefault(grp, []).append((name, desc))
        order = ["Session", "Memory", "Vault", "Tasks", "System", "Files"]
        for grp in order:
            if grp in groups:
                ui.header(grp)
                for name, desc in groups[grp]:
                    ui.kv(name, desc)
        return False
    if cmd == "vault":
        sub = arg.lower() if arg else ""
        if sub == "list":
            ui.info(_vault_list())
            return False
        elif sub == "status":
            ui.info(_vault_status())
            return False
        else:
            ui.warning("Usage: /vault list | /vault status")
            return False
    if cmd == "tasks":
        ui.info(_tasks.task_list())
        return False
    if cmd == "kill":
        if not arg:
            ui.warning("Usage: /kill <task_id>")
            return False
        ui.info(_tasks.task_kill(arg))
        return False
    if cmd == "log":
        n = 30
        if arg:
            try:
                n = int(arg)
            except ValueError:
                ui.warning("Usage: /log [number_of_lines]")
                return False
        ui.info(_tail_log(n))
        return False
    if cmd in ("new", "clear"):
        new_id = f"session_{uuid.uuid4().hex[:8]}"
        config["configurable"]["thread_id"] = new_id
        _start_journal(new_id)
        ui.success(f"Started a fresh conversation (thread {new_id}).")
        return False
    if cmd == "sessions":
        ui.info(_format_sessions())
        return False
    if cmd == "session":
        parts2 = arg.split(maxsplit=1)
        if len(parts2) < 2:
            ui.warning("Usage: /session <id> <query>")
            return False
        ui.info(_search_session(parts2[0], parts2[1]))
        return False
    if cmd == "config":
        ui.kv("config file", app_config.describe())
        if not app_config.loaded:
            ui.info(f"Create ./{CONFIG_FILENAME} (or set ${CONFIG_ENV_VAR}) to override "
                    f"defaults; see {CONFIG_FILENAME}.example.")
        ui.kv("llm", f"{LLM_MODEL} @ {BASE_URL} (temp {LLM_TEMPERATURE})")
        ui.kv("checkpoint db", SQLITE_DB_PATH)
        ui.kv("memory store", _memory_store.CHROMA_PERSIST_DIR)
        ui.kv("memory search", f"min similarity {_memory_store.MIN_SIMILARITY}, "
                               f"overfetch x{_memory_store.RECALL_OVERFETCH}, "
                               f"mmr lambda {_memory_store.MMR_LAMBDA}, "
                               f"lexical {'on' if _memory_store.LEXICAL_SEARCH else 'off'}")
        ui.kv("scratch dir", _scratch.SCRATCH_DIR)
        ui.kv("sessions dir", str(_journals_dir()))
        ui.kv("tasks dir", _tasks.TASKS_DIR)
        ui.kv("inline caps", f"file {_file_ops.READ_INLINE_CHARS}, "
                             f"grep {_code_search.GREP_INLINE_LINES} lines, "
                             f"fetch {_web_tools.FETCH_INLINE_CHARS}")
        ui.kv("memory worker", f"queue {_memory_worker_mod.MAX_QUEUE_SIZE}, "
                               f"batch {_memory_worker_mod.MAX_BATCH}")
        ui.kv("distillation", _distill_llm.describe())
        ui.kv("summarizer", _summarize_llm.describe())
        ui.kv("log file", str(_log_path() or "(stderr)"))
        return False
    if cmd == "info":
        ui.kv("model", LLM_MODEL)
        ui.kv("config file", app_config.describe())
        ui.kv("tools", str(len(tools)))
        ui.kv("thread_id", config["configurable"]["thread_id"])
        ui.kv("memories", str(_memory_count()))
        ui.kv("memory queue depth", str(_memory_worker.qsize()))
        ui.kv("checkpointer", "sqlite" if SQLITE_AVAILABLE else "memory")
        return False
    if cmd == "health":
        ui.kv("checkpointer", "sqlite" if SQLITE_AVAILABLE else "memory")
        ui.kv("memories", str(_memory_count()))
        ui.kv("memory queue depth", str(_memory_worker.qsize()))
        ui.kv("memory jobs dropped", str(_memory_worker.dropped_count()))
        # Non-zero recovered calls means the server is not constraining tool-call
        # decoding (start llama-server with --jinja); see README.
        _repairs = _repair_stats()
        ui.kv("tool-call repairs", f"{_repairs['recovered_calls']} recovered, "
                                   f"{_repairs['cleaned_answers']} answers cleaned")
        import components.routing as _routing
        _rst = _routing.stats()
        ui.kv("routing nudges", f"{_rst['nudges_permission']} permission, "
                               f"{_rst['nudges_code_block']} code block")
        ui.kv("routing drift telemetry", f"{_rst['near_miss_permission_hedges']} near-misses")
        ui.kv("distillation", _distill_llm.describe())
        ui.kv("summarizer", _summarize_llm.describe())
        ui.kv("warmup", _warmup.summary())
        ui.kv("disk freed at start", _sweep_summary)
        ui.kv("context", _ctx.stats_summary())
        ui.kv("vault creds", str(len(_VAULT_ENV_LOADED)))
        ui.kv("bg tasks", str(len(_tasks.manager.list())))
        ui.kv("log file", str(_log_path() or "(stderr)"))
        return False
    if cmd == "ls":
        ui.info(_glob_list(os.path.join(arg or ".", "*")))
        return False
    if cmd == "knowledge":
        if not arg:
            ui.warning("Usage: /knowledge <query>")
            return False
        # Scores and provenance are shown here because this view is how you tune
        # memory.min_similarity: a query whose good hits sit below the floor (or
        # whose junk hits sit above it) tells you which way to move it.
        mems = _search_memories(arg, n=5)
        if not mems:
            ui.info("No memory above the relevance threshold "
                    f"({_memory_store.MIN_SIMILARITY}).")
            return False
        ui.info("\n".join(
            f"- [{m.id[:8]} | {m.score:.2f} {'+'.join(m.matched) or 'dense'}] {m.text}"
            + ("  " + " ".join(f"#{t}" for t in m.tags) if m.tags else "")
            + f"  ({m.source or 'unknown'}, {m.timestamp or 'no timestamp'})"
            for m in mems
        ))
        return False
    if cmd == "tags":
        tags = _memory_store.list_tags()
        if not tags:
            ui.info("No tags stored in memory.")
            return False
        ui.header("Memory Tags")
        for tag, count in tags:
            ui.kv(f"#{tag}", f"{count} fact(s)")
        return False
    if cmd == "forget":
        if not arg:
            ui.warning("Usage: /forget <memory_id>")
            return False
        deleted = _memory_store.delete_memory(arg)
        if deleted:
            ui.success(f"Deleted memory with id '{arg}'")
        else:
            ui.warning(f"Memory with id '{arg}' not found.")
        return False
    if cmd == "reflex":
        sub, rest = (arg.split(maxsplit=1) + [""])[:2] if arg else ("", "")
        sub = sub.lower()
        if sub == "disable":
            if not rest:
                ui.warning("Usage: /reflex disable <rule_id>")
                return False
            if _reflex_store.disable(rest.strip()):
                ui.success(f"Disabled reflex rule '{rest.strip()}'")
            else:
                ui.warning(f"Reflex rule '{rest.strip()}' not found.")
            return False
        if sub == "enable":
            if not rest:
                ui.warning("Usage: /reflex enable <rule_id>")
                return False
            if _reflex_store.enable(rest.strip()):
                ui.success(f"Re-enabled reflex rule '{rest.strip()}'")
            else:
                ui.warning(f"Reflex rule '{rest.strip()}' not found.")
            return False
        rules = _reflex_store.all()
        if not rules:
            ui.info("No reflex rules stored.  The model can store one via the "
                    "reflex_distill tool after a successful shell command.")
            return False
        ui.header(f"Reflex Rules ({len(rules)})")
        for r in rules:
            state = "disabled" if r.get("disabled") else f"{r.get('uses', 0)} use(s)"
            ui.kv(f"{r['id']} — {r['trigger']}", f"`{r['command']}` [{state}]")
        return False
    if cmd == "compact":
        state = app.get_state(config)
        messages = state.values.get("messages", []) if state else []
        summary = state.values.get("summary", "") if state else ""
        if not messages:
            ui.info("No conversation history to compact.")
            return False
            
        preserve_facts = ""
        if summary:
            preserve_facts = _extract_summary_facts(summary)
            
        dropped, recent, new_summary = _ctx.compact(
            messages, _summarize, summary,
            keep_last=_ctx.KEEP_LAST_MESSAGES,
            keep_last_tokens=_ctx.KEEP_LAST_TOKENS,
            preserve_facts=preserve_facts
        )
        if not dropped:
            ui.info("No older messages available to compact (below keep_last limit).")
            return False
            
        removable = [m for m in dropped if getattr(m, "id", None)]
        if removable:
            try:
                app.update_state(config, {
                    "messages": [RemoveMessage(id=m.id) for m in removable],
                    "summary": new_summary
                })
                _ctx.record_compaction(len(removable), _ctx.total_tokens(removable))
                _journal_event("condensation", kind="summary", summary=new_summary,
                              dropped=len(removable), source="system")
                ui.success(f"Successfully compacted {len(removable)} message(s) into rolling summary.")
            except Exception as e:  # noqa: BLE001 — checkpoint update is best-effort
                ui.warning(f"Failed to update checkpoint state: {e}")
        else:
            ui.warning("Older messages exist but lack stable checkpoint IDs to be removed.")
        return False
    if cmd == "save":
        if not arg:
            ui.warning("Usage: /save <fact to remember> [#tag ...]")
            return False
        fact, tags = _split_trailing_tags(arg)
        if not fact:
            ui.warning("A fact cannot be only tags.")
            return False
        before = _memory_count()
        _store_memory(fact, tags=tags)
        if _memory_count() == before:
            ui.info("Already in long-term memory (duplicate).")
        else:
            ui.success("Saved to long-term memory"
                       + (" " + " ".join(f"#{t}" for t in tags) if tags else "") + ".")
        return False

    ui.warning(f"Unknown command: /{cmd}  (try /help)")
    return False


def run_repl(app, config):
    """Interactive read-eval-print loop.

    A failure while handling one turn (LLM error, tool crash, bad checkpoint
    state, etc.) must not tear down the whole session — it is caught, surfaced
    to the user, and the loop continues to the next prompt. Ctrl+C during a
    running turn interrupts *that turn* and returns to the prompt; Ctrl+C at an
    empty prompt (or Ctrl+D) ends the session.
    """
    while True:
        try:
            ctx_bar = ui.context_usage_bar()
            if ctx_bar:
                ui._write(f"  {ctx_bar}")
            user_input = read_input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            ui.info("Session closing...")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ('quit', 'exit'):
            break
        if user_input.startswith('/'):
            if _handle_slash(user_input, config, app):
                break
            continue

        global _turn
        _turn += 1
        _journal_event("user_message", text=user_input, source="user")
        _journal_set_title(user_input)
        # Track F: a whole-message approval ("yes", "ok", "go ahead", …)
        # approves the pending gray-zone call; any other message clears it.
        _note_user_input(user_input)

        try:
            _stream_turn(app, config, user_input)
        except KeyboardInterrupt:
            # Abort just this turn, not the whole session.
            ui.warning("Interrupted — returning to the prompt.")
            continue
        except EOFError:
            ui.info("Session closing...")
            break
        except Exception as e:  # noqa: BLE001 — session loop must not crash
            # Don't kill the session over a single failed turn.
            logger.exception("Error while processing turn")
            err_msg = str(e)
            if "503" in err_msg and "Loading model" in err_msg:
                ui.error("Local LLM model is still loading on server (503). Give the server a few seconds to load weights into VRAM, then try again.")
            elif "500" in err_msg and ("parse error" in err_msg or "Failed to parse" in err_msg):
                ui.error("The local LLM server encountered a context parse error (500).")
                ui.info("Try typing /new to start a fresh, clean conversation thread.")
            else:
                ui.error(f"{e}")
            ui.info("The session is still active — try again or type 'quit' to exit.")


def main() -> None:
    """Console entrypoint: set up the REPL, compile the graph, and run it."""
    setup_readline()
    ui.banner("langbot", "unrestricted shell / file / web agent")
    ui.warning("This agent has UNRESTRICTED shell, file, and web access.")
    if _VAULT_ENV_LOADED:
        ui.info(f"Vault: loaded {len(_VAULT_ENV_LOADED)} credential(s) into the environment.")
    if not SQLITE_AVAILABLE:
        ui.warning("langgraph-checkpoint-sqlite is not installed — conversation history "
                   "will not persist.")
    ui.startup_tip(LLM_MODEL)
    session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    # Set before the sweep runs, so this session's checkpoint rows are never the
    # ones it prunes.
    global _active_thread_id
    _active_thread_id = session_id
    _warmup.start()
    _start_journal(session_id)
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        if SQLITE_AVAILABLE:
            with SqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
                app = builder.compile(checkpointer=checkpointer)
                run_repl(app, config)
        else:
            checkpointer = MemorySaver()
            app = builder.compile(checkpointer=checkpointer)
            run_repl(app, config)
    finally:
        _memory_worker.shutdown(timeout=10.0)
        _vault_save()


if __name__ == "__main__":
    main()
