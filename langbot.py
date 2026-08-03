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
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Send every component's log records to ./memory/langbot.log before anything can
# emit one: unconfigured logging prints to stderr, i.e. into the middle of the
# REPL's panels and prompt (see components/logging_setup.py).
from components.logging_setup import log_path as _log_path, setup as _setup_logging

_setup_logging()

from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
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
from components.tool_router import select_tools as _select_tools, register as _register_plugin_tools
from tools.plugins import discover_plugins
from components.routing import (
    RECURSION_LIMIT,
    is_nudge,
    nudge_agent,
    route_agent,
    split_repeated_calls,
)
from components.tool_call_repair import repair_message, stats as _repair_stats

import components.console as ui
from components.input import read_input, setup_readline
from components.vault import (
    bootstrap as _vault_bootstrap,
    run_action as _vault_run,
    redact as _vault_redact,
    save as _vault_save,
)

# ------------------------------------------------------------------------------
# Configuration — every value below has a working default, so ./langbot.config.json
# (see components/config.py for the search order) is entirely optional.
# ------------------------------------------------------------------------------
BASE_URL = app_config.get("llm.base_url", "http://127.0.0.1:8080/v1")
LLM_MODEL = app_config.get("llm.model", "local-model")
LLM_TEMPERATURE = app_config.get("llm.temperature", 0.1)
LLM_MAX_RETRIES = app_config.get("llm.max_retries", 10)
THINKING_MODE = app_config.get("llm.thinking_mode", "auto")
SQLITE_DB_PATH = app_config.get("paths.checkpoint_db", "./memory/agent_checkpoints.db")

# ------------------------------------------------------------------------------
# 0. Credential Vault — load stored secrets into the environment before the LLM
#    and tools are constructed, so *_API_KEY values are available to them.
# ------------------------------------------------------------------------------
_VAULT_ENV_LOADED = _vault_bootstrap()

# ------------------------------------------------------------------------------
# 1. LLM & Embeddings
# ------------------------------------------------------------------------------
llm = ChatOpenAI(
    model=LLM_MODEL,
    base_url=BASE_URL,
    api_key="not-needed",
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
# 3. Tools (original + memory)
# ------------------------------------------------------------------------------
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
    except Exception as e:
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
    except Exception as e:
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
        if not output:
            return f"Command '{command}' executed successfully."
        return _offload(output, prefix="shell", inline_chars=MAX_OUTPUT_CHARS,
                        label="full output")
    except subprocess.TimeoutExpired:
        return f"Timeout ({timeout}s): '{command}'"
    except Exception as e:
        return f"Execution failed: {e}"

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

    Result sets over 20 matches are paged via 'read_scratch'.
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
def task_start(command: str, cwd: str = "") -> str:
    """Start a long-running command as a managed background task; returns its id.

    Use for servers, watchers, or anything that should keep running while you
    continue working. Inspect with task_list/task_output; stop with task_kill.
    """
    return _tasks.task_start(command, cwd=cwd)

@tool
def task_list() -> str:
    """List background tasks and their status."""
    return _tasks.task_list()

@tool
def task_status(task_id: str) -> str:
    """Show the status of one background task."""
    return _tasks.task_status(task_id)

@tool
def task_output(task_id: str, offset: int = 0) -> str:
    """Read a background task's captured output, paged by byte offset."""
    return _tasks.task_output(task_id, offset=offset)

@tool
def task_kill(task_id: str) -> str:
    """Terminate a running background task."""
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

tools = [
    execute_shell_command, read_any_file, write_any_file,
    patch_file, batch_patch, git_diff,
    find_in_files, read_many_files, glob_list,
    task_start, task_list, task_status, task_output, task_kill,
    search_web, fetch_url, read_scratch,
    remember, recall, vault,
]

# Dynamically loaded plugin tools (tools/plugins/*.py).
# Each plugin exports TOOLS, DESCRIPTIONS, and TRIGGERS.
_plugin_tools, _plugin_descs, _plugin_triggers = discover_plugins()
if _plugin_tools:
    tools.extend(_plugin_tools)
    _register_plugin_tools(_plugin_descs, _plugin_triggers)

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
        except Exception:
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
    """One cheap completion, with no tools bound, used to compact history."""
    return llm.invoke([HumanMessage(content=prompt)]).content or ""


def compact_context(state: AgentState):
    """Fold the oldest messages into the rolling summary when over budget.

    Runs before every agent step, so a single huge tool result is caught on the
    step that follows it rather than on the next user turn. Messages are dropped
    from the checkpointed state with ``RemoveMessage``, which is what keeps the
    thread from growing forever on disk as well as in the prompt.
    """
    messages = state["messages"]
    summary = state.get("summary") or ""
    if not _ctx.needs_compaction(messages, summary):
        return {}
    dropped, _kept, new_summary = _ctx.compact(
        messages, _summarize, summary, keep_last=_ctx.KEEP_LAST_MESSAGES)
    removable = [m for m in dropped if getattr(m, "id", None)]
    if not removable:
        return {}
    _ctx.record_compaction(len(removable), _ctx.total_tokens(removable))
    return {
        "messages": [RemoveMessage(id=m.id) for m in removable],
        "summary": new_summary,
    }


_tool_node = ToolNode(tools)

def tools_node(state: AgentState):
    """Run tools, then scrub any stored credential values from their output
    before it re-enters the model's context (see vault.redact).

    A call the model already made verbatim this turn is answered from the transcript
    instead of executed (``routing.split_repeated_calls``) — where a stuck loop stops
    costing shell commands and fetches. Blocked calls still get a ToolMessage, so
    every call in the assistant message is answered and the next request stays valid.
    """
    messages = state["messages"]
    to_run, blocked = split_repeated_calls(messages)
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
    if blocked:
        result = {**result, "messages": list(result.get("messages", [])) + blocked}
    return result

builder = StateGraph(AgentState)
builder.add_node("compact", compact_context)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)
builder.add_node("nudge", nudge_agent)
builder.add_node("distill", distill_knowledge)

builder.add_edge(START, "compact")
builder.add_edge("compact", "agent")
builder.add_conditional_edges("agent", route_agent, ["tools", "nudge", "distill"])
builder.add_edge("tools", "compact")
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
                if node not in ("agent", "tools") or not update:
                    continue
                for msg in update.get("messages", []):
                    is_final_answer = (
                        getattr(msg, "type", None) == "ai"
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
                    answered = answered or bool(is_final_answer)
    finally:
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
    ("/help", "Show this help"),
    ("/quit, /exit", "End the session"),
    ("/new, /clear", "Start a fresh conversation (new memory thread)"),
    ("/info", "Show model, tool count, thread, memory size"),
    ("/health", "Show checkpointer, memory, vault, and task status"),
    ("/config", "Show the active config file (or that defaults are in use)"),
    ("/ls [dir]", "List files in a directory"),
    ("/knowledge <q>", "Search long-term memory (a '#tag' query filters by tag)"),
    ("/save <fact> [#tag ...]", "Store a fact in long-term memory, optionally tagged"),
]


def _handle_slash(text: str, config: dict) -> bool:
    """Handle a /command. Returns True if the session should end.

    These are local REPL commands (advertised by input.py's tab-completer);
    they never reach the LLM.
    """
    parts = text[1:].strip().split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("quit", "exit"):
        return True
    if cmd == "help":
        ui.header("Commands")
        for name, desc in _SLASH_HELP:
            ui.kv(name, desc)
        return False
    if cmd in ("new", "clear"):
        new_id = f"session_{uuid.uuid4().hex[:8]}"
        config["configurable"]["thread_id"] = new_id
        ui.success(f"Started a fresh conversation (thread {new_id}).")
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
        ui.kv("tasks dir", _tasks.TASKS_DIR)
        ui.kv("inline caps", f"file {_file_ops.READ_INLINE_CHARS}, "
                             f"grep {_code_search.GREP_INLINE_LINES} lines, "
                             f"fetch {_web_tools.FETCH_INLINE_CHARS}")
        ui.kv("memory worker", f"queue {_memory_worker_mod.MAX_QUEUE_SIZE}, "
                               f"batch {_memory_worker_mod.MAX_BATCH}")
        ui.kv("distillation", _distill_llm.describe())
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
        ui.kv("distillation", _distill_llm.describe())
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
            f"- [{m.score:.2f} {'+'.join(m.matched) or 'dense'}] {m.text}"
            + ("  " + " ".join(f"#{t}" for t in m.tags) if m.tags else "")
            + f"  ({m.source or 'unknown'}, {m.timestamp or 'no timestamp'})"
            for m in mems
        ))
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
            prompt_line = f"\nYou: {ctx_bar}  " if ctx_bar else "\nYou: "
            user_input = read_input(prompt_line)
        except (KeyboardInterrupt, EOFError):
            ui.info("Session closing...")
            break

        if not user_input.strip():
            continue
        if user_input.strip().lower() in ('quit', 'exit'):
            break
        if user_input.startswith('/'):
            if _handle_slash(user_input, config):
                break
            continue

        try:
            _stream_turn(app, config, user_input)
        except KeyboardInterrupt:
            # Abort just this turn, not the whole session.
            ui.warning("Interrupted — returning to the prompt.")
            continue
        except EOFError:
            ui.info("Session closing...")
            break
        except Exception as e:
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
