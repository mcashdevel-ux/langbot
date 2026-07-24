import os

# Silence HuggingFace / transformers / tokenizers progress bars and chatter
# *before* those libraries are imported so the embedding model loads quietly
# (no "Loading weights: 100%|█| 103/103 ..." lines on the console).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import subprocess
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    SQLITE_AVAILABLE = True
except ModuleNotFoundError:
    from langgraph.checkpoint.memory import MemorySaver
    SQLITE_AVAILABLE = False
    print("Warning: langgraph-checkpoint-sqlite not installed – conversation history will not persist.")

import chromadb
from chromadb.config import Settings

from components.web_tools import search_web as _search_web, fetch_url as _fetch_url, read_scratch as _read_scratch
from components.utils import truncate, suppress_native_output, strip_code_fences
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

import components.console as ui
from components.input import read_input, setup_readline
from components.vault import (
    bootstrap as _vault_bootstrap,
    run_action as _vault_run,
    redact as _vault_redact,
    save as _vault_save,
)

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "local-model"
SQLITE_DB_PATH = "./memory/agent_checkpoints.db"
CHROMA_PERSIST_DIR = "./memory/agent_memory_chroma"

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
    temperature=0.1,
    max_retries=10,
)

def _load_embeddings():
    """Construct the embedding model without leaking its loading progress bars
    onto the console. The heavy transformers/tqdm output is written to the raw
    stderr fd, so we mute it at the fd level while the weights load."""
    ui.info("Loading embedding model...")
    try:
        import transformers  # noqa: WPS433 (optional, only to quiet it)

        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except Exception:
        pass
    with suppress_native_output():
        model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
        )
    ui.success("Embedding model ready.")
    return model


embeddings = _load_embeddings()

# ------------------------------------------------------------------------------
# 2. Semantic Memory Store
# ------------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)
memory_collection = chroma_client.get_or_create_collection(
    name="agent_longterm_memory",
    metadata={"hnsw:space": "cosine"},
)

def _store_memory(text: str) -> str:
    mem_id = str(uuid.uuid4())
    vector = embeddings.embed_query(text)
    memory_collection.add(
        ids=[mem_id],
        embeddings=[vector],
        metadatas=[{"text": text, "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}],
    )
    return mem_id

def _recall_memories(query: str, n: int = 3) -> list[str]:
    count = memory_collection.count()
    if count == 0:
        # Chroma rejects n_results < 1, so guard the empty-store case explicitly.
        return []
    query_vec = embeddings.embed_query(query)
    results = memory_collection.query(
        query_embeddings=[query_vec],
        n_results=min(n, count),
    )
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return []
    return [meta.get("text", "") for meta in results["metadatas"][0] if meta.get("text")]

# ------------------------------------------------------------------------------
# 3. Tools (original + memory)
# ------------------------------------------------------------------------------
@tool
def remember(fact: str) -> str:
    """Manually store a fact in long-term memory."""
    try:
        mem_id = _store_memory(fact)
        return f"Memory stored (id {mem_id}): {fact[:200]}..."
    except Exception as e:
        return f"Failed to store memory: {e}"

@tool
def recall(query: str, n: int = 3) -> str:
    """Search long-term memory for similar facts."""
    try:
        memories = _recall_memories(query, n)
        if not memories:
            return "No relevant memories found."
        return "Relevant memories:\n" + "\n".join(f"- {m}" for m in memories)
    except Exception as e:
        return f"Failed to recall memories: {e}"

@tool
def execute_shell_command(command: str, cwd: str = "", timeout: int = 120) -> str:
    """Execute a shell command synchronously and return its output.

    Optionally run in ``cwd`` with a custom ``timeout`` (seconds; 0 = no limit).
    For servers, watchers, or anything long-running, use 'task_start' instead so
    the process is tracked and can be inspected or killed.
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
        return truncate(output) or f"Command '{command}' executed successfully."
    except subprocess.TimeoutExpired:
        return f"Timeout ({timeout}s): '{command}'"
    except Exception as e:
        return f"Execution failed: {e}"

@tool
def read_any_file(file_path: str) -> str:
    """Read any text file. Binary files are reported by size, not dumped."""
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
    """Search for a text pattern across source/text files (recursive)."""
    return _find_in_files(pattern, path)

@tool
def read_many_files(pattern: str, max_files: int = 20) -> str:
    """Read multiple files matching a glob pattern (e.g. 'src/**/*.py')."""
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
llm_with_tools = llm.bind_tools(tools)

# ------------------------------------------------------------------------------
# 4. System Prompt (autonomy + memory)
# ------------------------------------------------------------------------------
system_prompt = SystemMessage(content=(
    "You are a fully autonomous AI assistant with unrestricted shell, file, and web access, "
    "as well as a persistent, searchable long-term memory. You act proactively and execute multi-step tasks to completion.\n\n"
    "CRITICAL AUTONOMY RULES:\n"
    "1. REASONING & CHAIN-OF-THOUGHT (CoT): Always enclose your step-by-step thinking inside <thought>...</thought> tags before you act. Formulate a plan and then execute it using tools immediately.\n"
    "2. NEVER ASK FOR PERMISSION: Do NOT output phrases like 'Would you like me to proceed?', 'Should I fetch...', or 'Shall I run...'. If the user asks for a task (e.g. 'validate all secrets'), DO NOT describe how you would do it in text — ACTUALLY INVOKE THE TOOLS to execute it right now.\n"
    "3. USE NATIVE TOOL CALLING ONLY: You have native tools available (e.g., `search_web`, `fetch_url`, `execute_shell_command`). DO NOT write python scripts to import these tools. DO NOT write dummy `curl` commands in text blocks. You must invoke the provided tools directly via your function calling interface.\n"
    "4. AUTOMATIC MULTI-STEP RETRY: If a tool call fails or returns empty/partial results, try alternative parameters, tools, or shell commands immediately in the same turn. Do not stop and hand control back to the user until the full objective is achieved.\n\n"
    "For editing files, prefer 'patch_file' over rewriting whole files. For background tasks, use 'task_start'. Use 'recall' to check long-term memory when relevant."
))

# ------------------------------------------------------------------------------
# 5. Agent Node
# ------------------------------------------------------------------------------
def agent(state: MessagesState):
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ------------------------------------------------------------------------------
# 6. Automatic Knowledge Distillation Node
# ------------------------------------------------------------------------------
def distill_knowledge(state: MessagesState) -> MessagesState:
    """
    Extract important facts from the most recent user request and assistant response,
    and save them to long-term memory automatically.

    Guard: only distil when the turn actually executed at least one tool call that
    returned a result.  If the model only *described* what it would do (no tool
    messages in this turn), there is nothing factual to extract — storing the
    assistant's intentions as facts would poison the memory with hallucinations.
    """
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in state["messages"] if m.type == "ai" and m.content]

    if not user_msgs or not ai_msgs:
        return state

    # ── Fix 1: only distil from turns where tools were actually invoked ──────
    # Find the index of the last HumanMessage so we only inspect the current turn.
    last_human_idx = max(
        i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)
    )
    turn_msgs = state["messages"][last_human_idx:]
    tool_results = [m for m in turn_msgs if getattr(m, "type", None) == "tool"]
    if not tool_results:
        logger.debug("Knowledge distillation skipped: no tool results in this turn.")
        return state
    # ─────────────────────────────────────────────────────────────────────────

    last_user = user_msgs[-1].content
    last_ai = ai_msgs[-1].content

    # Build context from actual tool outputs so the distillation model has
    # grounded evidence rather than the assistant's prose descriptions.
    tool_context_lines = []
    for m in tool_results:
        name = getattr(m, "name", "tool")
        content = m.content if isinstance(m.content, str) else str(m.content)
        tool_context_lines.append(f"[{name}]: {content[:400]}")
    tool_context = "\n".join(tool_context_lines)

    distillation_prompt = f"""
You are a knowledge extraction module. Look at the following user request, the tool
results that were actually returned this turn, and the assistant's final response.
Extract only factual information that is GROUNDED IN THE TOOL RESULTS — do not infer
or store anything the assistant merely described doing without evidence in the tool output.
Useful facts include:
- User preferences (e.g., "the user prefers DuckDuckGo")
- Confirmed facts from tool output (e.g., "project located at ~/code/myapp")
- Decisions or conclusions that are supported by evidence
- Context helpful for future interactions

User request: {last_user}
Tool results this turn:
{tool_context}
Assistant response: {last_ai}

Return ONLY a JSON array of strings, each a standalone factual statement grounded in
the tool results above. If nothing is clearly supported by evidence, return [].
Do not include explanations, markdown, or extra text.
"""
    try:
        raw = llm.invoke(distillation_prompt).content.strip()
        raw = strip_code_fences(raw)
        facts = json.loads(raw)
        if not isinstance(facts, list):
            logger.warning("Knowledge distillation skipped: model output was not a JSON array")
            return state
        for fact in facts:
            if isinstance(fact, str) and fact.strip():
                _store_memory(fact)
            # Optional console feedback – quiet by default
    except json.JSONDecodeError as e:
        # Model returned something that wasn't a JSON array; distillation is a
        # bonus, so we don't break the main loop — but we don't hide it either.
        logger.warning("Knowledge distillation skipped: could not parse model output as JSON: %s", e)
    except Exception as e:
        logger.warning("Knowledge distillation failed: %s", e, exc_info=True)

    return state

# ------------------------------------------------------------------------------
# 7. Build Graph with Distillation & Autonomous Guardrail
# ------------------------------------------------------------------------------
_tool_node = ToolNode(tools)

def tools_node(state: MessagesState):
    """Run tools, then scrub any stored credential values from their output
    before it re-enters the model's context (see vault.redact)."""
    result = _tool_node.invoke(state)
    for msg in result.get("messages", []):
        # Skip the vault tool itself — 'get' is meant to return the value.
        if getattr(msg, "type", None) == "tool" and getattr(msg, "name", None) != "vault" \
                and isinstance(getattr(msg, "content", None), str):
            msg.content = _vault_redact(msg.content)
    return result

# ── Fix 2: failure-mode detection phrases ────────────────────────────────────
# Phrases that indicate the model is asking for permission instead of acting.
PERMISSION_PHRASES = (
    "would you like me to",
    "would you like to proceed",
    "should i proceed",
    "shall i proceed",
    "do you want me to",
    "can i proceed",
    "let me know if you would like",
    "if you would like me to",
    "would you like to",
    "would you like me",
    "please confirm",
    "please let me know",
    "do you want to proceed",
)

# Patterns that indicate the model is hallucinating tool calls as code blocks
# instead of invoking the actual function-calling interface.
TOOL_AVOIDANCE_PATTERNS = (
    "import search_web",
    "import fetch_url",
    "import execute_shell",
    "search_web.search(",
    "fetch_url(",
    "requests.get(",          # writing raw HTTP calls instead of using tools
    "requests.post(",
    "subprocess.run(",         # using shell inside a code block instead of the tool
    "```python\nimport",       # code fence opening with an import
    "```bash\ncurl",           # writing curl in a bash block instead of the tool
    "```\ncurl",
    "curl -h ",
    "curl -o ",
    "{{vault_get",            # hallucinated template syntax
    "{{vault",
)
# ─────────────────────────────────────────────────────────────────────────────

_NUDGE_PERMISSION = (
    "[AUTONOMOUS AGENT DIRECTIVE]: You just asked for permission or confirmation instead of acting. "
    "Do NOT ask the user whether to proceed. "
    "Invoke the required tool calls RIGHT NOW to complete the user request."
)

_NUDGE_CODE_BLOCK = (
    "[AUTONOMOUS AGENT DIRECTIVE]: You wrote code or curl commands in a text block instead of "
    "calling your native tools. You have tools available — search_web, fetch_url, "
    "execute_shell_command, vault, etc. DO NOT write Python or bash blocks that pretend to use "
    "these tools. Call them DIRECTLY via the function-calling interface RIGHT NOW."
)


def _ai_turns_since_human(messages) -> int:
    """Count consecutive AI messages back to (not including) the last HumanMessage."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "type", None) == "ai":
            count += 1
    return count


def nudge_agent(state: MessagesState):
    """Nudge node: inject a targeted correction when the model either asked for
    permission or hallucinated tool calls as code blocks instead of invoking them."""
    last_msg = state["messages"][-1]
    content = getattr(last_msg, "content", "") or ""
    content_lower = content.lower()

    # Pick the nudge message most appropriate to the detected failure mode.
    if any(pat in content_lower for pat in TOOL_AVOIDANCE_PATTERNS):
        nudge_text = _NUDGE_CODE_BLOCK
    else:
        nudge_text = _NUDGE_PERMISSION

    return {"messages": [SystemMessage(content=nudge_text)]}


def route_agent(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.type == "ai" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    if last_msg.type == "ai" and isinstance(getattr(last_msg, "content", None), str):
        content = last_msg.content
        content_lower = content.lower()

        needs_nudge = (
            any(phrase in content_lower for phrase in PERMISSION_PHRASES)
            or any(pat in content_lower for pat in TOOL_AVOIDANCE_PATTERNS)
        )

        if needs_nudge:
            # Allow up to 5 re-tries per human turn before giving up.
            if _ai_turns_since_human(state["messages"]) < 5:
                return "nudge"

    return "distill"

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)
builder.add_node("nudge", nudge_agent)
builder.add_node("distill", distill_knowledge)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_agent, ["tools", "nudge", "distill"])
builder.add_edge("tools", "agent")
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
    """
    spinner = ui.GradientSpinner("Thinking...")
    spinner.start()
    spinner_running = True
    try:
        for chunk in app.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                if node not in ("agent", "tools") or not update:
                    continue
                for msg in update.get("messages", []):
                    if spinner_running:
                        spinner.stop()
                        spinner_running = False
                    _render_message(msg)
    finally:
        if spinner_running:
            spinner.stop()


_SLASH_HELP = [
    ("/help", "Show this help"),
    ("/quit, /exit", "End the session"),
    ("/new, /clear", "Start a fresh conversation (new memory thread)"),
    ("/info", "Show model, tool count, thread, memory size"),
    ("/health", "Show checkpointer, memory, vault, and task status"),
    ("/ls [dir]", "List files in a directory"),
    ("/knowledge <q>", "Search long-term memory"),
    ("/save <fact>", "Store a fact in long-term memory"),
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
    if cmd == "info":
        ui.kv("model", LLM_MODEL)
        ui.kv("tools", str(len(tools)))
        ui.kv("thread_id", config["configurable"]["thread_id"])
        ui.kv("memories", str(memory_collection.count()))
        ui.kv("checkpointer", "sqlite" if SQLITE_AVAILABLE else "memory")
        return False
    if cmd == "health":
        ui.kv("checkpointer", "sqlite" if SQLITE_AVAILABLE else "memory")
        ui.kv("memories", str(memory_collection.count()))
        ui.kv("vault creds", str(len(_VAULT_ENV_LOADED)))
        ui.kv("bg tasks", str(len(_tasks.manager.list())))
        return False
    if cmd == "ls":
        ui.info(_glob_list(os.path.join(arg or ".", "*")))
        return False
    if cmd == "knowledge":
        if not arg:
            ui.warning("Usage: /knowledge <query>")
            return False
        mems = _recall_memories(arg, n=5)
        ui.info("\n".join(f"- {m}" for m in mems) if mems else "No relevant memories.")
        return False
    if cmd == "save":
        if not arg:
            ui.warning("Usage: /save <fact to remember>")
            return False
        _store_memory(arg)
        ui.success("Saved to long-term memory.")
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
            user_input = read_input("\nYou: ")
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
    ui.startup_tip(LLM_MODEL)
    session_id = f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    config = {"configurable": {"thread_id": session_id}}

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
        _vault_save()


if __name__ == "__main__":
    main()
