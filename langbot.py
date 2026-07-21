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
llm = ChatOpenAI(model=LLM_MODEL, base_url=BASE_URL, api_key="not-needed", temperature=0.1)

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
    "as well as a persistent, searchable long-term memory. You act proactively: if a task "
    "can be done safely, do it without asking. Only request explicit user confirmation for "
    "potentially destructive operations.\n\n"
    "When using tools: if a call does not yield the desired result, immediately try alternative "
    "parameters, engines, or approaches. Do not stop to ask the user for guidance.\n\n"
    "You have long-term memory tools ('remember' and 'recall'), but the system also automatically "
    "distills important facts from our conversation. Use 'recall' to retrieve relevant past "
    "information before tackling new tasks.\n\n"
    "For editing files, prefer 'patch_file' (surgical find/replace) over rewriting whole files, "
    "and use 'find_in_files'/'read_many_files'/'glob_list' to navigate code. Run long-lived "
    "processes (servers, watchers) with 'task_start' and manage them via task_list/task_output/"
    "task_kill instead of blocking shell calls.\n\n"
    "Be concise but thorough – give complete answers, not play‑by‑play commentary."
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
    """
    user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in state["messages"] if m.type == "ai" and m.content]

    if not user_msgs or not ai_msgs:
        return state

    last_user = user_msgs[-1].content
    last_ai = ai_msgs[-1].content

    distillation_prompt = f"""
You are a knowledge extraction module. Look at the following user request and assistant response.
Extract any factual information that would be useful for a long-term memory system.
These could be:
- User preferences (e.g., "the user likes Python", "they prefer DuckDuckGo")
- Important facts learned (e.g., "project located at ~/code/myapp", "the weather API key is...")
- Decisions or conclusions (e.g., "decided to use SQLite for storage")
- Context that will help future interactions (e.g., "the user is working on a web scraping project")

User request: {last_user}
Assistant response: {last_ai}

Return ONLY a JSON array of strings, each a standalone factual statement. If nothing important, return an empty array [].
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
# 7. Build Graph with Distillation
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

builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", tools_node)
builder.add_node("distill", distill_knowledge)

builder.add_edge(START, "agent")

# After agent: if there are tool calls -> tools, else -> distill
def route_agent(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.type == "ai" and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "distill"

builder.add_conditional_edges("agent", route_agent, ["tools", "distill"])
builder.add_edge("tools", "agent")
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
    config = {"configurable": {"thread_id": "root_access_session_1"}}

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
