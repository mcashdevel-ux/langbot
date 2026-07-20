import os
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

from web_tools import search_web as _search_web, fetch_url as _fetch_url, read_scratch as _read_scratch
from utils import truncate

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BASE_URL = "http://127.0.0.1:8080/v1"
LLM_MODEL = "local-model"
SQLITE_DB_PATH = "./agent_checkpoints.db"
CHROMA_PERSIST_DIR = "./agent_memory_chroma"

# ------------------------------------------------------------------------------
# 1. LLM & Embeddings
# ------------------------------------------------------------------------------
llm = ChatOpenAI(model=LLM_MODEL, base_url=BASE_URL, api_key="not-needed", temperature=0.1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

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
    query_vec = embeddings.embed_query(query)
    results = memory_collection.query(
        query_embeddings=[query_vec],
        n_results=min(n, memory_collection.count()),
    )
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return []
    return [meta["text"] for meta in results["metadatas"][0]]

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
def execute_shell_command(command: str) -> str:
    """Execute a bash command."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]:\n{result.stderr}"
        return truncate(output) or f"Command '{command}' executed successfully."
    except subprocess.TimeoutExpired:
        return f"Timeout: '{command}'"
    except Exception as e:
        return f"Execution failed: {e}"

@tool
def read_any_file(file_path: str) -> str:
    """Read any file."""
    if not os.path.exists(file_path):
        return f"Error: Path '{file_path}' does not exist."
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return truncate(f.read())
    except Exception as e:
        return f"Failed to read {file_path}: {e}"

@tool
def write_any_file(file_path: str, content: str) -> str:
    """Write content to any file."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to '{file_path}'."
    except Exception as e:
        return f"Failed to write: {e}"

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

tools = [
    execute_shell_command, read_any_file, write_any_file,
    search_web, fetch_url, read_scratch,
    remember, recall,
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
        # Remove possible markdown fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        facts = json.loads(raw)
        for fact in facts:
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
builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
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
def run_repl(app, config):
    """Interactive read-eval-print loop.

    A failure while handling one turn (LLM error, tool crash, bad checkpoint
    state, etc.) must not tear down the whole session — it is caught, surfaced
    to the user, and the loop continues to the next prompt. Only KeyboardInterrupt
    and EOFError intentionally end the session.
    """
    while True:
        try:
            user_input = input("\nYou: ")
        except (KeyboardInterrupt, EOFError):
            print("\nSession closing...")
            break

        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input.strip():
            continue

        try:
            events = app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            )
            for event in events:
                latest_msg = event["messages"][-1]
                if latest_msg.type == "ai" and latest_msg.content:
                    print(f"\nAgent: {latest_msg.content}")
                elif latest_msg.type == "tool":
                    print(f"\n[System: Executed '{latest_msg.name}' -> {len(latest_msg.content)} chars]")
        except (KeyboardInterrupt, EOFError):
            print("\nSession closing...")
            break
        except Exception as e:
            # Don't kill the session over a single failed turn.
            logger.exception("Error while processing turn")
            print(f"\n[Error: {e}]\nThe session is still active — try again or type 'quit' to exit.")


if __name__ == "__main__":
    print("===================================================================")
    print(" ⚠️ WARNING: THIS AGENT HAS UNRESTRICTED SHELL/FILE/WEB ACCESS ⚠️ ")
    print("===================================================================")
    config = {"configurable": {"thread_id": "root_access_session_1"}}

    if SQLITE_AVAILABLE:
        with SqliteSaver.from_conn_string(SQLITE_DB_PATH) as checkpointer:
            app = builder.compile(checkpointer=checkpointer)
            run_repl(app, config)
    else:
        checkpointer = MemorySaver()
        app = builder.compile(checkpointer=checkpointer)
        run_repl(app, config)
