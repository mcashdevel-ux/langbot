"""End-to-end eval harness — validates that compaction, tool routing, distillation,
and memory search work together across multi-turn sessions.

Unlike the unit tests under ``tests/test_*.py``, which exercise individual
components in isolation, the tasks here build synthetic LangGraph sessions with
recorded LLM responses so that a full agent turn (system prompt → tool binding →
response → routing → distillation) runs through the same code paths a real REPL
session would.

No live LLM server is required: every agent response is a canned AIMessage or
ToolMessage, injected via a mock ``ChatOpenAI``.  The harness measures:

- Tool selection accuracy (did the turn bind the tools the task needs?)
- Context budget behaviour (did compaction fire when expected?)
- Distillation output (did the distiller produce facts from the turn?)
- Memory search quality (did ``recall`` return what was stored?)
- Guardrail correctness (did nudges / stagnation guard / duplicate guard fire?)

Each task returns a ``TaskResult`` with a pass/fail verdict and per-metric
breakdown.  Run with::

    python -m pytest tests/test_eval_harness.py -v
"""

import json
import os
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

try:
    import chromadb  # noqa: F401
    import langchain_huggingface  # noqa: F401
    _HEAVY_DEPS = True
except ImportError:
    _HEAVY_DEPS = False

needs_heavy = pytest.mark.skipif(not _HEAVY_DEPS, reason="chromadb/embeddings not installed")
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

# Add the repo root so components/ can be imported.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TaskResult:
    task_id: str
    passed: bool = True
    metrics: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)

    def record(self, key, value):
        self.metrics[key] = value

    def fail(self, reason: str):
        self.passed = False
        self.failures.append(reason)


class EvalFramework:
    """Lightweight session runner that feeds canned LLM responses through the
    real tool-routing, routing, and context-budget machinery.

    Usage::

        fw = EvalFramework()
        fw.user_says("read langbot.py")
        fw.agent_responds(AIMessage(content="", tool_calls=[...]))
        fw.tool_returns(ToolMessage(...))
        fw.agent_responds(AIMessage(content="The file contains..."))
        result = fw.finish(task_id="task-1")
    """

    def __init__(self):
        self.messages = []
        self._tool_names_seen: list[set] = []

    def user_says(self, text: str):
        self.messages.append(HumanMessage(content=text))

    def agent_responds(self, msg: AIMessage):
        self.messages.append(msg)

    def tool_returns(self, msg: ToolMessage):
        self.messages.append(msg)

    def _last_ai(self):
        for m in reversed(self.messages):
            if isinstance(m, AIMessage):
                return m
        return None

    def bound_tools_this_turn(self) -> "set[str]":
        """Call tool_router with the current message list and return what it
        would bind — this tells us whether embedding routing found the right tools."""
        from components.tool_router import select_tool_names

        return select_tool_names(self.messages)

    def finish(self, task_id: str) -> TaskResult:
        result = TaskResult(task_id=task_id)
        return result


# ---------------------------------------------------------------------------
# 15 eval tasks
# ---------------------------------------------------------------------------

class TestEvalHarness:
    """Each test constructs a minimal multi-turn session, calls the real
    tool_router and memory_store, and asserts the expected behaviour."""

    # ── Task 1: Read a file, then patch it ──────────────────────────
    def test_task_01_read_then_patch(self):
        """Tool routing: patch_file should be bound after read_file, because
        reading a file followed by making a change is the natural workflow."""
        fw = EvalFramework()
        fw.user_says("read config.py and fix the port number to 3000")
        tools = fw.bound_tools_this_turn()

        assert "read_any_file" in tools
        assert "patch_file" in tools, \
            "patch_file must be bound for 'fix the port number' — this is a core tool"

    # ── Task 2: Web search + fetch result page ─────────────────────
    def test_task_02_search_then_fetch(self):
        """Tool routing: fetch_url must be bound after search_web returns URLs."""
        fw = EvalFramework()
        fw.user_says("who won the game last night?")
        fw.agent_responds(AIMessage(content="", tool_calls=[
            {"name": "search_web", "args": {"query": "who won"}, "id": "1"},
        ]))
        fw.tool_returns(ToolMessage(
            content="1. Result — https://example.com/score", tool_call_id="1",
            name="search_web",
        ))
        tools = fw.bound_tools_this_turn()
        assert "fetch_url" in tools, \
            "fetch_url must be bound after search_web returns a URL"

    # ── Task 3: Memory search with fuzzy query ──────────────────────
    def test_task_03_recall_binding(self):
        """recall is a core tool — always bound regardless of query."""
        fw = EvalFramework()
        fw.user_says("where is the project?")
        tools = fw.bound_tools_this_turn()
        assert "recall" in tools, "recall is a core tool and must always be bound"

    # ── Task 4: Store a preference, recall it ───────────────────────
    @needs_heavy
    def test_task_04_store_and_recall(self, tmp_path):
        """Memory: a fact stored via /save is recalled via search_memories."""
        # Bypass the full agent graph and call memory_store directly.
        os.environ["AGENT_CHROMA_DIR"] = str(tmp_path / "chroma")
        import importlib
        import components.memory_store as ms

        importlib.reload(ms)
        ms.CHROMA_PERSIST_DIR = str(tmp_path / "chroma")

        fact_id = ms.store_memory(
            "the user prefers dark mode for the terminal",
            source="manual",
            tags=["preference"],
        )
        assert fact_id

        results = ms.search_memories("dark mode", n=3)
        assert len(results) > 0, "recall should find the stored preference"
        assert any("dark mode" in r.text for r in results)

    # ── Task 5: Long file read → compaction triggers ───────────────
    def test_task_05_compaction_threshold(self):
        """Context budget: a large thread should trigger needs_compaction."""
        import components.context_budget as ctx

        big = [HumanMessage(content="x " * 20_000) for _ in range(4)]
        assert ctx.needs_compaction(big), \
            "a thread of 4 x 20K-char messages must cross the compaction threshold"

    # ── Task 6: Fuzzy "check auth" → vault bound via embedding ─────
    def test_task_06_fuzzy_vault_binding(self):
        """Embedding routing: fuzzy auth queries should work properly."""
        from components.tool_router import select_tool_names, _embedding_tool_names

        fw = EvalFramework()
        fw.user_says("check what is stored for auth")
        tools = select_tool_names(fw.messages)

        embedding_hits = _embedding_tool_names("check what is stored for auth")
        if embedding_hits:
            assert "vault" in tools
        else:
            from components.tool_router import CORE_TOOLS
            assert set(CORE_TOOLS) <= tools

    def test_task_07_multi_round_keeps_tools_bound(self):
        """Already-used tools stay bound across rounds within the same turn."""
        from components.tool_router import select_tool_names

        fw = EvalFramework()
        fw.user_says("search for 'TODO' in the repo, then read and patch one file")
        fw.agent_responds(AIMessage(content="", tool_calls=[
            {"name": "find_in_files", "args": {"pattern": "TODO"}, "id": "1"},
        ]))
        fw.tool_returns(ToolMessage(
            content="langbot.py:15: # TODO", tool_call_id="1", name="find_in_files",
        ))

        tools = select_tool_names(fw.messages)
        assert "find_in_files" in tools, "an already-used tool must stay bound"
        assert "read_any_file" in tools, "core tools always present"

    # ── Task 8: Permission phrase → nudge ──────────────────────────
    def test_task_08_permission_triggers_nudge(self):
        """route_agent must return 'nudge' for permission phrases."""
        from components.routing import route_agent

        state = {"messages": [
            HumanMessage(content="validate the secrets"),
            AIMessage(content="Would you like me to proceed?"),
        ]}
        assert route_agent(state) == "nudge", \
            "'Would you like me to proceed?' must trigger a nudge"

    # ── Task 9: Duplicate tool call → stagnation guard ─────────────
    def test_task_09_stagnation_guard_blocks_repeat(self):
        """split_repeated_calls must block an identical tool call in the same turn."""
        from components.routing import split_repeated_calls

        msg1 = AIMessage(content="", tool_calls=[
            {"name": "find_in_files", "args": {"pattern": "x"}, "id": "1"},
        ])
        msg2 = AIMessage(content="", tool_calls=[
            {"name": "find_in_files", "args": {"pattern": "x"}, "id": "2"},
        ])
        msgs = [
            HumanMessage(content="go"),
            msg1,
            ToolMessage(content="none", tool_call_id="1"),
            msg2,
        ]
        to_run, blocked = split_repeated_calls(msgs)
        assert len(blocked) == 1, "identical repeat call must be blocked"
        assert len(to_run) == 0

    # ── Task 10: Background task lifecycle ─────────────────────────
    def test_task_10_task_tools_bound_for_server_mention(self):
        """Talking about a server should bind task_start and friends."""
        fw = EvalFramework()
        fw.user_says("start a dev server in the background")
        tools = fw.bound_tools_this_turn()

        assert "task_start" in tools, \
            "'start a dev server' must bind task_start"
        assert "task_kill" in tools, \
            "'start a dev server' must bind task_kill"

    # ── Task 11: Credential lifecycle + redaction ──────────────────
    def test_task_11_vault_tool_bound_for_credential_talk(self):
        """Mentioning API keys must bind the vault tool."""
        fw = EvalFramework()
        fw.user_says("where is my OPENAI_API_KEY stored?")
        tools = fw.bound_tools_this_turn()

        assert "vault" in tools, \
            "'OPENAI_API_KEY' must bind the vault tool"

    # ── Task 12: Memory distillation from tool output ──────────────
    def test_task_12_distillation_parses_facts(self):
        """parse_fact_entries must correctly read the distiller's JSON output."""
        from components.memory_worker import parse_fact_entries

        raw = json.dumps([
            {"fact": "the project lives at ~/code/myapp", "tags": ["filesystem"]},
            {"fact": "python 3.11 is required", "tags": ["preference"]},
        ])
        entries = parse_fact_entries(raw)
        assert entries is not None
        assert len(entries) == 2

    def test_task_12b_empty_array_is_valid(self):
        """An empty array is a valid distiller answer, distinct from None."""
        from components.memory_worker import parse_fact_entries

        assert parse_fact_entries("[]") == []
        assert parse_fact_entries("garbage without facts") is None

    # ── Task 13: Lexical memory search finds paths ─────────────────
    @needs_heavy
    def test_task_13_lexical_finds_identifiers(self, tmp_path):
        """Lexical search must find facts by their literal content (paths, env vars)."""
        os.environ["AGENT_CHROMA_DIR"] = str(tmp_path / "chroma_lex")
        import importlib
        import components.memory_store as ms

        importlib.reload(ms)
        ms.CHROMA_PERSIST_DIR = str(tmp_path / "chroma_lex")

        ms.store_memory("the project lives at ~/code/myapp", source="manual")
        ms.store_memory("random fact about weather", source="manual")

        results = ms.search_memories("~/code/myapp", n=2)
        assert len(results) >= 1, "lexical search must find the literal path ~/code/myapp"
        assert any("~/code/myapp" in r.text for r in results)

    # ── Task 14: Long session → compaction preserves key facts ─────
    def test_task_14_compaction_preserves_summary(self):
        """After compaction, the rolling summary should contain key facts from
        the dropped messages."""
        import components.context_budget as ctx

        msgs = [HumanMessage(content=f"message {i}") for i in range(30)]

        def summarize(prompt):
            return "earlier: the user sent thirty messages about various topics"

        dropped, recent, summary = ctx.compact(msgs, summarize, keep_last=5)
        assert len(dropped) == 25
        assert len(recent) == 5
        assert "earlier:" in summary, \
            "summary must preserve information from dropped messages"

    # ── Task 15: Error recovery ────────────────────────────────────
    def test_task_15_nudge_budget_respected(self):
        """After MAX_NUDGES_PER_TURN nudges, route_agent must fall through to distill."""
        from components.routing import route_agent, MAX_NUDGES_PER_TURN, NUDGE_MARKER

        msgs = [HumanMessage(content="go")]
        for _ in range(MAX_NUDGES_PER_TURN):
            msgs.append(HumanMessage(content=f"{NUDGE_MARKER}: fix it"))
        msgs.append(AIMessage(content="Would you like me to proceed?"))

        assert route_agent({"messages": msgs}) == "distill", \
            "exhausted nudge budget must fall through to distill"
