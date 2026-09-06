"""Track D integration tests — the reflex node is wired into the graph.

These import the real langbot module (heavy: langchain/langgraph/chromadb),
so they're kept in their own file and can be skipped in constrained
environments if needed. They verify:

  - a matching rule fires through ``reflex_node`` and returns an AIMessage
    answer without invoking the LLM (the ``agent`` node is never reached);
  - the answer carries the ``[REFLEX]`` marker so routing/rendering can tell it
    apart from an LLM answer;
  - a non-matching message falls through (``route_reflex`` → ``compact``);
  - ``reflex_distill`` stores a rule the node then fires.

 The store is redirected to a temp file so tests never touch the real
 ``./memory/reflexes.json``.

"""

import os

import pytest

os.environ.setdefault("LANGBOT_VAULT_PASSWORD", "test-only-password")

langbot = pytest.importorskip(
    "langbot", reason="requires the full runtime dependency set (langchain/langgraph/chromadb)"
)

from langchain_core.messages import AIMessage, HumanMessage


@pytest.fixture(autouse=True)
def reflex_file(tmp_path, monkeypatch):
    """Redirect the reflex store into a temp file for every test."""
    p = tmp_path / "reflexes.json"
    monkeypatch.setattr("components.reflex.REFLEXES_FILE", str(p))
    # Re-point the already-constructed store at the new path.

    langbot._reflex_store = langbot._ReflexStore(path=p)
    return p


class TestReflexNode:
    def test_matching_rule_fires_without_llm(self, monkeypatch):
        langbot._reflex_store.distill("check disk space", "echo reflex-fired-ok")
        called = {}

        def _fake_agent(state):
            called["agent"] = True
            raise AssertionError("agent node must not run when a reflex fires")

        monkeypatch.setattr(langbot, "agent", _fake_agent)

        state = {"messages": [HumanMessage(content="please check the disk space now")]}
        out = langbot.reflex_node(state)
        assert called == {}
        msgs = out["messages"]
        assert len(msgs) == 1
        assert isinstance(msgs[0], AIMessage)
        assert "[REFLEX]" in msgs[0].content
        assert "reflex-fired-ok" in msgs[0].content

    def test_use_counter_increments(self):
        rid = langbot._reflex_store.distill("check disk space", "echo hi")
        langbot.reflex_node({"messages": [HumanMessage(content="please check the disk space")]})
        langbot.reflex_node({"messages": [HumanMessage(content="please check the disk space again")]})
        rules = langbot._reflex_store.all()
        assert rules[0]["uses"] == 2

    def test_no_match_falls_through(self):
        langbot._reflex_store.distill("check disk space", "echo hi")
        out = langbot.reflex_node({"messages": [HumanMessage(content="what is the weather")]})
        assert out == {}

    def test_disabled_rule_does_not_fire(self):
        rid = langbot._reflex_store.distill("check disk space", "echo hi")
        langbot._reflex_store.disable(rid)
        out = langbot.reflex_node({"messages": [HumanMessage(content="please check the disk space")]})
        assert out == {}

    def test_catastrophic_command_is_refused_by_node(self, monkeypatch):
        langbot._reflex_store.distill("wipe everything", "rm -rf /")
        called = {}

        def _fake_run(*args, **kwargs):
            called["ran"] = True
            raise AssertionError("subprocess.run must not run for a catastrophic reflex command")

        monkeypatch.setattr(langbot.subprocess, "run", _fake_run)
        out = langbot.reflex_node({"messages": [HumanMessage(content="please wipe everything now")]})
        assert called == {}
        assert "Refused" in out["messages"][0].content


class TestRouteReflex:
    def test_match_routes_to_distill(self):
        state = {"messages": [AIMessage(content="[REFLEX] check disk space → `df -h`\n\nok")]}
        assert langbot.route_reflex(state) == "distill"

    def test_no_match_routes_to_compact(self):
        state = {"messages": [HumanMessage(content="hello")]}
        assert langbot.route_reflex(state) == "compact"


class TestReflexDistillTool:
    def test_stores_rule(self):
        res = langbot.reflex_distill.func("check disk space", "df -h")
        assert "Reflex stored" in res
        rules = langbot._reflex_store.all()
        assert len(rules) == 1
        assert rules[0]["trigger"] == "check disk space"

    def test_rejects_empty_fields(self):
        res = langbot.reflex_distill.func("", "df -h")
        assert "both 'trigger' and 'command'" in res
        assert langbot._reflex_store.all() == []