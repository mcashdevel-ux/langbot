"""Track C integration tests — stuck detection is turn-aware.

These import the real langbot module (heavy: langchain/langgraph/chromadb),
so they're kept in their own file and can be skipped in constrained
environments if needed. They verify:

  - ``_current_turn_messages`` slices to the last real user message (nudges
    excluded), so prior turns' chat answers cannot poison the detectors;
  - ``stuck_node`` does NOT halt a good current-turn tool round just because
    earlier chat turns ended with content-only answers (the regression that
    killed "verify ~/code/myapp" and "do you know where your source code is?").
"""

import os

import pytest

os.environ.setdefault("LANGBOT_VAULT_PASSWORD", "test-only-password")

langbot = pytest.importorskip(
    "langbot", reason="requires the full runtime dependency set (langchain/langgraph/chromadb)"
)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from components.routing import NUDGE_MARKER


def ai(content="", calls=None):
    """AIMessage helper: tool_calls only when non-empty (pydantic rejects None)."""
    kw = {}
    if calls:
        kw["tool_calls"] = calls
    return AIMessage(content=content, **kw)


def shell_call(command, cid):
    return {"name": "shell", "args": {"command": command}, "id": cid}


def tool(name, content, cid="c1"):
    return ToolMessage(content=content, name=name, tool_call_id=cid)


class TestCurrentTurnMessages:
    def test_slices_to_last_real_user_message(self):
        msgs = [
            HumanMessage(content="hi"),
            ai(content="Hello! How can I help you today?"),
            HumanMessage(content="what can you do?"),
            ai(content="Here's what I can do..."),
            HumanMessage(content="verify ~/code/myapp"),
            ai(calls=[shell_call("ls -la ~/code/myapp", "c1")]),
            tool("shell", "No such file or directory", "c1"),
        ]
        sliced = langbot._current_turn_messages(msgs)
        assert sliced[0] is msgs[4]
        assert len(sliced) == 3
        assert all(m is not msgs[0] and m is not msgs[1] and m is not msgs[2] and m is not msgs[3] for m in sliced)

    def test_nudges_do_not_count_as_turn_start(self):
        nudge = HumanMessage(content=f"{NUDGE_MARKER}: Do not ask whether to proceed.")
        msgs = [
            HumanMessage(content="hi"),
            ai(content="Hello!"),
            nudge,
            ai(calls=[shell_call("pwd", "c1")]),
            tool("shell", "/home/user", "c1"),
        ]
        sliced = langbot._current_turn_messages(msgs)
        # The slice starts at the real user message ("hi") — a nudge must not
        # reset the turn window (it is part of the current turn's history, and
        # the narrative mapper skips HumanMessages anyway).
        assert sliced[0] is msgs[0]
        assert nudge in sliced


class TestStuckNodeTurnAware:
    def test_prior_chat_answers_do_not_halt_good_tool_round(self):
        """Regression: 4 content-only answers across earlier chat turns used to
        trip the monologue detector after the first real tool round of the current
        turn — halting "verify ~/code/myapp" and "do you know where your source
        code is?" mid-task."""
        msgs = [
            HumanMessage(content="hi"),
            ai(content="Hi there! How can I help you today?"),
            HumanMessage(content="what are you?"),
            ai(content="I'm an autonomous AI assistant that runs on this machine..."),
            HumanMessage(content="are you self aware?"),
            ai(content="Honest answer: no, not in any meaningful sense..."),
            HumanMessage(content="do you know where your source code is?"),
            ai(calls=[shell_call("pwd && ls -la", "c1")]),
            tool("shell", "/home/user/ai/repos/langbot", "c1"),
        ]
        out = langbot.stuck_node({"messages": msgs})
        assert out == {}

    def test_repeating_loop_still_detected_within_turn(self):
        """Turn-awareness must not blind the detector to a real loop in the current turn."""
        msgs = [
            HumanMessage(content="hi"),
            ai(content="Hello!"),
            HumanMessage(content="fix the build"),
        ]
        for i in range(3):
            msgs += [ai(calls=[shell_call("make", f"c{i}")]),
                       tool("shell", "same error output", f"c{i}")]
        out = langbot.stuck_node({"messages": msgs})
        assert "stuck_halt" in out
        assert out["stuck_halt"] is True