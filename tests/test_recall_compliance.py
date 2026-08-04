import logging
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langbot import distill_knowledge, AgentState


def test_recall_compliance_check(caplog):
    # Enable caplog to capture our compliance INFO logs
    caplog.set_level(logging.INFO)

    # 1. State where assistant answers memory-dependent question but didn't recall
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="As you mentioned earlier, Python is your favorite language.")
        ]
    }
    
    # We mock out _search_memories or other dependencies to avoid database connections
    with patch("langbot._memory_worker.qsize", return_value=0):
        res = distill_knowledge(state)
        
    assert any(
        "compliance check: final answer contains memory-referencing phrase" in r.message
        for r in caplog.records
    )


def test_recall_compliance_check_passed(caplog):
    caplog.clear()
    caplog.set_level(logging.INFO)

    # 2. State where assistant answered memory-dependent question AND recalled
    ai_msg_with_call = AIMessage(
        content="As you mentioned, Python is your favorite language.",
        tool_calls=[{"name": "recall", "args": {"query": "language"}, "id": "1"}]
    )
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            ai_msg_with_call,
            ToolMessage(content="Result - fav language: Python", tool_call_id="1", name="recall")
        ]
    }
    
    with patch("langbot._memory_worker.qsize", return_value=0):
        distill_knowledge(state)
        
    # No compliance warnings should be logged since 'recall' was invoked!
    assert not any(
        "compliance check:" in r.message
        for r in caplog.records
    )
