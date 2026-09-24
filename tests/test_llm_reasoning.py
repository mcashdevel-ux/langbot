"""Tests for ReasoningChatOpenAI — keeping the provider's reasoning channel.

``langchain_openai.ChatOpenAI`` documents that non-standard provider fields
(``reasoning_content``, ``reasoning_details``) are not extracted or preserved, and
the hosted proxy this agent uses returns exactly those. So a subclass copies the
reasoning into ``additional_kwargs``, which is part of the message and therefore
survives checkpointing.

The fake responses below mirror the shapes the real endpoint returns, captured
from a live request: ``reasoning_content`` plus a ``provider_specific_fields``
envelope, with the field exposed on the SDK model's ``model_extra``.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from components.llm_reasoning import ReasoningChatOpenAI, _raw_reasoning

REASONING = "17*20=340, 17*3=51, so 391."
OPEN = "<" "think>"
CLOSE = "</" "think>"


def _response(message: dict, usage: dict | None = None) -> dict:
    return {
        "id": "gen-test", "object": "chat.completion", "created": 0,
        "model": "deepseek-v4.1-flash",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": usage or {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _message(content: str = "391", **extra) -> dict:
    base = {"role": "assistant", "content": content, "refusal": None}
    base.update(extra)
    return base


class TestRawReasoning:
    def test_reasoning_content_field(self):
        resp = _response(_message(reasoning_content=REASONING))
        assert _raw_reasoning(resp) == REASONING

    def test_provider_specific_fields_envelope(self):
        resp = _response(_message(provider_specific_fields={
            "reasoning": REASONING,
            "reasoning_details": [{"type": "reasoning.text", "text": REASONING}],
        }))
        assert _raw_reasoning(resp) == REASONING

    def test_sdk_model_extra_is_read(self):
        # What the openai SDK actually hands langchain: the undeclared field lands
        # in model_extra rather than becoming an attribute.
        class Msg:
            content = "391"
            model_extra = {"reasoning_content": REASONING}

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        assert _raw_reasoning(Resp()) == REASONING

    def test_absent_reasoning_is_empty(self):
        assert _raw_reasoning(_response(_message())) == ""

    def test_malformed_response_does_not_raise(self):
        for bad in ({}, {"choices": []}, {"choices": [{}]}, None):
            assert _raw_reasoning(bad) == ""


class TestCreateChatResult:
    """The subclass must add the reasoning *and* leave langchain's own parsing alone."""

    def _result(self, message):
        llm = ReasoningChatOpenAI(model="m", api_key="not-needed", base_url="http://x/v1")
        return llm._create_chat_result(_response(message))

    def test_reasoning_lands_in_additional_kwargs(self):
        result = self._result(_message(reasoning_content=REASONING))
        msg = result.generations[0].message
        assert msg.additional_kwargs["reasoning_content"] == REASONING
        assert msg.content == "391"

    def test_reasoning_survives_a_json_round_trip(self):
        # The checkpointer persists messages, so the reasoning has to be part of
        # the message data rather than an attribute that would not come back.
        msg = self._result(_message(reasoning_content=REASONING)).generations[0].message
        revived = AIMessage(**json.loads(json.dumps({
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs,
            "type": msg.type,
        })))
        assert revived.additional_kwargs["reasoning_content"] == REASONING

    def test_clean_answer_is_untouched_when_there_is_no_reasoning(self):
        result = self._result(_message(content="plain"))
        msg = result.generations[0].message
        assert msg.content == "plain"
        assert "reasoning_content" not in msg.additional_kwargs

    def test_copying_is_idempotent(self):
        # Parsing the same response twice must not duplicate or alter the field;
        # the guard is what keeps a message that already carries reasoning intact.
        llm = ReasoningChatOpenAI(model="m", api_key="not-needed", base_url="http://x/v1")
        resp = _response(_message(reasoning_content=REASONING))
        first = llm._create_chat_result(resp).generations[0].message
        second = llm._create_chat_result(resp).generations[0].message
        assert first.additional_kwargs == second.additional_kwargs
        assert first.additional_kwargs["reasoning_content"] == REASONING

    def test_tool_calls_still_parse(self):
        result = self._result(_message(content="", tool_calls=[{
            "id": "call_1", "type": "function",
            "function": {"name": "execute_shell_command",
                         "arguments": json.dumps({"command": "df -h"})},
        }], reasoning_content=REASONING))
        msg = result.generations[0].message
        assert msg.tool_calls[0]["name"] == "execute_shell_command"
        assert msg.tool_calls[0]["args"] == {"command": "df -h"}

    def test_inline_think_tags_are_not_touched_here(self):
        # Inline tags are a different channel: they stay in the content and are
        # split out at render time, so the subclass must not rewrite the answer.
        result = self._result(_message(content=f"{OPEN}because{CLOSE}391"))
        assert result.generations[0].message.content == f"{OPEN}because{CLOSE}391"
