"""Tests for tool_call_repair — recovering textual tool calls from weak models.

The motivating case is a local Qwen LoRA that answered every prompt with
``{"content": ..., "tool_calls": [...]}`` as plain text, so nothing executed.
"""

import pytest

from components import tool_call_repair
from components.tool_call_repair import (
    parse_tool_calls,
    repair_message,
    strip_reasoning,
    unwrap_content,
)

NAMES = {"glob_list", "remember", "recall", "execute_shell_command", "read_any_file"}
ALIASES = {"recall": {"q": "query", "text": "query", "limit": "n"},
           "remember": {"text": "fact"}}


class FakeAI:
    """Minimal stand-in for AIMessage (repair_message only duck-types these)."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class TestArgAliases:
    """Models that get the protocol wrong also get parameter names wrong."""

    def test_alias_is_renamed_to_the_real_parameter(self):
        raw = '{"name": "recall", "args": {"q": "where is the repo", "limit": 5}}'
        _, calls = parse_tool_calls(raw, NAMES, ALIASES)
        assert calls[0]["args"] == {"query": "where is the repo", "n": 5}

    def test_correct_name_wins_over_its_alias(self):
        raw = '{"name": "recall", "args": {"query": "real", "text": "stray"}}'
        _, calls = parse_tool_calls(raw, NAMES, ALIASES)
        assert calls[0]["args"] == {"query": "real"}

    def test_unknown_tools_and_args_are_left_alone(self):
        raw = '{"name": "glob_list", "args": {"text": "*.py"}}'
        _, calls = parse_tool_calls(raw, NAMES, ALIASES)
        assert calls[0]["args"] == {"text": "*.py"}

    def test_no_alias_table_is_a_no_op(self):
        raw = '{"name": "recall", "args": {"q": "where is the repo"}}'
        _, calls = parse_tool_calls(raw, NAMES)
        assert calls[0]["args"] == {"q": "where is the repo"}


class TestParse:
    def test_wrapper_object_from_the_bug_report(self):
        raw = ('{"content": "Analyzing the current working directory.", '
               '"tool_calls": [{"name": "glob_list", "args": {"pattern": ".", '
               '"max_results": 10}}]}')
        content, calls = parse_tool_calls(raw, NAMES)
        assert content == "Analyzing the current working directory."
        assert len(calls) == 1
        assert calls[0]["name"] == "glob_list"
        assert calls[0]["args"] == {"pattern": ".", "max_results": 10}
        assert calls[0]["type"] == "tool_call"
        assert calls[0]["id"]

    def test_multiple_calls(self):
        raw = ('{"tool_calls": [{"name": "remember", "args": {"fact": "a"}}, '
               '{"name": "glob_list", "args": {"pattern": "*.py"}}]}')
        _, calls = parse_tool_calls(raw, NAMES)
        assert [c["name"] for c in calls] == ["remember", "glob_list"]

    def test_fenced_json(self):
        raw = '```json\n{"tool_calls": [{"name": "glob_list", "args": {"pattern": "."}}]}\n```'
        _, calls = parse_tool_calls(raw, NAMES)
        assert len(calls) == 1

    def test_json_embedded_in_prose_keeps_the_prose(self):
        raw = ('Let me look around.\n'
               '{"tool_calls": [{"name": "glob_list", "args": {"pattern": "."}}]}\n'
               'Then I will report back.')
        content, calls = parse_tool_calls(raw, NAMES)
        assert len(calls) == 1
        assert "Let me look around." in content
        assert "Then I will report back." in content
        assert "tool_calls" not in content

    def test_openai_function_shape_with_string_arguments(self):
        raw = ('{"tool_calls": [{"id": "call_7", "function": {"name": '
               '"execute_shell_command", "arguments": "{\\"command\\": \\"ls\\"}"}}]}')
        _, calls = parse_tool_calls(raw, NAMES)
        assert calls[0]["args"] == {"command": "ls"}
        assert calls[0]["id"] == "call_7"

    def test_bare_single_call_object(self):
        content, calls = parse_tool_calls('{"name": "glob_list", "args": {"pattern": "*"}}', NAMES)
        assert calls[0]["name"] == "glob_list"
        assert content == ""

    def test_singular_tool_call_key_and_missing_args(self):
        _, calls = parse_tool_calls('{"tool_call": {"name": "glob_list"}}', NAMES)
        assert calls[0]["args"] == {}

    def test_alternate_arg_keys(self):
        _, calls = parse_tool_calls(
            '{"tool_calls": [{"tool": "remember", "parameters": {"fact": "x"}}]}', NAMES)
        assert calls[0]["name"] == "remember"
        assert calls[0]["args"] == {"fact": "x"}

    def test_nested_braces_in_args(self):
        raw = ('{"tool_calls": [{"name": "remember", "args": {"fact": '
               '"config is {\\"a\\": {\\"b\\": 1}}"}}]}')
        _, calls = parse_tool_calls(raw, NAMES)
        assert calls[0]["args"]["fact"] == 'config is {"a": {"b": 1}}'


class TestQwenXmlFormat:
    """Qwen3's own chat template documents <tool_call>{...}</tool_call> blocks,
    which arrive as content whenever the server does not parse the tags."""

    def test_single_tag(self):
        raw = ('Let me check.\n<tool_call>\n{"name": "glob_list", '
               '"arguments": {"pattern": "*.py"}}\n</tool_call>')
        content, calls = parse_tool_calls(raw, NAMES)
        assert content == "Let me check."
        assert calls[0]["name"] == "glob_list"
        assert calls[0]["args"] == {"pattern": "*.py"}

    def test_two_tags(self):
        raw = ('<tool_call>{"name": "remember", "arguments": {"fact": "a"}}</tool_call>'
               '<tool_call>{"name": "glob_list", "arguments": {"pattern": "."}}</tool_call>')
        content, calls = parse_tool_calls(raw, NAMES)
        assert [c["name"] for c in calls] == ["remember", "glob_list"]
        assert content == ""

    def test_unclosed_tag_from_a_truncated_generation(self):
        raw = '<tool_call>{"name": "glob_list", "arguments": {"pattern": "."}}'
        _, calls = parse_tool_calls(raw, NAMES)
        assert calls[0]["name"] == "glob_list"

    @pytest.mark.parametrize("raw, expected", [
        ("<think>hmm</think>\nThe answer.", "The answer."),
        ("<thinking>hmm</thinking>The answer.", "The answer."),
        ("The answer.<think>cut off mid-thought", "The answer."),
        ("No reasoning here.", "No reasoning here."),
    ])
    def test_strip_reasoning(self, raw, expected):
        assert strip_reasoning(raw) == expected

    def test_leaked_markup_is_stripped_from_the_answer(self):
        msg = FakeAI("<tool_response> I have recalled relevant information.<|im_end|>")
        assert repair_message(msg, NAMES) is True
        assert msg.content == "I have recalled relevant information."
        assert msg.tool_calls == []


class TestNoFalsePositives:
    @pytest.mark.parametrize("text", [
        "Here is the answer, no JSON at all.",
        "",
        '{"content": "just a content field"}',
        '{"tool_calls": [{"name": "rm_rf_everything", "args": {}}]}',   # unknown tool
        '{"tool_calls": [{"name": "glob_list", "args": "not-an-object"}]}',
        '{"tool_calls": [{"name": "glob_list", "args": "{bad json"}]}',
        '{"tool_calls": "glob_list"}',
        "{unbalanced",
    ])
    def test_content_is_returned_unchanged(self, text):
        content, calls = parse_tool_calls(text, NAMES)
        assert calls == []
        assert content == text

    def test_prose_mentioning_a_tool_name_is_untouched(self):
        text = 'You can call `glob_list` with {"pattern": "."} to list files.'
        content, calls = parse_tool_calls(text, NAMES)
        assert calls == []
        assert content == text

    def test_non_string_content(self):
        content, calls = parse_tool_calls([{"type": "text"}], NAMES)
        assert calls == []
        assert content == [{"type": "text"}]

    def test_candidate_scan_is_bounded(self, monkeypatch):
        monkeypatch.setattr(tool_call_repair, "MAX_CANDIDATES", 2)
        noise = "{} " * 5
        raw = noise + '{"tool_calls": [{"name": "glob_list", "args": {}}]}'
        _, calls = parse_tool_calls(raw, NAMES)
        assert calls == []  # real payload sat past the scan limit


class TestRepairMessage:
    def test_repairs_in_place(self):
        msg = FakeAI('{"content": "Looking.", "tool_calls": '
                     '[{"name": "glob_list", "args": {"pattern": "."}}]}')
        assert repair_message(msg, NAMES) is True
        assert msg.content == "Looking."
        assert msg.tool_calls[0]["name"] == "glob_list"

    def test_leaves_native_tool_calls_alone(self):
        native = [{"name": "remember", "args": {"fact": "x"}, "id": "1", "type": "tool_call"}]
        msg = FakeAI('{"tool_calls": [{"name": "glob_list", "args": {}}]}', native)
        assert repair_message(msg, NAMES) is False
        assert msg.tool_calls == native

    def test_leaves_a_normal_answer_alone(self):
        msg = FakeAI("The current directory holds 12 files.")
        assert repair_message(msg, NAMES) is False
        assert msg.tool_calls == []

    def test_disabled_by_config(self, monkeypatch):
        monkeypatch.setattr(tool_call_repair, "REPAIR_ENABLED", False)
        msg = FakeAI('{"tool_calls": [{"name": "glob_list", "args": {}}]}')
        assert repair_message(msg, NAMES) is False
        assert msg.tool_calls == []

    def test_logs_the_recovery(self, caplog):
        msg = FakeAI('{"tool_calls": [{"name": "glob_list", "args": {}}]}')
        with caplog.at_level("WARNING"):
            repair_message(msg, NAMES)
        assert "glob_list" in caplog.text


class TestUnwrapContent:
    def test_call_less_envelope_is_unwrapped(self):
        raw = '{"content": "The directory holds 12 files.", "tool_calls": []}'
        assert unwrap_content(raw) == "The directory holds 12 files."

    def test_fenced_and_thought_bearing_envelope(self):
        raw = '```json\n{"thought": "listing", "content": "Done.", "tool_calls": []}\n```'
        assert unwrap_content(raw) == "Done."

    @pytest.mark.parametrize("text", [
        "A plain answer.",
        '["a", "json", "array"]',
        '{"content": "x", "tool_calls": [{"name": "glob_list", "args": {}}]}',  # has calls
        '{"content": "x", "status": "ok"}',                                     # stray key
        '{"content": {"nested": 1}, "tool_calls": []}',                         # not a string
        '{"tool_calls": []}',                                                   # no content
        "{bad json",
        "",
    ])
    def test_left_alone(self, text):
        assert unwrap_content(text) is None

    def test_repair_message_unwraps_a_call_less_envelope(self):
        msg = FakeAI('{"content": "I have recalled the greeting.", "tool_calls": []}')
        assert repair_message(msg, NAMES) is True
        assert msg.content == "I have recalled the greeting."
        assert msg.tool_calls == []


class TestRoutingIntegration:
    def test_repaired_message_routes_to_tools(self):
        """The whole point: after repair, route_agent must dispatch to tools."""
        from langchain_core.messages import AIMessage, HumanMessage

        from components.routing import route_agent

        msg = AIMessage(content='{"content": "Analyzing.", "tool_calls": '
                                '[{"name": "glob_list", "args": {"pattern": "."}}]}')
        assert route_agent({"messages": [HumanMessage(content="analyze cwd"), msg]}) != "tools"
        assert repair_message(msg, NAMES) is True
        assert route_agent({"messages": [HumanMessage(content="analyze cwd"), msg]}) == "tools"
