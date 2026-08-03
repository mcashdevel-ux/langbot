"""Unit tests for tool_router.py — which tool schemas get bound per turn."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import components.tool_router as tool_router


class _FakeTool:
    def __init__(self, name):
        self.name = name


ALL = [_FakeTool(n) for n in [
    "execute_shell_command", "read_any_file", "write_any_file", "patch_file",
    "batch_patch", "git_diff", "find_in_files", "read_many_files", "glob_list",
    "task_start", "task_list", "task_status", "task_output", "task_kill",
    "search_web", "fetch_url", "read_scratch", "remember", "recall", "vault",
]]


def _names(messages):
    return {t.name for t in tool_router.select_tools(ALL, messages)}


class TestSelection:
    def test_core_tools_are_always_bound(self):
        names = _names([HumanMessage(content="hello")])
        assert set(tool_router.CORE_TOOLS) <= names

    def test_a_plain_greeting_binds_far_fewer_tools(self):
        assert len(_names([HumanMessage(content="hello")])) < len(ALL)

    def test_credential_talk_binds_the_vault(self):
        assert "vault" in _names([HumanMessage(content="is the api key stored?")])

    def test_git_talk_binds_the_diff_tool(self):
        assert "git_diff" in _names([HumanMessage(content="what changed in the repo?")])

    def test_background_work_binds_the_task_tools(self):
        names = _names([HumanMessage(content="start the dev server in the background")])
        assert {"task_start", "task_kill"} <= names

    def test_a_url_in_a_tool_result_binds_fetch_url(self):
        # search_web results carry urls; fetching one is the natural next step.
        messages = [
            HumanMessage(content="who won?"),
            AIMessage(content="", tool_calls=[
                {"name": "search_web", "args": {"query": "who won"}, "id": "1"},
            ]),
            ToolMessage(content="1. Result — https://example.com/x",
                        tool_call_id="1", name="search_web"),
        ]
        assert "fetch_url" in _names(messages)

    def test_a_tool_used_this_turn_stays_bound(self):
        # Unbinding mid-task would strand a multi-round job.
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[
                {"name": "glob_list", "args": {"pattern": "*"}, "id": "1"},
            ]),
            ToolMessage(content="a\nb", tool_call_id="1", name="glob_list"),
        ]
        assert "glob_list" in _names(messages)

    def test_a_tool_used_in_a_previous_turn_is_released(self):
        messages = [
            HumanMessage(content="hello"),
            AIMessage(content="", tool_calls=[
                {"name": "vault", "args": {"action": "list"}, "id": "1"},
            ]),
            ToolMessage(content="none", tool_call_id="1", name="vault"),
            HumanMessage(content="thanks"),
        ]
        assert "vault" not in _names(messages)

    def test_scratch_reference_binds_the_pager(self):
        messages = [
            HumanMessage(content="read it"),
            AIMessage(content="saved at scratch:file_1234abcd"),
        ]
        assert "read_scratch" in _names(messages)

    def test_dynamic_binding_can_be_disabled(self, monkeypatch):
        monkeypatch.setattr(tool_router, "DYNAMIC_BINDING", False)
        assert len(_names([HumanMessage(content="hello")])) == len(ALL)

    def test_an_empty_core_list_falls_back_to_every_tool(self, monkeypatch):
        # A model with no tools cannot act, so a bad core list must not strand it.
        monkeypatch.setattr(tool_router, "CORE_TOOLS", [])
        assert len(_names([HumanMessage(content="hello")])) == len(ALL)
