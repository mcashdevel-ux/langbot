"""Unit tests for routing.py — agent routing, nudges, duplicate-answer guard."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import components.routing as routing


def _nudge(text=None):
    """A nudge as the graph appends it: a HumanMessage carrying the marker."""
    return HumanMessage(content=text or routing.NUDGE_PERMISSION)


def _state(*messages):
    return {"messages": list(messages)}


class TestRouteAgent:
    def test_tool_calls_go_to_tools(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "read_any_file", "args": {"file_path": "x"}, "id": "1"},
        ])
        assert routing.route_agent(_state(HumanMessage(content="hi"), ai)) == "tools"

    def test_plain_answer_goes_to_distill(self):
        state = _state(HumanMessage(content="hi"), AIMessage(content="all done"))
        assert routing.route_agent(state) == "distill"

    def test_permission_phrase_nudges(self):
        state = _state(
            HumanMessage(content="validate the secrets"),
            AIMessage(content="Would you like me to proceed?"),
        )
        assert routing.route_agent(state) == "nudge"

    def test_code_block_pattern_nudges(self):
        state = _state(
            HumanMessage(content="search for x"),
            AIMessage(content="```python\nimport search_web\n```"),
        )
        assert routing.route_agent(state) == "nudge"

    def test_nudge_budget_exhausted_falls_through_to_distill(self):
        msgs = [HumanMessage(content="go")]
        msgs += [_nudge() for _ in range(routing.MAX_NUDGES_PER_TURN)]
        msgs.append(AIMessage(content="Would you like me to proceed?"))
        assert routing.route_agent(_state(*msgs)) == "distill"

    def test_tool_rounds_do_not_spend_the_nudge_budget(self):
        # The budget exists to stop a nudge loop; ordinary tool work must not
        # consume it, or the guardrails switch off on exactly the long tasks
        # they are there for.
        msgs = [HumanMessage(content="go")]
        for i in range(routing.MAX_NUDGES_PER_TURN + 3):
            msgs.append(AIMessage(content="working", tool_calls=[
                {"name": "recall", "args": {"query": "q"}, "id": str(i)},
            ]))
            msgs.append(ToolMessage(content="r", tool_call_id=str(i), name="recall"))
        msgs.append(AIMessage(content="Would you like me to proceed?"))
        assert routing.route_agent(_state(*msgs)) == "nudge"

    def test_duplicate_final_answer_goes_to_distill(self):
        state = _state(
            HumanMessage(content="hi"),
            AIMessage(content="answer 1"),
            AIMessage(content="answer 2"),
        )
        assert routing.route_agent(state) == "distill"

    def test_duplicate_guard_skips_the_nudge_path(self):
        # A second answer containing a permission phrase must NOT trigger
        # another nudge round-trip — the guard short-circuits to distill.
        state = _state(
            HumanMessage(content="hi"),
            AIMessage(content="answer 1"),
            AIMessage(content="Would you like me to proceed?"),
        )
        assert routing.route_agent(state) == "distill"

    def test_guard_does_not_block_first_answer_after_tool_use(self):
        ai_call = AIMessage(content="", tool_calls=[
            {"name": "recall", "args": {"query": "q"}, "id": "1"},
        ])
        state = _state(
            HumanMessage(content="hi"),
            ai_call,
            ToolMessage(content="result", tool_call_id="1", name="recall"),
            AIMessage(content="Would you like me to proceed?"),
        )
        assert routing.route_agent(state) == "nudge"

    def test_answer_from_a_previous_turn_does_not_trip_the_guard(self):
        state = _state(
            HumanMessage(content="first"),
            AIMessage(content="first answer"),
            HumanMessage(content="second"),
            AIMessage(content="Would you like me to proceed?"),
        )
        assert routing.route_agent(state) == "nudge"


class TestCounters:
    def test_ai_turns_since_human(self):
        msgs = [
            HumanMessage(content="a"),
            AIMessage(content="1"),
            _nudge(),
            AIMessage(content="2"),
        ]
        assert routing.ai_turns_since_human(msgs) == 2

    def test_final_answers_since_human_ignores_tool_calls(self):
        msgs = [
            HumanMessage(content="a"),
            AIMessage(content="thinking", tool_calls=[
                {"name": "recall", "args": {"query": "q"}, "id": "1"},
            ]),
            ToolMessage(content="r", tool_call_id="1", name="recall"),
            AIMessage(content="final"),
        ]
        assert routing.final_answers_since_human(msgs) == 1

    def test_final_answers_stops_at_last_human(self):
        msgs = [
            HumanMessage(content="a"),
            AIMessage(content="old answer"),
            HumanMessage(content="b"),
        ]
        assert routing.final_answers_since_human(msgs) == 0


class TestNudgeAgent:
    def test_nudge_is_not_a_system_message(self):
        # Served chat templates reject a system message that is not first, so a
        # mid-conversation nudge must arrive in the user role instead.
        out = routing.nudge_agent(_state(AIMessage(content="Should I proceed?")))
        message = out["messages"][0]
        assert isinstance(message, HumanMessage)
        assert routing.is_nudge(message)

    def test_nudges_do_not_end_the_turn_they_correct(self):
        # The nudge is a HumanMessage, so the counters must not mistake it for
        # the user speaking: otherwise the budget resets and the guard forgets
        # the answer it just rejected.
        msgs = [HumanMessage(content="go"), AIMessage(content="answer 1"), _nudge()]
        assert routing.nudges_since_human(msgs) == 1
        assert routing.final_answers_since_human(msgs) == 1
        assert routing.ai_turns_since_human(msgs) == 1

    def test_code_block_failure_gets_code_block_nudge(self):
        state = _state(AIMessage(content="```bash\ncurl https://x\n```"))
        out = routing.nudge_agent(state)["messages"][0]
        assert out.content == routing.NUDGE_CODE_BLOCK

    def test_permission_failure_gets_permission_nudge(self):
        state = _state(AIMessage(content="Should I proceed?"))
        out = routing.nudge_agent(state)["messages"][0]
        assert out.content == routing.NUDGE_PERMISSION


def _ai_call(name, args, call_id="1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class TestStagnationGuard:
    def test_a_first_time_call_runs(self):
        msgs = [HumanMessage(content="go"), _ai_call("find_in_files", {"pattern": "x"})]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(to_run) == 1 and blocked == []

    def test_an_identical_repeat_is_blocked(self):
        call = {"pattern": "x", "path": "."}
        msgs = [
            HumanMessage(content="go"),
            _ai_call("find_in_files", call, "1"),
            ToolMessage(content="no matches", tool_call_id="1"),
            _ai_call("find_in_files", call, "2"),
        ]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert to_run == []
        assert len(blocked) == 1
        assert blocked[0].tool_call_id == "2"
        assert blocked[0].content == routing.REPEATED_CALL_NOTICE

    def test_key_order_does_not_hide_a_repeat(self):
        msgs = [
            HumanMessage(content="go"),
            _ai_call("execute_shell_command", {"command": "ls", "cwd": "/tmp"}, "1"),
            ToolMessage(content="out", tool_call_id="1"),
            _ai_call("execute_shell_command", {"cwd": "/tmp", "command": "ls"}, "2"),
        ]
        _to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(blocked) == 1

    def test_different_arguments_still_run(self):
        msgs = [
            HumanMessage(content="go"),
            _ai_call("read_scratch", {"scratch_id": "d1", "offset": 0}, "1"),
            ToolMessage(content="page 1", tool_call_id="1"),
            _ai_call("read_scratch", {"scratch_id": "d1", "offset": 1500}, "2"),
        ]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(to_run) == 1 and blocked == []

    def test_polling_tools_are_exempt(self):
        # Asking a background task for its status again is progress, not a loop.
        msgs = [
            HumanMessage(content="go"),
            _ai_call("task_status", {"task_id": "t1"}, "1"),
            ToolMessage(content="running", tool_call_id="1"),
            _ai_call("task_status", {"task_id": "t1"}, "2"),
        ]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(to_run) == 1 and blocked == []

    def test_a_new_user_message_forgives_earlier_calls(self):
        call = {"pattern": "x"}
        msgs = [
            HumanMessage(content="go"),
            _ai_call("find_in_files", call, "1"),
            ToolMessage(content="none", tool_call_id="1"),
            HumanMessage(content="try again"),
            _ai_call("find_in_files", call, "2"),
        ]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(to_run) == 1 and blocked == []

    def test_a_nudge_does_not_forgive_earlier_calls(self):
        # Nudges are HumanMessages; treating one as a fresh turn would hand the
        # loop a clean slate every time the guardrails fire.
        call = {"pattern": "x"}
        msgs = [
            HumanMessage(content="go"),
            _ai_call("find_in_files", call, "1"),
            ToolMessage(content="none", tool_call_id="1"),
            _nudge(),
            _ai_call("find_in_files", call, "2"),
        ]
        _to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(blocked) == 1

    def test_a_duplicate_within_one_message_is_blocked_once(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "git_diff", "args": {"file_path": "."}, "id": "1"},
            {"name": "git_diff", "args": {"file_path": "."}, "id": "2"},
        ])
        to_run, blocked = routing.split_repeated_calls([HumanMessage(content="go"), ai])
        assert len(to_run) == 1 and len(blocked) == 1

    def test_mixed_batch_runs_the_new_call_only(self):
        old = {"pattern": "x"}
        ai = AIMessage(content="", tool_calls=[
            {"name": "find_in_files", "args": old, "id": "3"},
            {"name": "glob_list", "args": {"pattern": "*.py"}, "id": "4"},
        ])
        msgs = [
            HumanMessage(content="go"),
            _ai_call("find_in_files", old, "1"),
            ToolMessage(content="none", tool_call_id="1"),
            ai,
        ]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert [c["name"] for c in to_run] == ["glob_list"]
        assert [m.tool_call_id for m in blocked] == ["3"]

    def test_unserialisable_arguments_do_not_raise(self):
        ai = AIMessage(content="", tool_calls=[
            {"name": "batch_patch", "args": {"patches": [{"f": object()}]}, "id": "1"},
        ])
        to_run, blocked = routing.split_repeated_calls([HumanMessage(content="go"), ai])
        assert len(to_run) == 1 and blocked == []

    def test_disabled_guard_runs_everything(self, monkeypatch):
        monkeypatch.setattr(routing, "STAGNATION_GUARD", False)
        call = {"pattern": "x"}
        msgs = [
            HumanMessage(content="go"),
            _ai_call("find_in_files", call, "1"),
            ToolMessage(content="none", tool_call_id="1"),
            _ai_call("find_in_files", call, "2"),
        ]
        to_run, blocked = routing.split_repeated_calls(msgs)
        assert len(to_run) == 1 and blocked == []

    def test_a_message_without_tool_calls_is_a_no_op(self):
        msgs = [HumanMessage(content="go"), AIMessage(content="done")]
        assert routing.split_repeated_calls(msgs) == ([], [])


class TestDuplicateAnswerGuardB33:
    """B3.3: enhanced duplicate-answer root-cause diagnostics."""

    def test_guard_logs_prior_answer_for_root_cause(self, caplog):
        """When the duplicate-answer guard fires, the log must carry the prior
        answer's content so the root cause can be diagnosed."""
        import logging
        caplog.set_level(logging.WARNING)
        state = _state(
            HumanMessage(content="hi"),
            AIMessage(content="The project is at ~/code/myapp."),
            AIMessage(content="The project lives at ~/code/myapp, as mentioned."),
        )
        assert routing.route_agent(state) == "distill"
        assert "duplicate final answer" in caplog.text
        assert "The project is at" in caplog.text
        assert "prior answer preview" in caplog.text

    def test_guard_with_interleaved_tool_calls_still_blocks_second_answer(self):
        """A no-tool-call answer after tool use, followed by another, is still a
        duplicate — the guard counts only no-tool-call answers."""
        state = _state(
            HumanMessage(content="hi"),
            AIMessage(content="", tool_calls=[
                {"name": "recall", "args": {"query": "q"}, "id": "1"},
            ]),
            ToolMessage(content="result", tool_call_id="1", name="recall"),
            AIMessage(content="here is what I found"),
            AIMessage(content="as I said, here is what I found"),
        )
        assert routing.route_agent(state) == "distill"

    def test_log_includes_message_count(self, caplog):
        """The log should carry the message count to help determine if the
        duplication is a graph re-entry or a server double-completion."""
        import logging
        caplog.set_level(logging.WARNING)
        state = _state(
            HumanMessage(content="hi"),
            AIMessage(content="first response"),
            AIMessage(content="second response"),
        )
        routing.route_agent(state)
        assert "messages total" in caplog.text
