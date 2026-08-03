"""Unit tests for routing.py — agent routing, nudges, duplicate-answer guard."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import components.routing as routing


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
        msgs += [SystemMessage(content=routing.NUDGE_PERMISSION)
                 for _ in range(routing.MAX_NUDGES_PER_TURN)]
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
            SystemMessage(content="nudge"),
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
    def test_code_block_failure_gets_code_block_nudge(self):
        state = _state(AIMessage(content="```bash\ncurl https://x\n```"))
        out = routing.nudge_agent(state)["messages"][0]
        assert out.content == routing.NUDGE_CODE_BLOCK

    def test_permission_failure_gets_permission_nudge(self):
        state = _state(AIMessage(content="Should I proceed?"))
        out = routing.nudge_agent(state)["messages"][0]
        assert out.content == routing.NUDGE_PERMISSION
