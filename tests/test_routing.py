import pytest
from langchain_core.messages import AIMessage, HumanMessage
from components import routing


def test_routing_stats_basic():
    # Initial state
    st = routing.stats()
    assert "nudges_permission" in st
    assert "near_miss_permission_hedges" in st


def test_near_miss_hedges():
    # Strict phrase shouldn't be counted as a near-miss
    assert not routing._check_near_miss_hedges("Would you like me to proceed?")
    
    # Near-miss loose syntax should be counted
    assert routing._check_near_miss_hedges("Shall I continue?")
    assert routing._check_near_miss_hedges("Please tell me if I can start.")
    assert routing._check_near_miss_hedges("Do you wish me to proceed?")
    assert not routing._check_near_miss_hedges("Here is the final answer.")


def test_routing_nudge_counters():
    routing._STATS["nudges_permission"] = 0
    routing._STATS["nudges_code_block"] = 0

    state = {"messages": [AIMessage(content="Would you like me to run this?")]}
    routing.nudge_agent(state)
    assert routing._STATS["nudges_permission"] == 1
    assert routing._STATS["nudges_code_block"] == 0

    state = {"messages": [AIMessage(content="```bash\ncurl localhost\n```")]}
    routing.nudge_agent(state)
    assert routing._STATS["nudges_permission"] == 1
    assert routing._STATS["nudges_code_block"] == 1


def test_route_agent_near_miss_telemetry():
    routing._STATS["near_miss_permission_hedges"] = 0

    # Under route_agent, a final clean answer with near-miss language should trigger the telemetry increment
    state = {
        "messages": [
            HumanMessage(content="Hello"),
            AIMessage(content="Do you wish me to proceed?")
        ]
    }
    decision = routing.route_agent(state)
    assert decision == "distill"
    assert routing._STATS["near_miss_permission_hedges"] == 1
