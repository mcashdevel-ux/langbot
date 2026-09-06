"""Track C tests — stuck detection over the current turn's message list.

Ported from neo's ``neo/tests/test_stuck.py`` (adapted from journaled events to
langchain messages per docs/neo-port-plan.md Track C):

  - pattern 1: identical (tool, args) → identical result, N times
  - pattern 2: same tool erroring N times consecutively
  - pattern 3: N consecutive content-only llm_responses
  - pattern 4: A-B-A-B alternation (nudge)
  - pattern  ́5: soft repetition with drifting literals (nudge)
  - non-stuck cases (varied calls, recovered errors, mixed responses)
"""

from langchain_core.messages import AIMessage, ToolMessage

from components.stuck import StuckDetector


def ai(content="", calls=None):
    """AIMessage helper: tool_calls only when non-empty (pydantic rejects None)."""
    kw = {}
    if calls:
        kw["tool_calls"] = calls
    return AIMessage(content=content, **kw)


def tool(name, content, cid="c1"):
    return ToolMessage(content=content, name=name, tool_call_id=cid)


def shell_call(command, cid):
    return {"name": "shell", "args": {"command": command}, "id": cid}


# ── pattern 1: repeating action ─────────────────────────────────────────────

def test_repeating_action_detected():
    msgs = []
    for i in range(3):
        msgs += [ai(calls=[shell_call("ls", f"c{i}")]),
                   tool("shell", "same output", f"c{i}")]
    v = StuckDetector().check(msgs)
    assert v is not None and v.pattern == "repeating_action"
    assert v.severity == "halt"


def test_same_call_different_results_not_stuck():
    msgs = []
    for i in range(3):
        msgs += [ai(calls=[shell_call("date", f"c{i}")]),
                   tool("shell", f"output {i}", f"c{i}")]
    assert StuckDetector().check(msgs) is None


def test_different_args_not_stuck():
    msgs = []
    for i in range(3):
        msgs += [ai(calls=[shell_call(f"ls dir{i}", f"c{i}")]),
                   tool("shell", "same", f"c{i}")]
    assert StuckDetector().check(msgs) is None


def test_below_threshold_not_stuck():
    msgs = [ai(calls=[shell_call("ls", "c1")]), tool("shell", "same", "c1"),
               ai(calls=[shell_call("ls", "c2")]), tool("shell", "same", "c2")]
    assert StuckDetector().check(msgs) is None


# ── pattern 2: repeating error ──────────────────────────────────────────────

def test_repeating_error_detected():
    msgs = [tool("shell", "[tool error] boom", f"c{i}") for i in range(3)]
    v = StuckDetector().check(msgs)
    assert v is not None and v.pattern == "repeating_error"
    assert v.severity == "halt"


def test_mixed_tools_errors_not_stuck():
    msgs = [tool("shell", "[tool error] a", "c1"),
               tool("python_run", "[tool error] b", "c2"),
               tool("shell", "[tool error] c", "c3")]
    assert StuckDetector().check(msgs) is None


def test_error_recovery_not_stuck():
    msgs = [tool("shell", "[tool error] a", "c1"),
               tool("shell", "[tool error] b", "c2"),
               tool("shell", "worked", "c3")]
    assert StuckDetector().check(msgs) is None


# ── pattern 3: monologue ────────────────────────────────────────────────────

def test_monologue_detected():
    msgs = [ai(content=f"thinking {i}...") for i in range(4)]
    v = StuckDetector().check(msgs)
    assert v is not None and v.pattern == "monologue"
    assert v.severity == "halt"


def test_tool_calls_break_monologue():
    msgs = [ai(content="a"), ai(content="b"),
               ai(content="c", calls=[shell_call("ls", "x")]),
               ai(content="d")]
    assert StuckDetector().check(msgs) is None


# ── pattern 4: alternation (nudge) ────────────────────────────────────────

def test_alternating_detected():
    msgs = []
    for i in range(2):
        msgs += [ai(calls=[shell_call("pkill x", f"a{i}")]),
                   tool("shell", "ok", f"a{i}"),
                   ai(calls=[shell_call("start y", f"b{i}")]),
                   tool("shell", "ok", f"b{i}")]
    v = StuckDetector().check(msgs)
    assert v is not None and v.pattern == "alternating"
    assert v.severity == "nudge"


def test_polling_excluded_from_alternation():
    # sleep/wait commands repeat legitimately — must not count as flailing
    msgs = []
    for i in range(2):
        msgs += [ai(calls=[shell_call("sleep 1", f"a{i}")]),
                   tool("shell", "ok", f"a{i}"),
                   ai(calls=[shell_call("start y", f"b{i}")]),
                   tool("shell", "ok", f"b{i}")]
    assert StuckDetector().check(msgs) is None


# ── pattern 5: soft repetition (nudge) ────────────────────────────────────

def test_soft_repeating_detected():
    # Same command with drifting seq numbers (4+ digits) — exact-match blind,
    # but token-overlap similarity fires.

    msgs = []
    for i in range(4):
        msgs += [ai(calls=[shell_call(f"adb logcat --pid {1000 + i}", f"c{i}")]),
                   tool("shell", f"output line {i}", f"c{i}")]
    v = StuckDetector().check(msgs)
    assert v is not None and v.pattern == "soft_repeating"
    assert v.severity == "nudge"


def test_soft_repeating_timestamp_drift():
    msgs = []
    for i in range(4):
        msgs += [ai(calls=[shell_call(f"logcat -T \"12:{10 + i}:01\"", f"c{i}")]),
                   tool("shell", f"line {i}", f"c{i}")]
    v = StuckDetector().check(msgs)
    assert v is not None and v.pattern == "soft_repeating"


# ── non-stuck ─────────────────────────────────────────────────────────────────────

def test_varied_turn_not_stuck():
    msgs = []
    for i in range(3):
        msgs += [ai(calls=[shell_call(f"ls dir{i}", f"c{i}")]),
                   tool("shell", "out", f"c{i}")]
    assert StuckDetector().check(msgs) is None


def test_mixed_turn_not_stuck():
    # varied calls, recovered errors, and content-only responses interleaved
    msgs = [ai(calls=[shell_call("ls", "c1")]), tool("shell", "out", "c1"),
               ai(content="thinking..."),
               ai(calls=[shell_call("date", "c2")]), tool("shell", "[tool error] x", "c2"),
               ai(calls=[shell_call("pwd", "c3")]), tool("shell", "out", "c3"),
               ai(content="done")]
    assert StuckDetector().check(msgs) is None
