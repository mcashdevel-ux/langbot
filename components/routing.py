"""Agent-loop routing and autonomy guardrails.

Pure message-list logic, deliberately free of langgraph/LLM imports so it can be
unit-tested against synthetic message lists:

- ``route_agent``  — decides whether the graph goes to tools, a nudge, or distill.
- ``nudge_agent``  — builds the corrective message for a detected failure.
- ``split_repeated_calls`` — stops a tool call the model already made this turn from
  running again (see the stagnation guard below).
- failure-mode phrase/pattern tables used by both.

A nudge is delivered as a ``HumanMessage``, not a ``SystemMessage``: many served
chat templates reject a system message that is not the first in the list (llama.cpp
returns a 500 with "System message must be at the beginning"), and a nudge lands
mid-conversation by definition. It carries ``NUDGE_MARKER`` so the counters below
can still tell nudges apart from what the user actually typed.
"""

import json
import logging

from langchain_core.messages import HumanMessage, ToolMessage

from .config import config

logger = logging.getLogger(__name__)

# Phrases that indicate the model is asking for permission instead of acting.
PERMISSION_PHRASES = (
    "would you like me to",
    "would you like to proceed",
    "should i proceed",
    "shall i proceed",
    "do you want me to",
    "can i proceed",
    "let me know if you would like",
    "if you would like me to",
    "would you like to",
    "would you like me",
    "please confirm",
    "please let me know",
    "do you want to proceed",
)

# Patterns that indicate the model is hallucinating tool calls as code blocks
# instead of invoking the actual function-calling interface.
TOOL_AVOIDANCE_PATTERNS = (
    "import search_web",
    "import fetch_url",
    "import execute_shell",
    "search_web.search(",
    "fetch_url(",
    "requests.get(",          # writing raw HTTP calls instead of using tools
    "requests.post(",
    "subprocess.run(",         # using shell inside a code block instead of the tool
    "```python\nimport",       # code fence opening with an import
    "```bash\ncurl",           # writing curl in a bash block instead of the tool
    "```\ncurl",
    "curl -h ",
    "curl -o ",
    "{{vault_get",            # hallucinated template syntax
    "{{vault",
)

# Nudge re-tries allowed per human turn before the loop gives up and finalizes.
# Each one costs a wasted generation plus a re-send of the whole thread, so on a
# small local model a generous budget is spent on context, not on recovery: a
# model that ignored two corrections will ignore the third.
MAX_NUDGES_PER_TURN = config.get("routing.max_nudges_per_turn", 3)

# Marker every nudge carries, so the budget counts nudges rather than AI turns.
NUDGE_MARKER = "[AUTONOMOUS AGENT DIRECTIVE]"

# Graph super-steps allowed per human turn. One tool round costs two steps
# (agent -> tools -> agent), so 100 leaves room for ~48 tool rounds before
# langgraph raises GraphRecursionError; its own default of 25 caps a turn at
# roughly a dozen tool calls, which a research or build task exhausts easily.
RECURSION_LIMIT = config.get("routing.recursion_limit", 100)

NUDGE_PERMISSION = (
    f"{NUDGE_MARKER}: Do not ask whether to proceed. Make the tool calls now."
)

NUDGE_CODE_BLOCK = (
    f"{NUDGE_MARKER}: That was a code block imitating a tool, not a tool call. "
    "Call the tool through the function-calling interface now."
)

# Stagnation guard: a small model that has lost the thread often re-issues a call it
# already made, verbatim, and each repeat costs a full re-send of the window plus
# whatever the tool does (a shell command, a fetch). Repeats are answered from the
# transcript instead of executed.
#
# Some tools are *meant* to be called with identical arguments repeatedly, because
# their answer changes over time: polling a background task is progress, not a loop.
STAGNATION_GUARD = config.get("routing.stagnation_guard", True)
STAGNATION_EXEMPT_TOOLS = set(config.get(
    "routing.stagnation_exempt_tools",
    ["task_status", "task_output", "task_list"],
))

REPEATED_CALL_NOTICE = (
    "This exact call was already made in this turn and was not run again. Its result "
    "is already above — use it, call with different arguments, or answer the user."
)


def is_nudge(message) -> bool:
    """True for a message this module injected, rather than one the user sent."""
    content = getattr(message, "content", "")
    return isinstance(content, str) and content.startswith(NUDGE_MARKER)


def _turn_start(message) -> bool:
    """True where the current turn begins: a real user message, nudges excluded."""
    return isinstance(message, HumanMessage) and not is_nudge(message)


def ai_turns_since_human(messages) -> int:
    """Count consecutive AI messages back to (not including) the last user message."""
    count = 0
    for m in reversed(messages):
        if _turn_start(m):
            break
        if getattr(m, "type", None) == "ai":
            count += 1
    return count


def nudges_since_human(messages) -> int:
    """Count nudges already issued in the current turn.

    Counting AI turns instead would spend the budget on ordinary tool rounds, so
    a turn with several tool calls would silently lose its guardrails.
    """
    count = 0
    for m in reversed(messages):
        if _turn_start(m):
            break
        if is_nudge(m):
            count += 1
    return count


def final_answers_since_human(messages) -> int:
    """Count no-tool-call AI messages back to (not including) the last user message."""
    count = 0
    for m in reversed(messages):
        if _turn_start(m):
            break
        if getattr(m, "type", None) == "ai" and not (getattr(m, "tool_calls", None) or []):
            count += 1
    return count


def _call_signature(call) -> str:
    """Identity of a tool call: its name plus its arguments, key order ignored."""
    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
    args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
    try:
        rendered = json.dumps(args, sort_keys=True, default=str)
    except (TypeError, ValueError):        # unhashable/unserialisable args: not a repeat
        rendered = repr(args)
    return f"{name}({rendered})"


def _calls_this_turn(messages) -> set:
    """Signatures of every tool call made since the last real user message."""
    seen = set()
    for m in reversed(messages):
        if _turn_start(m):
            break
        for call in getattr(m, "tool_calls", None) or []:
            seen.add(_call_signature(call))
    return seen


def split_repeated_calls(messages) -> tuple:
    """Split the last AI message's tool calls into those to run and canned replies.

    Returns ``(calls_to_run, tool_messages)``. Every blocked call still gets a
    ``ToolMessage`` carrying its ``tool_call_id``: an assistant message whose calls
    are left unanswered is invalid for the next request, so refusing to answer one
    would break the session rather than the loop.
    """
    last_msg = messages[-1]
    calls = list(getattr(last_msg, "tool_calls", None) or [])
    if not STAGNATION_GUARD or not calls:
        return calls, []

    earlier = _calls_this_turn(messages[:-1])
    to_run, blocked, seen_here = [], [], set()
    for call in calls:
        name = call.get("name")
        signature = _call_signature(call)
        # A duplicate inside one message is a repeat too, exempt tools aside.
        repeat = signature in earlier or signature in seen_here
        seen_here.add(signature)
        if repeat and name not in STAGNATION_EXEMPT_TOOLS:
            logger.warning("stagnation guard: refused a repeated call to %s", signature)
            blocked.append(ToolMessage(
                content=REPEATED_CALL_NOTICE,
                name=name,
                tool_call_id=call.get("id") or "",
                status="error",
            ))
        else:
            to_run.append(call)
    return to_run, blocked


def nudge_agent(state):
    """Inject a targeted correction when the model either asked for permission or
    hallucinated tool calls as code blocks instead of invoking them."""
    last_msg = state["messages"][-1]
    content_lower = (getattr(last_msg, "content", "") or "").lower()
    if any(pat in content_lower for pat in TOOL_AVOIDANCE_PATTERNS):
        nudge_text = NUDGE_CODE_BLOCK
    else:
        nudge_text = NUDGE_PERMISSION
    return {"messages": [HumanMessage(content=nudge_text)]}


def route_agent(state):
    """Route the agent node's output to ``tools``, ``nudge``, or ``distill``.

    At most one final (no-tool-call) AI message per human turn is allowed to
    reach ``distill``/the user: a second one is a duplicate answer and is routed
    straight to ``distill`` without a further nudge round-trip.
    """
    messages = state["messages"]
    last_msg = messages[-1]
    if last_msg.type == "ai" and getattr(last_msg, "tool_calls", None):
        return "tools"

    if last_msg.type == "ai" and isinstance(getattr(last_msg, "content", None), str):
        if final_answers_since_human(messages[:-1]) > 0:
            logger.warning(
                "route_agent: dropping a duplicate final answer this turn "
                "(content preview: %r)", last_msg.content[:120]
            )
            return "distill"

        content_lower = last_msg.content.lower()
        needs_nudge = (
            any(phrase in content_lower for phrase in PERMISSION_PHRASES)
            or any(pat in content_lower for pat in TOOL_AVOIDANCE_PATTERNS)
        )
        if needs_nudge and nudges_since_human(messages) < MAX_NUDGES_PER_TURN:
            return "nudge"

    return "distill"
