"""Agent-loop routing and autonomy guardrails.

Pure message-list logic, deliberately free of langgraph/LLM imports so it can be
unit-tested against synthetic message lists:

- ``route_agent``  — decides whether the graph goes to tools, a nudge, or distill.
- ``nudge_agent``  — builds the corrective system message for a detected failure.
- failure-mode phrase/pattern tables used by both.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

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
MAX_NUDGES_PER_TURN = config.get("routing.max_nudges_per_turn", 5)

NUDGE_PERMISSION = (
    "[AUTONOMOUS AGENT DIRECTIVE]: You just asked for permission or confirmation instead of acting. "
    "Do NOT ask the user whether to proceed. "
    "Invoke the required tool calls RIGHT NOW to complete the user request."
)

NUDGE_CODE_BLOCK = (
    "[AUTONOMOUS AGENT DIRECTIVE]: You wrote code or curl commands in a text block instead of "
    "calling your native tools. You have tools available — search_web, fetch_url, "
    "execute_shell_command, vault, etc. DO NOT write Python or bash blocks that pretend to use "
    "these tools. Call them DIRECTLY via the function-calling interface RIGHT NOW."
)


def ai_turns_since_human(messages) -> int:
    """Count consecutive AI messages back to (not including) the last HumanMessage."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "type", None) == "ai":
            count += 1
    return count


def final_answers_since_human(messages) -> int:
    """Count no-tool-call AI messages back to (not including) the last HumanMessage."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "type", None) == "ai" and not (getattr(m, "tool_calls", None) or []):
            count += 1
    return count


def nudge_agent(state):
    """Inject a targeted correction when the model either asked for permission or
    hallucinated tool calls as code blocks instead of invoking them."""
    last_msg = state["messages"][-1]
    content_lower = (getattr(last_msg, "content", "") or "").lower()
    if any(pat in content_lower for pat in TOOL_AVOIDANCE_PATTERNS):
        nudge_text = NUDGE_CODE_BLOCK
    else:
        nudge_text = NUDGE_PERMISSION
    return {"messages": [SystemMessage(content=nudge_text)]}


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
        if needs_nudge and ai_turns_since_human(messages) < MAX_NUDGES_PER_TURN:
            return "nudge"

    return "distill"
