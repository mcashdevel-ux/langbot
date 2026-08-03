"""Which tools get bound to the model this turn.

Every bound tool costs its JSON schema in the prompt, on every step, for the
whole session. Worse than the tokens: a small model picks worse from a long
menu, so a twenty-tool list hurts accuracy exactly where the budget is
tightest. Core tools are always bound; the rest are bound only when the turn
looks like it needs them, or once they have already been used in it (a
multi-round task must keep the tool it started with).

Matching is deliberately keyword-based and generous: binding a tool that turns
out to be unnecessary costs a schema, while failing to bind one the model needs
costs the whole turn. Set ``tools.dynamic_binding`` to false to bind everything.
"""

import re

from .config import config

DYNAMIC_BINDING = config.get("tools.dynamic_binding", True)

# Always bound. ``recall`` is here despite being a memory tool: the system
# prompt requires a recall before any answer that depends on what we know, so
# leaving it unbound would silently disable that rule.
CORE_TOOLS = config.get("tools.core", [
    "execute_shell_command",
    "read_any_file",
    "write_any_file",
    "patch_file",
    "search_web",
    "recall",
])

_TASKS = (r"background|long[- ]running|server|daemon|watch|tail|"
          r"start .*(?:server|process)|\btask\b")

# Situational tools and the words that pull them in. Keys are tool names.
TRIGGERS = {
    "vault": r"vault|credential|secret|api[ _-]?key|token|password|passphrase|env var",
    "remember": r"remember|memoriz|keep in mind|note that|for future|i prefer|my name",
    "git_diff": r"\bgit\b|\bdiff\b|commit|staged|unstaged|working tree|repo",
    "batch_patch": r"batch|refactor|rename|across .*files|every file|all files",
    "fetch_url": r"https?://|www\.|\.com\b|\.org\b|url|link|web ?page|fetch|article|docs?\b",
    "find_in_files": r"grep|search|find|where is|which file|usage|references|occurrenc|\bTODO\b",
    "read_many_files": r"read .*files|several files|these files|whole (?:dir|folder|module)|overview",
    "glob_list": r"list .*files|\*\.|glob|directory|folder contents|tree\b|what files",
    "task_start": _TASKS,
    "task_list": _TASKS,
    "task_status": _TASKS,
    "task_output": _TASKS,
    "task_kill": _TASKS,
    "read_scratch": r"scratch:|scratch id|full (?:output|file|page|diff)|show me the rest|page (?:through|more)",
}

_COMPILED = {name: re.compile(pattern, re.I) for name, pattern in TRIGGERS.items()}


def _turn_text(messages) -> str:
    """Text the triggers are matched against: the current turn, or its tail.

    Tool results are included so a tool can pull in its natural follow-up — a
    ``search_web`` result contains urls, which is what binds ``fetch_url``.
    """
    last_human = -1
    for i, msg in enumerate(messages):
        if getattr(msg, "type", None) == "human":
            last_human = i
    window = messages[last_human:] if last_human >= 0 else messages[-4:]
    parts = []
    for msg in window:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and content:
            parts.append(content[:2000])
    return "\n".join(parts)


def _already_used(messages) -> "set[str]":
    """Tools called since the last human message."""
    last_human = -1
    for i, msg in enumerate(messages):
        if getattr(msg, "type", None) == "human":
            last_human = i
    used = set()
    for msg in messages[last_human + 1:] if last_human >= 0 else messages:
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name")
            if name:
                used.add(name)
    return used


def select_tool_names(messages) -> "set[str]":
    """Names of the tools to bind for this step."""
    selected = set(CORE_TOOLS) | _already_used(messages)
    text = _turn_text(messages)
    for name, pattern in _COMPILED.items():
        if pattern.search(text):
            selected.add(name)
    return selected


def select_tools(all_tools, messages) -> list:
    """Subset of ``all_tools`` to bind, preserving their declared order."""
    if not DYNAMIC_BINDING:
        return list(all_tools)
    names = select_tool_names(messages)
    chosen = [t for t in all_tools if t.name in names]
    # A model with no tools cannot act; fall back to the full set rather than
    # letting a misconfigured core list strand the agent.
    return chosen or list(all_tools)
