"""Recover tool calls from models that emit them as text instead of natively.

Small local models (and LoRA fine-tunes of them) frequently ignore the OpenAI
function-calling channel and instead print the call they *would* have made into
the assistant content, e.g.

    {"content": "Analyzing the current working directory.",
     "tool_calls": [{"name": "glob_list", "args": {"pattern": "."}}]}

To langbot that is a no-tool-call AI message, so it renders as the final answer
and nothing ever executes — the agent looks like it is narrating instead of
acting. This module parses such content back into real tool calls.

Deliberately conservative, since the alternative failure mode (hijacking a
legitimate answer that merely *discusses* JSON) is worse than missing a repair:

- only runs on messages that have no native ``tool_calls``
- a parsed call is kept only if its name is a genuinely registered tool
- args must be an object; a call whose args cannot be parsed is dropped

Pure and dependency-free (``json``/``re``/``uuid``) apart from ``.config``;
``repair_message`` touches only ``.content``/``.tool_calls`` by duck typing, so
the module stays independent of LangChain's message classes.
"""

import json
import logging
import re
import uuid

from .config import config

logger = logging.getLogger(__name__)

# Master switch: set "compat.repair_json_tool_calls" to false to render such
# content verbatim instead (useful when debugging what a model actually emits).
REPAIR_ENABLED = config.get("compat.repair_json_tool_calls", True)
# Cap on JSON candidates examined per message — bounds the scan on long answers.
MAX_CANDIDATES = config.get("compat.repair_max_candidates", 20)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?|\n?```$")

# Keys a chat-envelope object may carry while still being *only* an envelope.
# Any other key means the object is real data the caller wants verbatim.
_ENVELOPE_KEYS = frozenset({
    "content", "tool_calls", "tool_call", "thought", "thoughts", "reasoning",
})


def _strip_fences(text: str) -> str:
    """Drop one leading/trailing Markdown fence, keeping the inner text."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    return _FENCE_RE.sub("", stripped).strip()


def _json_objects(text: str):
    """Yield (start, end, obj) for balanced top-level ``{...}`` spans in ``text``.

    A plain ``json.loads`` fails as soon as there is prose around the object, and
    a regex cannot match nested braces, so this walks the string tracking depth
    (and string/escape state, so a ``}`` inside a JSON string does not end it).
    """
    depth = 0
    start = -1
    in_string = False
    escaped = False
    found = 0
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = text[start:i + 1]
                try:
                    obj = json.loads(chunk)
                except (ValueError, TypeError):
                    obj = None
                if isinstance(obj, dict):
                    yield start, i + 1, obj
                    found += 1
                    if found >= MAX_CANDIDATES:
                        return


def _normalize_call(entry, valid_names) -> "dict | None":
    """Turn one textual call entry into a LangChain-shaped tool_call dict."""
    if not isinstance(entry, dict):
        return None
    function = entry.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        raw_args = function.get("arguments")
    else:
        name = entry.get("name") or entry.get("tool") or entry.get("tool_name")
        raw_args = entry.get("args")
        if raw_args is None:
            raw_args = entry.get("arguments")
        if raw_args is None:
            raw_args = entry.get("parameters")
    if not isinstance(name, str) or name not in valid_names:
        return None
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args) if raw_args.strip() else {}
        except (ValueError, TypeError):
            logger.warning("tool_call_repair: unparseable args for %r, dropping", name)
            return None
    if raw_args is None:
        raw_args = {}
    if not isinstance(raw_args, dict):
        return None
    return {
        "name": name,
        "args": raw_args,
        "id": entry.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        "type": "tool_call",
    }


def _calls_from_object(obj, valid_names) -> "list[dict]":
    """Extract calls from one parsed object, or [] if it isn't a call payload."""
    raw = obj.get("tool_calls")
    if raw is None:
        raw = obj.get("tool_call")
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        calls = [c for c in (_normalize_call(e, valid_names) for e in raw) if c]
        return calls
    # A bare {"name": ..., "args": {...}} object, with no wrapper.
    single = _normalize_call(obj, valid_names)
    return [single] if single else []


def parse_tool_calls(content: str, valid_names) -> "tuple[str, list[dict]]":
    """Extract tool calls embedded in ``content``.

    Returns ``(remaining_content, calls)``. ``calls`` is empty when nothing
    recognizable is found, in which case ``remaining_content`` is ``content``
    unchanged. Otherwise ``remaining_content`` is whatever prose accompanied the
    call (the payload's own ``"content"`` field, or the surrounding text).
    """
    if not content or not isinstance(content, str):
        return content, []
    valid_names = set(valid_names)

    text = _strip_fences(content)
    for start, end, obj in _json_objects(text):
        calls = _calls_from_object(obj, valid_names)
        if not calls:
            continue
        if "content" in obj and isinstance(obj["content"], str):
            remaining = obj["content"].strip()
        else:
            remaining = (text[:start] + text[end:]).strip()
        return remaining, calls
    return content, []


def unwrap_content(text) -> "str | None":
    """Unwrap a call-less chat envelope, returning its inner ``content``.

    The same models that emit textual tool calls also wrap plain answers, e.g.
    ``{"content": "The directory holds 12 files.", "tool_calls": []}``, which
    would otherwise be rendered to the user as raw JSON.

    Returns None (leave the text alone) unless the whole text is a single JSON
    object whose keys are all envelope keys, whose ``content`` is a string, and
    which carries no tool calls — calls mean ``parse_tool_calls`` should handle
    it instead, and a stray key means the object is real data, not an envelope.
    """
    if not text or not isinstance(text, str):
        return None
    stripped = _strip_fences(text)
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict) or not set(obj) <= _ENVELOPE_KEYS:
        return None
    if obj.get("tool_calls") or obj.get("tool_call"):
        return None
    inner = obj.get("content")
    if not isinstance(inner, str):
        return None
    return inner.strip()


def repair_message(message, valid_names) -> bool:
    """Rewrite ``message`` in place if its content hides tool calls.

    Returns True when a repair happened. No-ops when the message already has
    native tool calls, when repair is disabled, or when nothing parses.
    """
    if not REPAIR_ENABLED:
        return False
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return False
    remaining, calls = parse_tool_calls(content, valid_names)
    if not calls:
        # No calls, but the answer itself may still be wrapped in an envelope.
        inner = unwrap_content(content)
        if inner is None:
            return False
        logger.debug("tool_call_repair: unwrapped a call-less JSON envelope")
        message.content = inner
        return True
    logger.warning(
        "tool_call_repair: model emitted %d tool call(s) as text instead of "
        "native tool calls (%s) — recovered",
        len(calls), ", ".join(c["name"] for c in calls),
    )
    message.content = remaining
    message.tool_calls = calls
    return True
