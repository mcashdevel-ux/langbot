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

# Qwen's own chat template documents calls as <tool_call>{"name":…,"arguments":…}
# </tool_call>, which a server that does not parse the tags passes through as
# content. An unclosed tag (truncated generation) still yields the JSON.
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.DOTALL)
# Chat-template markup that leaks into content when the server does not strip it.
_MARKUP_RE = re.compile(
    r"</?tool_call>|</?tool_response>|<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>"
)
_BLANK_RUN_RE = re.compile(r"\n{3,}")
# Qwen3 reasoning blocks. Unclosed means the generation was cut off mid-thought,
# so everything from the tag on is reasoning.
_THINK_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)
_OPEN_THINK_RE = re.compile(r"<think(?:ing)?>.*\Z", re.DOTALL)


def _clean_markup(text: str) -> str:
    """Remove leaked chat-template markup, keeping the prose intact."""
    return _BLANK_RUN_RE.sub("\n\n", _MARKUP_RE.sub("", text)).strip()


def strip_reasoning(text: str) -> str:
    """Drop ``<think>`` blocks and leaked markup.

    For consumers that need the model's *payload* rather than its narration —
    the knowledge distiller parses a JSON array, and reasoning text around it
    (often containing brackets of its own) derails that.
    """
    if not isinstance(text, str):
        return text
    without = _THINK_RE.sub("", text)
    if "<think" in without:
        without = _OPEN_THINK_RE.sub("", without)
    return _clean_markup(without)


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


def _rename_args(name: str, args: dict, aliases) -> dict:
    """Map known wrong-but-obvious argument names onto the real parameters.

    Weak models get parameter *names* wrong as readily as the call protocol
    (``recall(memory_id=...)`` instead of ``recall(query=...)``), and an unknown
    keyword fails the call outright. Only aliases the caller declares are
    renamed, and never over an argument the model already got right.
    """
    table = (aliases or {}).get(name)
    if not table:
        return args
    renamed = {}
    for key, value in args.items():
        target = table.get(key, key)
        if target != key and target in args:
            continue                       # the correct name is already present
        renamed[target] = value
    if renamed != args:
        logger.debug("tool_call_repair: renamed %s args %s -> %s",
                     name, sorted(args), sorted(renamed))
    return renamed


def _normalize_call(entry, valid_names, aliases=None) -> "dict | None":
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
        "args": _rename_args(name, raw_args, aliases),
        "id": entry.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        "type": "tool_call",
    }


def _calls_from_object(obj, valid_names, aliases=None) -> "list[dict]":
    """Extract calls from one parsed object, or [] if it isn't a call payload."""
    raw = obj.get("tool_calls")
    if raw is None:
        raw = obj.get("tool_call")
    if isinstance(raw, dict):
        raw = [raw]
    if isinstance(raw, list):
        calls = [c for c in (_normalize_call(e, valid_names, aliases) for e in raw) if c]
        return calls
    # A bare {"name": ..., "args": {...}} object, with no wrapper.
    single = _normalize_call(obj, valid_names, aliases)
    return [single] if single else []


def parse_tool_calls(content: str, valid_names, arg_aliases=None) -> "tuple[str, list[dict]]":
    """Extract tool calls embedded in ``content``.

    Returns ``(remaining_content, calls)``. ``calls`` is empty when nothing
    recognizable is found, in which case ``remaining_content`` is ``content``
    unchanged. Otherwise ``remaining_content`` is whatever prose accompanied the
    call (the payload's own ``"content"`` field, or the surrounding text).
    """
    if not content or not isinstance(content, str):
        return content, []
    valid_names = set(valid_names)

    # Qwen-style <tool_call>{...}</tool_call> blocks first: the tags are that
    # family's native (text) protocol, so the JSON inside is unambiguous.
    xml_calls = []
    remaining_xml = content
    for match in _TOOL_CALL_TAG_RE.finditer(content):
        for _, _, obj in _json_objects(match.group(1)):
            xml_calls.extend(_calls_from_object(obj, valid_names, arg_aliases))
        remaining_xml = remaining_xml.replace(match.group(0), " ")
    if xml_calls:
        return _clean_markup(remaining_xml), xml_calls

    text = _strip_fences(content)
    for start, end, obj in _json_objects(text):
        calls = _calls_from_object(obj, valid_names, arg_aliases)
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


def repair_message(message, valid_names, arg_aliases=None) -> bool:
    """Rewrite ``message`` in place if its content hides tool calls or markup.

    Returns True when the message was changed. No-ops when it already has native
    tool calls, when repair is disabled, or when the content is clean prose.
    """
    if not REPAIR_ENABLED:
        return False
    if getattr(message, "tool_calls", None):
        return False
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return False

    remaining, calls = parse_tool_calls(content, valid_names, arg_aliases)
    if calls:
        logger.warning(
            "tool_call_repair: model emitted %d tool call(s) as text instead of "
            "native tool calls (%s) — recovered",
            len(calls), ", ".join(c["name"] for c in calls),
        )
        message.content = _clean_markup(remaining)
        message.tool_calls = calls
        return True

    # No calls: the answer may still be wrapped in an envelope, or carry leaked
    # chat-template markup such as <tool_response>.
    inner = unwrap_content(content)
    cleaned = _clean_markup(inner if inner is not None else content)
    if cleaned == content:
        return False
    logger.debug("tool_call_repair: cleaned a call-less answer (envelope/markup)")
    message.content = cleaned
    return True
