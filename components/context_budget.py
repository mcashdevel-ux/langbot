"""Token budget for the graph's message history.

The checkpointer replays the whole thread on every step, so without a budget a
session grows until the server truncates it from the front — silently dropping
the system prompt — or rejects the request outright. Small local models make
this acute: a 9B model on a 32k window spends most of it on tool schemas and
tool output, not on conversation.

This module decides *what to keep*, not *how to summarize*: callers pass in a
``summarize`` callable (the same LLM, one cheap call) so the policy here stays
pure and unit-testable. Compaction keeps the most recent exchanges verbatim and
folds everything older into a rolling summary that the agent node prepends to
the prompt.

Budgets are measured in tokens, not messages: one grep result can outweigh
twenty turns of conversation. ``tiktoken`` is used when available (it ships with
langchain-openai) and a 4-chars-per-token approximation otherwise; both are
estimates for a local model's own tokenizer, so the threshold leaves headroom.
"""

import json
import logging

from .config import config

logger = logging.getLogger(__name__)

# Total context window of the served model.
BUDGET_TOKENS = config.get("context.budget_tokens", 32768)
# Held back for the system prompt, tool schemas, and the answer being generated.
RESERVE_TOKENS = config.get("context.reserve_tokens", 8192)
# Fraction of the usable budget that triggers compaction.
COMPACT_AT = config.get("context.compact_at", 0.7)
# Messages always kept verbatim (the tail of the conversation).
KEEP_LAST_MESSAGES = config.get("context.keep_last_messages", 12)
# Cap on the rolling summary itself, so it cannot grow into the problem it solves.
SUMMARY_MAX_CHARS = config.get("context.summary_max_chars", 1500)
CHARS_PER_TOKEN = config.get("context.chars_per_token", 4)

_encoding = None
_encoding_loaded = False


def _get_encoding():
    """Return a tiktoken encoding, or ``None`` to fall back to the estimate."""
    global _encoding, _encoding_loaded
    if not _encoding_loaded:
        _encoding_loaded = True
        try:
            import tiktoken

            _encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            logger.debug("context_budget: tiktoken unavailable, estimating tokens")
            _encoding = None
    return _encoding


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    encoding = _get_encoding()
    if encoding is not None:
        return len(encoding.encode(text, disallowed_special=()))
    return max(1, len(text) // CHARS_PER_TOKEN)


def message_tokens(msg) -> int:
    """Token cost of one message, including its tool calls and role overhead."""
    content = getattr(msg, "content", "")
    if not isinstance(content, str):
        content = str(content)
    total = estimate_tokens(content)
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        total += estimate_tokens(json.dumps(tool_calls, default=str))
    return total + 4  # role/formatting tokens the chat template adds


def total_tokens(messages) -> int:
    return sum(message_tokens(m) for m in messages)


def usable_budget() -> int:
    return max(1, BUDGET_TOKENS - RESERVE_TOKENS)


def compaction_threshold() -> int:
    return int(usable_budget() * COMPACT_AT)


def split_for_compaction(messages, keep_last: int = KEEP_LAST_MESSAGES):
    """Split ``messages`` into ``(older, recent)`` at a safe boundary.

    ``recent`` never starts with a tool message: a ToolMessage separated from
    the AI message that requested it is rejected by the chat API, so the split
    point walks backwards until the boundary sits before the whole tool round.
    """
    if keep_last <= 0 or len(messages) <= keep_last:
        return [], list(messages)
    cut = len(messages) - keep_last
    while cut > 0 and getattr(messages[cut], "type", None) == "tool":
        cut -= 1
    return list(messages[:cut]), list(messages[cut:])


def needs_compaction(messages, summary: str = "") -> bool:
    """True when the thread (plus its summary) has crossed the threshold."""
    return total_tokens(messages) + estimate_tokens(summary) > compaction_threshold()


def render_for_summary(messages) -> str:
    """Flatten messages into the transcript handed to the summarizer."""
    lines = []
    for msg in messages:
        role = getattr(msg, "type", "message")
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            content = str(content)
        if role == "tool":
            name = getattr(msg, "name", None) or "tool"
            lines.append(f"[tool {name}]: {content[:400]}")
            continue
        for call in getattr(msg, "tool_calls", None) or []:
            lines.append(f"[called {call.get('name', 'tool')}]")
        if content:
            lines.append(f"[{role}]: {content[:600]}")
    return "\n".join(lines)


def summary_prompt(messages, previous_summary: str = "") -> str:
    previous = (
        f"Summary so far:\n{previous_summary}\n\n" if previous_summary else ""
    )
    return (
        "Compress the earlier part of an agent session into notes the assistant "
        "can work from after the transcript is discarded.\n"
        "Keep: what the user asked for, decisions taken, paths, commands, ids, "
        "scratch ids, and anything still unfinished. Drop pleasantries and "
        "superseded attempts. Write at most one short paragraph, no preamble.\n\n"
        f"{previous}Transcript:\n{render_for_summary(messages)}\n/no_think"
    )


def compact(messages, summarize, previous_summary: str = "",
            keep_last: int = KEEP_LAST_MESSAGES):
    """Return ``(dropped, recent, summary)`` for a thread over budget.

    ``summarize`` takes a prompt and returns text; a failure there is not fatal
    (the turn proceeds uncompacted) because losing a turn to a summarizer error
    would be worse than being briefly over budget.
    """
    older, recent = split_for_compaction(messages, keep_last)
    if not older:
        return [], list(messages), previous_summary
    try:
        summary = summarize(summary_prompt(older, previous_summary)).strip()
    except Exception:
        logger.warning("context_budget: summarization failed, keeping history",
                       exc_info=True)
        return [], list(messages), previous_summary
    if not summary:
        return [], list(messages), previous_summary
    return older, recent, summary[:SUMMARY_MAX_CHARS]
