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

The counters at the bottom exist because two of this module's constants are
guesses that only measurement can settle: whether ``reserve_tokens`` (2000) is
still right now that tool schemas are bound per turn, and what compaction costs
the server's prompt cache. ``stats()`` is what ``/health`` reports.
"""

import json
import logging
import threading

from .config import config

logger = logging.getLogger(__name__)

# Total context window of the served model.
BUDGET_TOKENS = config.get("context.budget_tokens", 32768)
# Held back for the system prompt, tool schemas, and the answer being generated.
# Default 2000 covers the measured overhead: system prompt (~191 tokens) +
# per-turn tool schemas (~950 tokens) + headroom for the rolling summary.
RESERVE_TOKENS = config.get("context.reserve_tokens", 2000)
# Fraction of the usable budget that triggers compaction.
COMPACT_AT = config.get("context.compact_at", 0.7)
# Messages always kept verbatim (the tail of the conversation).
KEEP_LAST_MESSAGES = config.get("context.keep_last_messages", 12)
# Minimum token count of messages to keep verbatim, as an alternative (or floor)
# to message-count-based keeping.  When set above 0, split_for_compaction keeps
# at least this many tokens' worth of recent messages regardless of the message
# count.  Set to 0 to use only the message count.  Default 0 (disabled).
KEEP_LAST_TOKENS = config.get("context.keep_last_tokens", 0)
# Cap on the rolling summary itself, so it cannot grow into the problem it solves.
SUMMARY_MAX_CHARS = config.get("context.summary_max_chars", 1500)
CHARS_PER_TOKEN = config.get("context.chars_per_token", 4)
# Multiplier applied to every token estimate.  cl100k_base (OpenAI's tokenizer)
# consistently undercounts for Qwen and Llama tokenizers; set this to bridge the
# gap until the estimate matches what the server counts.  Example values from
# real sessions: Qwen-9B ~1.18, Llama-3-8B ~1.14, Qwen-32B ~1.22.
# Default 1.0 keeps the status-quo for OpenAI-compatible tokenizers.
TOKENIZER_SCALE = config.get("context.tokenizer_scale", 1.0)

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
        raw = len(encoding.encode(text, disallowed_special=()))
    else:
        raw = max(1, len(text) // CHARS_PER_TOKEN)
    if TOKENIZER_SCALE != 1.0:
        return max(1, int(raw * TOKENIZER_SCALE))
    return raw


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


def split_for_compaction(messages, keep_last: int = KEEP_LAST_MESSAGES,
                          keep_last_tokens: int = KEEP_LAST_TOKENS):
    """Split ``messages`` into ``(older, recent)`` at a safe boundary.

    ``recent`` never starts with a tool message: a ToolMessage separated from
    the AI message that requested it is rejected by the chat API, so the split
    point walks backwards until the boundary sits before the whole tool round.

    When ``keep_last_tokens > 0``, the split respects the larger of the
    message-count floor and the token-count floor -- a single enormous tool
    result will not silently eat the budget while being kept verbatim.
    """
    # Determine the cut point: message-count floor first.
    msg_count = max(0, keep_last)
    if keep_last_tokens > 0:
        # Walk from the tail, accumulating tokens until we have enough.
        token_count = 0
        i = len(messages) - 1
        token_cut = 0
        while i >= 0:
            if i >= 1 and getattr(messages[i], "type", None) == "tool":
                token_count += message_tokens(messages[i])
                token_count += message_tokens(messages[i - 1])
                i -= 2
            else:
                token_count += message_tokens(messages[i])
                i -= 1
            if token_count >= keep_last_tokens:
                token_cut = i + 1
                break
        # Use the larger of the two floors — respect both the message-count
        # and token-count minima.
        if msg_count > 0:
            cut = max(msg_count, len(messages) - token_cut)
        else:
            cut = len(messages) - token_cut
    else:
        cut = msg_count

    if cut <= 0 or len(messages) <= cut:
        return [], list(messages)
    cut = len(messages) - cut
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


def summary_prompt(messages, previous_summary: str = "", preserve_facts: str = "") -> str:
    previous = (
        f"Summary so far:\n{previous_summary}\n\n" if previous_summary else ""
    )
    preserve = ""
    if preserve_facts:
        preserve = (
            f"Facts carried forward from earlier compactions "
            f"(these MUST appear in your summary):\n{preserve_facts}\n\n"
        )
    return (
        "Compress the earlier part of an agent session into notes the assistant "
        "can work from after the transcript is discarded.\n"
        "Keep: what the user asked for, decisions taken, paths, commands, ids, "
        "scratch ids, and anything still unfinished. Drop pleasantries and "
        "superseded attempts. Write at most one short paragraph, no preamble.\n\n"
        f"{preserve}"
        f"{previous}Transcript:\n{render_for_summary(messages)}\n/no_think"
    )


def compact(messages, summarize, previous_summary: str = "",
            keep_last: int = KEEP_LAST_MESSAGES,
            keep_last_tokens: int = KEEP_LAST_TOKENS,
            preserve_facts: str = ""):
    """Return ``(dropped, recent, summary)`` for a thread over budget.

    ``summarize`` takes a prompt and returns text; a failure there is not fatal
    (the turn proceeds uncompacted) because losing a turn to a summarizer error
    would be worse than being briefly over budget.

    When ``preserve_facts`` is non-empty (recompaction), those facts are included
    in the summarization prompt as must-preserve material that a future compaction
    could otherwise lose.
    """
    older, recent = split_for_compaction(messages, keep_last, keep_last_tokens)
    if not older:
        return [], list(messages), previous_summary
    try:
        summary = summarize(
            summary_prompt(older, previous_summary, preserve_facts)
        ).strip()
    except Exception:
        logger.warning("context_budget: summarization failed, keeping history",
                       exc_info=True)
        return [], list(messages), previous_summary
    if not summary:
        return [], list(messages), previous_summary
    return older, recent, summary[:SUMMARY_MAX_CHARS]


# ---------------------------------------------------------------------------
# Instrumentation
# ---------------------------------------------------------------------------
# Prompt cost is split the way the budget is: what the reserve is meant to cover
# (the system prompt, the rolling summary, the bound tool schemas, and the answer
# still to be generated) versus the conversation it is protecting. `prefix` is the
# leading run of messages identical to the previous step's, i.e. what a server
# with prompt caching can reuse instead of reprocessing.
_stats = {
    "steps": 0,
    "peak_prompt": 0,
    "peak_history": 0,
    "peak_overhead": 0,      # system + summary + tool schemas: what the reserve covers
    "peak_schemas": 0,
    "prefix_reused": 0,      # tokens the server could keep, summed over steps
    "prefix_reprocessed": 0, # tokens it had to read again
    "compactions": 0,
    "tokens_dropped": 0,
    "thinking_tokens": 0,    # tokens spent inside <think> blocks (stripped for display)
    "thinking_calls": 0,     # steps where a <think> block was present
}
_stats_lock = threading.Lock()


def shared_prefix_tokens(previous, current) -> int:
    """Tokens of the leading messages identical in both prompts.

    Prompt caching is prefix-based, so the first difference invalidates everything
    after it: one changed leading message costs a full reprocess of the window.
    """
    total = 0
    for before, after in zip(previous, current):
        if before != after:
            break
        total += estimate_tokens(before)
    return total


def record_step(history_tokens: int, overhead_tokens: int, schema_tokens: int = 0,
                prefix_tokens: int = 0) -> None:
    """Record one agent step's prompt composition."""
    prompt = history_tokens + overhead_tokens
    global _last_prompt_tokens
    _last_prompt_tokens = prompt
    with _stats_lock:
        _stats["steps"] += 1
        _stats["peak_prompt"] = max(_stats["peak_prompt"], prompt)
        _stats["peak_history"] = max(_stats["peak_history"], history_tokens)
        _stats["peak_overhead"] = max(_stats["peak_overhead"], overhead_tokens)
        _stats["peak_schemas"] = max(_stats["peak_schemas"], schema_tokens)
        _stats["prefix_reused"] += min(prefix_tokens, prompt)
        _stats["prefix_reprocessed"] += max(0, prompt - prefix_tokens)
    logger.debug(
        "context: step prompt %d tokens (history %d, overhead %d of which schemas %d), "
        "cache prefix %d", prompt, history_tokens, overhead_tokens, schema_tokens,
        prefix_tokens,
    )


def record_compaction(dropped_messages: int, dropped_tokens: int) -> None:
    with _stats_lock:
        _stats["compactions"] += 1
        _stats["tokens_dropped"] += dropped_tokens
    # The rolling summary lives inside the leading system message, so rewriting it
    # changes the prompt's first token: the next step reuses nothing and reprocesses
    # the whole window. That is the trade compaction makes, and the numbers above are
    # how to judge it.
    logger.info(
        "context: compacted %d messages (%d tokens) into the summary; "
        "the next step's prompt cache starts from zero",
        dropped_messages, dropped_tokens,
    )


def stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def reset_stats() -> None:
    with _stats_lock:
        for key in _stats:
            _stats[key] = 0


# Current prompt size, updated by each record_step() call, so the REPL
# prompt can show a live context-usage bar.
_last_prompt_tokens: int = 0


def usage_fraction() -> float:
    """Return the last recorded prompt size as a fraction of the compaction threshold."""
    if not _last_prompt_tokens:
        return 0.0
    return min(1.0, _last_prompt_tokens / max(1, compaction_threshold()))


def count_thinking_tokens(text: str) -> int:
    """Tokens spent inside ``<think>...</think>`` blocks that are stripped for display.

    Returns 0 when no such blocks are present, so callers can skip the overhead
    of counting when the model's output is already clean.
    """
    if not text or "<think>" not in text:
        return 0
    import re
    total = 0
    for match in re.finditer(r"<think>(.*?)</think>", text, re.DOTALL):
        total += estimate_tokens(match.group(1))
    return total


def record_thinking_tokens(text: str) -> None:
    """Record tokens the model spent reasoning inside ``<think>`` blocks this step."""
    tokens = count_thinking_tokens(text)
    if not tokens:
        return
    with _stats_lock:
        _stats["thinking_tokens"] += tokens
        _stats["thinking_calls"] += 1


def stats_summary() -> str:
    """One line for `/health`, aimed at the two open questions: is the reserve the
    right size, and what is compaction costing the prompt cache?"""
    current = stats()
    if not current["steps"]:
        return "no agent steps yet"
    total_prefix = current["prefix_reused"] + current["prefix_reprocessed"]
    reuse = 100 * current["prefix_reused"] / total_prefix if total_prefix else 0
    scale_note = f", scale x{TOKENIZER_SCALE:.2f}" if TOKENIZER_SCALE != 1.0 else ""
    parts = [
        f"{current['steps']} steps, peak prompt {current['peak_prompt']} tokens{scale_note}"
        f" (history {current['peak_history']}, overhead {current['peak_overhead']} "
        f"of reserve {RESERVE_TOKENS}, schemas {current['peak_schemas']})",
        f"cache reuse {reuse:.0f}%",
        f"{current['compactions']} compactions dropping {current['tokens_dropped']} tokens",
    ]
    if current["thinking_calls"]:
        parts.append(
            f"thinking {current['thinking_tokens']} tokens in {current['thinking_calls']} step(s)"
        )
    return ", ".join(parts)
