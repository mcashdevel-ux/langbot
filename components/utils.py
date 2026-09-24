"""Shared utility helpers used across the agent modules.

Small, dependency-free helpers factored out of langbot.py / vault.py to avoid
duplicated logic (output truncation, atomic JSON persistence).
"""

import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Union

from .config import config

# Default cap for tool output / file reads surfaced to the model. Sized for a
# small local model on a 32k window: 20k chars is a third of that window spent
# on one result, and the full text is a read_scratch call away anyway.
MAX_OUTPUT_CHARS = config.get("tools.max_output_chars", 8000)
TRUNCATION_MARKER = "\n...[truncated]"


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS,
             marker: str = TRUNCATION_MARKER) -> str:
    """Return ``text`` capped at ``max_chars``, appending ``marker`` if cut."""
    if len(text) > max_chars:
        return text[:max_chars] + marker
    return text


def thinking_content(response) -> str:
    """Return a response's reasoning text, whichever channel it arrived on.

    Providers that support reasoning split it out of the answer, and each does so
    differently: an OpenAI-compatible endpoint (DeepSeek, vLLM, OpenRouter, …)
    puts it in ``reasoning_content`` / ``reasoning`` — either on the message or
    nested under ``provider_specific_fields`` — and some wrap it as a content
    *block* (``{"type": "thinking", …}``). ``ChatOpenAI`` drops the top-level
    fields entirely (its documented scope is the official OpenAI schema), but a
    subclass or a future version keeps them in ``additional_kwargs``, so all of
    those shapes are checked.

    Returns ``""`` when the response carries no reasoning, so callers can skip
    rendering without inspecting the shape themselves.
    """
    def _first_text(value) -> str:
        """Pull prose out of a str / list-of-blocks / dict-shaped field."""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            for key in ("text", "reasoning", "reasoning_content", "reasoning_details",
                        "content"):
                found = _first_text(value.get(key))
                if found:
                    return found
            return ""
        if isinstance(value, (list, tuple)):
            for item in value:
                found = _first_text(item)
                if found:
                    return found
            return ""
        return ""

    sources = []
    for holder in (response, getattr(response, "additional_kwargs", None)):
        if not holder:
            continue
        if isinstance(holder, dict):
            sources.append(holder)
            continue
        for attr in ("reasoning_content", "reasoning", "provider_specific_fields"):
            value = getattr(holder, attr, None)
            if value:
                sources.append(value)

    for source in sources:
        if isinstance(source, dict):
            for key in ("reasoning_content", "reasoning", "reasoning_details",
                        "provider_specific_fields"):
                found = _first_text(source.get(key))
                if found:
                    return found
        else:
            found = _first_text(source)
            if found:
                return found

    content = getattr(response, "content", None)
    if isinstance(content, (list, tuple)):
        for block in content:
            if isinstance(block, dict) and "think" in str(block.get("type", "")):
                found = _first_text(block)
                if found:
                    return found
    return ""


def reasoning_tokens(response) -> int:
    """Provider-reported reasoning tokens for a response, or 0 if unreported.

    ``usage_metadata`` carries them under ``output_token_details.reasoning`` on
    the hosted proxy; a local llama.cpp server reports nothing at all.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    details = {}
    if isinstance(usage, dict):
        details = usage.get("output_token_details") or {}
    else:                                    # pydantic UsageMetadata
        details = getattr(usage, "output_token_details", None) or {}
    try:
        return int((details or {}).get("reasoning") or 0)
    except (TypeError, ValueError):
        return 0


def strip_code_fences(raw: str) -> str:
    """Strip a leading/trailing Markdown code fence from a model reply.

    Handles single-line fenced replies (e.g. ```["a"]```) without the
    ``IndexError`` a naive ``split("\\n", 1)[1]`` would raise, as well as
    fences carrying a language tag (```json). Returns the inner text, stripped.
    """
    text = raw.strip()
    if not text.startswith("```"):
        return text
    # Drop the opening fence line (may include a language tag).
    parts = text.split("\n", 1)
    text = parts[1] if len(parts) == 2 else parts[0][3:]
    text = text.strip()
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


@contextlib.contextmanager
def suppress_native_output():
    """Silence *all* output — including from C extensions and child threads —
    for the duration of the ``with`` block.

    Progress bars emitted while loading ML models (e.g. HuggingFace/tqdm's
    ``Loading weights: 100%|█| 103/103``) are written straight to the process's
    stderr file descriptor, so redirecting ``sys.stdout``/``sys.stderr`` alone
    is not enough. We duplicate and replace the underlying fds (1 and 2) with
    ``os.devnull`` and restore them afterwards. Falls back to Python-level
    redirection when the streams have no real file descriptor (e.g. under
    pytest's capture).
    """
    try:
        stdout_fd = sys.stdout.fileno()
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # No real fds (captured streams) — redirect at the Python level only.
        with open(os.devnull, "w") as devnull, \
                contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
            yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(stdout_fd)
    saved_err = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stdout_fd)
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, stdout_fd)
        os.dup2(saved_err, stderr_fd)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull_fd)


def atomic_write_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Writes to a temp file in the same directory, then ``os.replace`` so readers
    never observe a partially written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", delete=False, dir=str(path.parent), suffix=".tmp"
    )
    with tmp:
        json.dump(data, tmp, indent=indent)
    os.replace(tmp.name, str(path))
