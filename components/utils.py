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

# Default cap for tool output / file reads surfaced to the model.
MAX_OUTPUT_CHARS = 20000
TRUNCATION_MARKER = "\n...[truncated]"


def truncate(text: str, max_chars: int = MAX_OUTPUT_CHARS,
             marker: str = TRUNCATION_MARKER) -> str:
    """Return ``text`` capped at ``max_chars``, appending ``marker`` if cut."""
    if len(text) > max_chars:
        return text[:max_chars] + marker
    return text


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
