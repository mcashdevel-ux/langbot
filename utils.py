"""Shared utility helpers used across the agent modules.

Small, dependency-free helpers factored out of langbot.py / vault.py to avoid
duplicated logic (output truncation, atomic JSON persistence).
"""

import json
import os
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
