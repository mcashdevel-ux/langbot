"""Head+tail truncation — keep both ends, save full content to a file.

Ported from neo's ``neo/core/truncate.py`` (itself inspired by openhands-sdk's
``maybe_truncate``, simplified).  Where langbot's scratch-based tools already
offload large results to the scratchpad (``components/scratch.py``)and page
through them with ``read_scratch``, this module is the *kernel-level* fallback:
a tool result that did not self-offload (e.g. a plugin tool, or a tool that
returns a huge string verbatim) is caught here instead of being dumped whole
into the message state — the head and tail stay inline, the full content is
saved to a hash-named file under ``./memory/truncated/``, and the notice tells
the model where to read it back. No silent data loss, no custom paging
protocol.

Design rules (mirroring neo):
  - head (~60%)and tail (~40%) inline, so the model sees both the beginning
    and end of the result.
  - full content saved to a sha256-dedup filename (same content → same file,
    no second write).
  - notice names the file path so the model can navigate it with existing
    tools (``read_any_file``).
  - results under the cap are returned verbatim, no file created.
  - ``truncate_after=0`` (or ``None``) disables truncation entirely.


Near-leaf module by design — it depends only on ``config``, so any tool module
can import it without risking an import cycle..
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

from .config import config

# Results under this many chars are returned verbatim.  Matches neo's default;
# the kernel fallback only fires on results that did not self-offload, which are
# rare, so the cap can sit above langbot's own per-tool inline caps (8000).
TRUNCATE_CHARS = config.get("tools.truncate_chars", 12_000)
# Where full-content files land.  Per MEMORY_POLICY.md: under ./memory/,
# configurable via ``paths.truncated_dir`` (or ``AGENT_TRUNCATED_DIR``).
TRUNCATED_DIR = config.get("paths.truncated_dir", "./memory/truncated",
                             env="AGENT_TRUNCATED_DIR")

# Notice inserted between head and tail when truncated..
_TRUNCATE_NOTICE = (
    "\n[... {omitted} chars omitted — full output saved to {file_path} "
    "(use read_any_file to see it) ...]\n"
)

# When the notice itself wouldn't fit, use this shorter form..
_SHORT_NOTICE = "\n[... truncated ...]\n"


def _save_full(content: str, save_dir: Path, prefix: str) -> Optional[str]:
    """Save full content to a hash-named file. Returns the path or None."""
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    h = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:10]
    filename = f"{prefix}_{h}.txt"
    p = save_dir / filename
    if not p.exists():
        try:
            p.write_text(content, encoding="utf-8")
        except Exception:  # noqa: BLE001 — a save failure degrades to the short notice
            return None
    return str(p)


def maybe_truncate(
    content: str,
    truncate_after: Optional[int] = TRUNCATE_CHARS,
    save_dir: Optional[Path] = None,
    prefix: str = "output",
) -> str:
    """Truncate the middle of content if it exceeds ``truncate_after`` chars.


    Keeps head (~60%)and tail (~40%) so the model sees both the beginning
    and end. Full content is saved to a file the model can read with
    existing tools — no custom paging protocol, no silent data loss.



    Args:
        content: text to potentially truncate..
        truncate_after: max chars before truncation (0 or None = no limit).
        save_dir: directory for full-content files (default: ``paths.truncated_dir``).
        prefix: filename prefix (e.g. "shell", "grep", "glob_list").

    Returns:
        Original content if under limit, or head+notice+tail..
    """
    if not truncate_after or len(content) <= truncate_after:
        return content

    # Where to save the full content
    if save_dir is None:
        save_dir = Path(TRUNCATED_DIR)

    file_path = _save_full(content, save_dir, prefix)
    omitted = len(content) - truncate_after

    if file_path:
        notice = _TRUNCATE_NOTICE.format(omitted=omitted, file_path=file_path)
    else:
        notice = _SHORT_NOTICE

    # If the notice is too long relative to the budget, use the short form
    if len(notice) > truncate_after // 4:
        notice = _SHORT_NOTICE

    available = truncate_after - len(notice)
    if available <= 0:
        return notice

    head_chars = max(0, int(available * 0.6))
    tail_chars = max(0, available - head_chars)

    return (
        content[:head_chars]
        + notice
        + (content[-tail_chars:] if tail_chars > 0 else "")
    )