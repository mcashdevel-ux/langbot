"""On-disk scratchpad shared by every tool that can return a large payload.

Tools save the full result here under a short id and return only a small
preview to the model; ``read_scratch`` pages through the rest on demand. This
keeps big results (web pages, file contents, grep output) out of the message
thread while still reachable.

What is saved is never truncated: capping belongs to the preview, which the
model can always widen by paging. A scratch entry that silently lost its tail
would be worse than no entry at all, because the preview advertises it as the
full result.

Near-leaf module by design — it depends only on ``config`` (itself dependency
free), so any tool module can import it without risking an import cycle.
"""

import os
import uuid

from .config import config

SCRATCH_DIR = config.get("paths.scratch_dir", "./memory/agent_scratch",
                         env="AGENT_SCRATCH_DIR")
os.makedirs(SCRATCH_DIR, exist_ok=True)


def _new_scratch_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def save_to_scratch(content: str, prefix: str = "doc") -> str:
    """Write ``content`` verbatim to a new scratch entry and return its id."""
    sid = _new_scratch_id(prefix)
    path = os.path.join(SCRATCH_DIR, f"{sid}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return sid


def offload(content: str, prefix: str, inline_chars: int, label: str) -> str:
    """Return ``content`` inline, or a preview plus a scratch id once it is
    longer than ``inline_chars``.

    ``label`` names what was saved ("full file", "full output", ...) and is
    shown in the header so the model knows what paging will get it.
    """
    if len(content) <= inline_chars:
        return content
    sid = save_to_scratch(content, prefix=prefix)
    return (f"{len(content)} chars, showing first {inline_chars} "
            f"({label} at scratch:{sid}):\n{content[:inline_chars]}")


def _valid_utf8_prefix_len(data: bytes) -> int:
    """Return the length of the longest prefix of ``data`` that is valid UTF-8."""
    try:
        data.decode("utf-8")
        return len(data)
    except UnicodeDecodeError as e:
        return e.start


def read_scratch(scratch_id: str, offset: int = 0, length: int = 1500) -> str:
    """Page through a previously saved tool result.

    Offsets and lengths are byte-based and stay consistent with the file's byte
    size, so paging works for non-ASCII (multi-byte UTF-8) content. When a page
    boundary lands in the middle of a multi-byte character, the read is extended
    to include the whole character (rather than dropping it), so paging with the
    returned ``end`` as the next ``offset`` reassembles the content losslessly.
    """
    path = os.path.join(SCRATCH_DIR, f"{scratch_id}.txt")
    if not os.path.exists(path):
        return f"(no scratch entry found for id '{scratch_id}')"
    offset = max(0, offset)
    total = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(max(0, length))
        # If we stopped mid-character (not at EOF), pull up to 3 more bytes to
        # complete it — a UTF-8 char is at most 4 bytes — then keep only the
        # complete-character prefix.
        if raw and offset + len(raw) < total and _valid_utf8_prefix_len(raw) < len(raw):
            raw += f.read(3)
            raw = raw[:_valid_utf8_prefix_len(raw)]
    end = offset + len(raw)
    more = end < total
    chunk = raw.decode("utf-8", errors="ignore")
    tail = f"\n...(more available, call read_scratch with offset={end})" if more else ""
    return f"[scratch:{scratch_id} bytes {offset}-{end}/{total}]\n{chunk}{tail}"
