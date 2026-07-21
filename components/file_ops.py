"""File operations — hardened read/write plus surgical patching and diffs.

Ported and adapted from sage-std/core/tools/{files,patches}.py into langbot's
dependency-light style. All functions are pure (return a status/result string)
so they can be unit-tested without importing the heavy top-level agent module.

- ``read_file``  — text read with binary detection and truncation.
- ``write_file`` — idempotent write with non-string content coercion.
- ``patch_file`` — surgical find/replace with .py syntax-check + auto-rollback.
- ``batch_patch``— apply many patches in one call (tolerant of sloppy input).
- ``git_diff``   — show a file's git diff.
"""

import json
import os
import subprocess

from .utils import truncate


def _format_size(size_bytes: float) -> str:
    """Human-readable byte size (e.g. ``4.2KB``)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{int(size_bytes)}B" if unit == "B" else f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def read_file(file_path: str) -> str:
    """Read a text file, detecting binary files instead of dumping their bytes.

    Binary files (NUL byte in the first 8 KB) are reported by name and size
    rather than decoded into the model's context. Text output is truncated.
    """
    if not file_path:
        return "Error: empty file path."
    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(path):
        return f"Error: Path '{file_path}' does not exist."
    if not os.path.isfile(path):
        return f"Error: Not a file: '{file_path}'."
    try:
        with open(path, "rb") as f:
            header = f.read(8192)
    except OSError as e:
        return f"Failed to read {file_path}: {e}"
    if b"\0" in header:
        size = os.path.getsize(path)
        return f"[Binary file: {os.path.basename(path)}, {_format_size(size)}, path: {path}]"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError as e:
        return f"Failed to read {file_path}: {e}"
    if not content:
        return "(empty file)"
    return truncate(content)


def _coerce_content(content) -> str:
    """Coerce non-string tool content (LLMs sometimes send lists/dicts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "\n".join(str(item) for item in content)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, indent=2)
    return str(content)


def write_file(file_path: str, content, append: bool = False) -> str:
    """Write ``content`` to ``file_path``.

    Idempotent: an overwrite whose content already matches is skipped. Coerces
    non-string content to text. Creates parent directories as needed.
    """
    if not file_path:
        return "Error: empty file path."
    path = os.path.abspath(os.path.expanduser(file_path))
    text = _coerce_content(content)
    parent = os.path.dirname(path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as e:
            return f"Error: cannot create directory {parent}: {e}"

    if not append and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                if f.read() == text:
                    return f"Already up-to-date: '{file_path}' (no change)."
        except OSError:
            pass  # fall through to write

    try:
        with open(path, "a" if append else "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        return f"Failed to write: {e}"
    verb = "Appended" if append else "Wrote"
    return f"{verb} {len(text)} characters to '{file_path}'."


def patch_file(file_path: str, old_text: str, new_text: str) -> str:
    """Surgically replace the first occurrence of ``old_text`` with ``new_text``.

    - Idempotent: if ``old_text`` is absent but ``new_text`` is already present,
      reports success without changing the file.
    - For ``.py`` files, the result is syntax-checked and rolled back on error.
    """
    if not file_path:
        return "Error: empty file path."
    path = os.path.abspath(os.path.expanduser(file_path))
    if not os.path.exists(path):
        return f"Error: File not found: {file_path}"
    if not os.path.isfile(path):
        return f"Error: Not a file: {file_path}"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        return f"Error: cannot read file: {e}"

    if old_text not in content:
        if new_text and new_text in content:
            return f"Idempotent: patch already applied to {os.path.basename(path)}."
        return (
            f"Error: old_text not found in {os.path.basename(path)}. "
            "Check exact whitespace/indentation."
        )

    new_content = content.replace(old_text, new_text, 1)
    if new_content == content:
        return "Error: no changes made (text identical)."

    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        return f"Error: cannot write file: {e}"

    if path.endswith(".py"):
        try:
            compile(new_content, path, "exec")
        except SyntaxError as e:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
            except OSError:
                pass
            return f"Error: syntax error after patch (rolled back): {e}"

    return f"Patched {os.path.basename(path)} ({len(old_text)} -> {len(new_text)} chars)."


def batch_patch(patches) -> str:
    """Apply a list of ``{file_path, old_text, new_text}`` patches.

    Tolerant of the common LLM mistakes of sending a JSON string instead of a
    list, or ``oldText``/``newText`` key casings. Each patch is independent.
    """
    if isinstance(patches, str):
        try:
            patches = json.loads(patches)
        except json.JSONDecodeError:
            return f"Error: patches is a string, not a list: {patches[:200]}"
    if not isinstance(patches, list):
        return f"Error: patches must be a list, got {type(patches).__name__}."
    if not patches:
        return "Error: no patches provided."

    results = []
    for i, p in enumerate(patches, 1):
        if not isinstance(p, dict):
            results.append(f"  [{i}/{len(patches)}] ?: Error: not an object.")
            continue
        fp = p.get("file_path") or p.get("filepath") or ""
        old = p.get("old_text", p.get("oldText", ""))
        new = p.get("new_text", p.get("newText", ""))
        res = patch_file(fp, old, new)
        results.append(f"  [{i}/{len(patches)}] {os.path.basename(fp) if fp else '?'}: {res}")

    applied = sum(1 for r in results if "Patched" in r)
    idem = sum(1 for r in results if "Idempotent" in r)
    failed = sum(1 for r in results if "Error" in r)
    summary = f"Batch patch: {applied} applied, {idem} skipped, {failed} failed"
    return summary + "\n" + "\n".join(results)


def _git_root(path: str):
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
            cwd=os.path.dirname(path) if os.path.isfile(path) else path,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def git_diff(file_path: str = ".", cached: bool = False) -> str:
    """Show ``git diff`` for a path (working tree, or index with ``cached``)."""
    path = os.path.abspath(os.path.expanduser(file_path or "."))
    if not os.path.exists(path):
        return f"Error: Path not found: {file_path}"
    root = _git_root(path)
    if not root:
        return "Error: not in a git repository."
    rel = os.path.relpath(path, root)
    cmd = ["git", "diff"] + (["--cached"] if cached else []) + ["--", rel]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=root)
    except subprocess.TimeoutExpired:
        return "Error: git diff timed out."
    except OSError as e:
        return f"Error: git diff failed: {e}"
    output = result.stdout or result.stderr
    if not output.strip():
        return "No changes."
    return truncate(output)
