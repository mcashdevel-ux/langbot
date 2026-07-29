"""Code / file navigation tools — search and batch read.

Ported from sage-std/core/tools/{code_search,read_many}.py, trimmed to the
pieces that add the most value to langbot: text search across a tree, batch
reads by glob, and glob listing. Pure functions, unit-testable in isolation.
"""

import glob as _glob
import os
import subprocess

from .config import config
from .scratch import save_to_scratch
from .utils import truncate

# Matches shown inline; larger result sets are saved whole to scratch and paged
# through with read_scratch.
GREP_INLINE_LINES = config.get("tools.grep_inline_lines", 20)
# Inline cap for read_many_files; the full concatenation goes to scratch.
MANYFILES_INLINE_CHARS = config.get("tools.manyfiles_inline_chars", 4000)


def _format_matches(pattern: str, lines: list[str]) -> str:
    """Render grep-style match lines, paging via scratch past the inline cap."""
    if not lines:
        return f"No matches for '{pattern}'."
    if len(lines) <= GREP_INLINE_LINES:
        return truncate("\n".join(lines))
    sid = save_to_scratch("\n".join(lines), prefix="grep")
    preview = "\n".join(lines[:GREP_INLINE_LINES])
    return (f"{len(lines)} matches for '{pattern}' (showing first "
            f"{GREP_INLINE_LINES}; full list at scratch:{sid}):\n{preview}")


def find_in_files(pattern: str, path: str = ".") -> str:
    """Search for ``pattern`` across common source/text files (grep -rn).

    Falls back to a pure-Python scan when ``grep`` is unavailable. Result sets
    larger than ``GREP_INLINE_LINES`` are saved to scratch and previewed, so no
    match is silently dropped.
    """
    if not pattern:
        return "Error: empty pattern."
    includes = [
        "*.py", "*.js", "*.ts", "*.tsx", "*.jsx", "*.md", "*.json",
        "*.yaml", "*.yml", "*.txt", "*.cfg", "*.ini", "*.toml", "*.sh",
    ]
    try:
        result = subprocess.run(
            ["grep", "-rn"] + [f"--include={g}" for g in includes]
            + ["--", pattern, path or "."],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"Search timed out for '{pattern}'."
    except FileNotFoundError:
        return _find_in_files_py(pattern, path)
    except OSError as e:
        return f"Error: {e}"
    output = (result.stdout or "").strip()
    return _format_matches(pattern, output.splitlines() if output else [])


def _find_in_files_py(pattern: str, path: str = ".") -> str:
    """Pure-Python fallback for ``find_in_files`` (no grep on the system)."""
    root = os.path.abspath(os.path.expanduser(path or "."))
    exts = {".py", ".js", ".ts", ".tsx", ".jsx", ".md", ".json",
            ".yaml", ".yml", ".txt", ".cfg", ".ini", ".toml", ".sh"}
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "node_modules", "__pycache__"}]
        for name in filenames:
            if os.path.splitext(name)[1] not in exts:
                continue
            fp = os.path.join(dirpath, name)
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if pattern in line:
                            results.append(f"{fp}:{i}:{line.strip()[:200]}")
            except OSError:
                continue
    return _format_matches(pattern, results)


def read_many_files(pattern: str, max_files: int = 20, max_chars_per_file: int = 10000) -> str:
    """Read files matching a glob pattern, concatenated with headers."""
    if not pattern:
        return "Error: empty pattern."
    try:
        files = sorted(_glob.glob(pattern, recursive=True))
    except OSError as e:
        return f"Error expanding glob: {e}"
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        return f"No files matching '{pattern}'."
    files = files[:max_files]
    parts = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read(max_chars_per_file)
        except OSError as e:
            parts.append(f"--- {f} ---\nError: {e}")
            continue
        parts.append(f"--- {f} ---\n{body}")
    full = "\n\n".join(parts)
    if len(full) <= MANYFILES_INLINE_CHARS:
        return full
    sid = save_to_scratch(full, prefix="manyfiles")
    return (f"{len(files)} file(s) matching '{pattern}' — {len(full)} chars, showing "
            f"first {MANYFILES_INLINE_CHARS} (full text at scratch:{sid}):\n"
            f"{full[:MANYFILES_INLINE_CHARS]}")


def glob_list(pattern: str, max_results: int = 100) -> str:
    """List files/dirs matching a glob pattern with sizes (no contents)."""
    if not pattern:
        return "Error: empty pattern."
    try:
        matches = sorted(_glob.glob(pattern, recursive=True))
    except OSError as e:
        return f"Error expanding glob: {e}"
    if not matches:
        return f"No files matching '{pattern}'."
    note = ""
    if len(matches) > max_results:
        note = f"\n... ({len(matches) - max_results} more omitted)"
        matches = matches[:max_results]
    lines = [f"Matches for '{pattern}':"]
    for m in matches:
        if os.path.isdir(m):
            lines.append(f"  [dir]  {m}")
        else:
            try:
                lines.append(f"  {os.path.getsize(m):>10} B  {m}")
            except OSError:
                lines.append(f"  {'?':>10}    {m}")
    lines.append(f"\n{len(matches)} shown{note}")
    return "\n".join(lines)
