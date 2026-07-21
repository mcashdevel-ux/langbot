"""Code / file navigation tools — search and batch read.

Ported from sage-std/core/tools/{code_search,read_many}.py, trimmed to the
pieces that add the most value to langbot: text search across a tree, batch
reads by glob, and glob listing. Pure functions, unit-testable in isolation.
"""

import glob as _glob
import os
import subprocess

from .utils import truncate


def find_in_files(pattern: str, path: str = ".") -> str:
    """Search for ``pattern`` across common source/text files (grep -rn).

    Falls back to a pure-Python scan when ``grep`` is unavailable.
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
            + ["-m", "5", "--", pattern, path or "."],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return f"Search timed out for '{pattern}'."
    except FileNotFoundError:
        return _find_in_files_py(pattern, path)
    except OSError as e:
        return f"Error: {e}"
    output = (result.stdout or "").strip()
    if not output:
        return f"No matches for '{pattern}'."
    return truncate(output)


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
                            if len(results) >= 100:
                                return truncate("\n".join(results))
            except OSError:
                continue
    return truncate("\n".join(results)) if results else f"No matches for '{pattern}'."


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
    parts, total = [], 0
    for f in files:
        try:
            with open(f, "r", encoding="utf-8", errors="replace") as fh:
                body = fh.read(max_chars_per_file)
        except OSError as e:
            parts.append(f"--- {f} ---\nError: {e}")
            continue
        parts.append(f"--- {f} ---\n{body}")
        total += len(body)
        if total > 50000:
            parts.append("... (truncated, total output exceeds 50K chars)")
            break
    return "\n\n".join(parts)


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
