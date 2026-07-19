#!/usr/bin/env python3
"""
Input handling — readline, history, tab completion, paste detection, escape cancel.
Adapted from sage-std/core/input.py + twig patterns.
"""

import os
import sys
import time
import re
import atexit
from pathlib import Path

from console import (
    Fore, Style, _write, _term_w, _unicode_safe,
    is_large_paste, collapse_paste, warning,
)

try:
    import readline
    _HAS_READLINE = True
except ImportError:
    _HAS_READLINE = False


def _readline_safe_prompt(prompt: str) -> str:
    """Wrap ANSI escape codes in \001/\002 markers for readline line-width calc."""
    if not _HAS_READLINE or not prompt:
        return prompt
    return re.sub(r'(\033\[[0-9;]*m)', r'\001\1\002', prompt)


def _drain_stdin() -> str:
    """Drain any pending stdin (multi-line paste detection)."""
    if sys.platform == "win32":
        return _drain_stdin_windows()
    return _drain_stdin_unix()


def _drain_stdin_unix() -> str:
    try:
        import fcntl
    except ImportError:
        return ""
    lines = []
    try:
        fd = sys.stdin.fileno()
        old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, old_flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    line = sys.stdin.readline()
                except (BlockingIOError, ValueError):
                    break
                if not line:
                    break
                lines.append(line.rstrip("\n\r"))
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old_flags)
    except Exception:
        pass
    return "\n".join(lines)


def _drain_stdin_windows() -> str:
    try:
        import msvcrt
    except ImportError:
        return ""
    lines = []
    try:
        t0 = time.time()
        while time.time() - t0 < 0.05:
            if msvcrt.kbhit():
                line = sys.stdin.readline().rstrip("\n\r")
                if line:
                    lines.append(line)
                    t0 = time.time()
                else:
                    break
            else:
                time.sleep(0.005)
    except Exception:
        pass
    return "\n".join(lines)


def setup_readline(histfile=None, history_length: int = 2000):
    """Configure readline with persistent history, arrow keys, tab completion."""
    if not _HAS_READLINE:
        return
    import readline as _rl
    if histfile is None:
        histfile = Path.home() / ".sage_history"
    hf = Path(histfile)
    hf.parent.mkdir(parents=True, exist_ok=True)
    try:
        _rl.read_history_file(str(hf))
    except (FileNotFoundError, OSError):
        pass
    atexit.register(lambda: _rl.write_history_file(str(hf)))
    _rl.set_history_length(history_length)
    _rl.parse_and_bind("set keymap emacs")
    _rl.parse_and_bind("set editing-mode emacs")
    _rl.parse_and_bind("tab: complete")

    _cmds = [
        "/help", "/quit", "/exit", "/clear", "/new", "/info",
        "/knowledge", "/health", "/ls", "/save",
    ]

    def _completer(text, state):
        if text.startswith("/"):
            opts = [c for c in _cmds if c.startswith(text)]
            return opts[state] if state < len(opts) else None
        return None

    _rl.set_completer(_completer)
    _rl.set_completer_delims(" \t\n;")


def read_input(prompt: str = "") -> str:
    """Read a line of input with full terminal UX.

    Features:
      - Arrow key history navigation (via readline)
      - Multi-line paste detection (auto-drains remaining stdin)
      - Large paste detection (>5 lines or >500 chars → display hint)
      - Escape key cancellation (Esc + Enter → returns "")
      - Ctrl+C → raises KeyboardInterrupt
      - Ctrl+D → raises EOFError
      - 5000 char cap

    Returns:
        The input string, or "" if cancelled via Escape.
    Raises:
        KeyboardInterrupt, EOFError
    """
    safe_prompt = _readline_safe_prompt(prompt)
    try:
        first = input(safe_prompt)
    except (EOFError, KeyboardInterrupt):
        raise

    # Escape key at start of line → cancel turn
    if first and first.startswith("\x1b"):
        return ""

    lines = [first]

    # Drain remaining stdin for multi-line paste detection
    extra = _drain_stdin()
    if extra:
        lines.append(extra)

    result = "\n".join(lines)

    # Print a display hint for large pastes
    if is_large_paste(result):
        collapsed = collapse_paste(result)
        try:
            sys.stdout.write(f"\r  {Fore.YELLOW}\U0001f4cb{Style.RESET_ALL} {collapsed}\n")
            sys.stdout.flush()
        except Exception:
            pass

    return result.strip()[:5000]
