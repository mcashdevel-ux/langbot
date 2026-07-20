#!/usr/bin/env python3
"""
Console UI — Terminal output with color, icons, spinners, and clean formatting.
Adapted from sage-std/console.py + twig UX patterns. No Rich dependency.
"""

import os
import sys
import time
import threading
import shutil
import re

# ── Colorama (optional, graceful fallback) ──
try:
    from colorama import Fore as _Fore, Style as _Style, init as _colorama_init
    _colorama_init()
    HAS_COLORAMA = True
except Exception:
    HAS_COLORAMA = False

if HAS_COLORAMA:
    class Fore:
        RED = _Fore.RED
        GREEN = _Fore.GREEN
        YELLOW = _Fore.YELLOW
        BLUE = _Fore.BLUE
        CYAN = _Fore.CYAN
        MAGENTA = _Fore.MAGENTA
        WHITE = _Fore.WHITE
        RESET = _Fore.RESET
        BLACK = _Fore.BLACK

    class Style:
        BRIGHT = _Style.BRIGHT
        DIM = _Style.DIM
        NORMAL = _Style.NORMAL
        RESET_ALL = _Style.RESET_ALL
else:
    class Fore:
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = WHITE = RESET = BLACK = ''
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ''

# ── Box-drawing / rule characters ──
# Kept as named constants so they can be referenced inside f-string expressions
# on Python < 3.12, which forbids backslashes in the expression part of f-strings.
_HLINE = "\u2500"        # ─
_DLINE = "\u2550"        # ═
_MIDDOT = "\u00b7"       # ·
_BLOCK_FULL = "\u2588"   # █
_BLOCK_LIGHT = "\u2591"  # ░

# ── Short ANSI aliases ──
G = Fore.GREEN
Y = Fore.YELLOW
C = Fore.CYAN
R = Fore.RED
M = Fore.MAGENTA
B = Fore.BLUE
W = Fore.WHITE
Z = Style.RESET_ALL
N = Style.NORMAL

# ── Thread-safe output ──
_OUTPUT_LOCK = threading.Lock()
_TERM_WIDTH = 80


def _term_w() -> int:
    global _TERM_WIDTH
    try:
        _TERM_WIDTH = shutil.get_terminal_size((80, 20)).columns
    except Exception:
        pass
    return _TERM_WIDTH


def _write(text: str):
    with _OUTPUT_LOCK:
        try:
            sys.stdout.write(text + '\n')
            sys.stdout.flush()
        except Exception:
            pass


def strip_ansi(text: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', text)


def ansi_len(text: str) -> int:
    return len(strip_ansi(text))


def truncate(text: str, max_len: int = 100) -> str:
    visible = strip_ansi(text)
    if len(visible) <= max_len:
        return text
    count = 0
    result = []
    i = 0
    while i < len(text) and count < max_len:
        if text[i] == '\033':
            end = text.find('m', i)
            if end >= 0:
                result.append(text[i:end+1])
                i = end + 1
            else:
                result.append(text[i])
                i += 1
        else:
            result.append(text[i])
            count += 1
            i += 1
    return ''.join(result) + '...'


# ═══════════════════════════════════════════════════════════════
# Status messages
# ═══════════════════════════════════════════════════════════════

def _status(icon: str, color: str, msg: str):
    _write(f"  {color}{icon}{Style.RESET_ALL} {msg}")


def info(msg: str):
    _status("\u2139", Fore.CYAN, msg)


def success(msg: str):
    _status("\u2713", Fore.GREEN, msg)


def warning(msg: str):
    _status("\u26a0", Fore.YELLOW, msg)


def error(msg: str):
    _status("\u2717", Fore.RED, msg)


# ═══════════════════════════════════════════════════════════════
# Section dividers
# ═══════════════════════════════════════════════════════════════

def rule(text: str = ""):
    w = _term_w()
    if text:
        t = f" {text} "
        pad = (w - ansi_len(t) - 2) // 2
        _write(f"  {Fore.MAGENTA}{_HLINE * pad}{Style.RESET_ALL}"
               f"{Fore.WHITE}{Style.BRIGHT}{t}{Style.RESET_ALL}"
               f"{Fore.MAGENTA}{_HLINE * pad}{Style.RESET_ALL}")
    else:
        _write(f"  {Fore.MAGENTA}{_HLINE * (w - 2)}{Style.RESET_ALL}")


def header(msg: str):
    w = _term_w()
    _write(f"\n  {Fore.MAGENTA}{_DLINE * (w - 2)}{Style.RESET_ALL}")
    _write(f"  {Fore.WHITE}{Style.BRIGHT}{msg}{Style.RESET_ALL}")
    _write(f"  {Fore.MAGENTA}{_DLINE * (w - 2)}{Style.RESET_ALL}")


def blank():
    _write("")


def separator():
    w = _term_w()
    _write(f"  {Fore.MAGENTA}{_MIDDOT * (w - 2)}{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════
# Banner (startup)
# ═══════════════════════════════════════════════════════════════

def banner(title: str, subtitle: str = ""):
    w = _term_w()
    inner = w - 4
    lines = [
        f"\n  {Fore.MAGENTA}\u2554{_DLINE * inner}\u2557{Style.RESET_ALL}",
        f"  {Fore.MAGENTA}\u2551{Style.RESET_ALL}"
        f"{Fore.WHITE}{Style.BRIGHT}{title:^{inner}}{Style.RESET_ALL}"
        f"{Fore.MAGENTA}\u2551{Style.RESET_ALL}",
    ]
    if subtitle:
        lines.insert(2, f"  {Fore.MAGENTA}\u2551{Style.RESET_ALL}{'':^{inner}}{Fore.MAGENTA}\u2551{Style.RESET_ALL}")
        lines.insert(3, f"  {Fore.MAGENTA}\u2551{Style.RESET_ALL}"
                        f"{Fore.CYAN}{subtitle:^{inner}}{Style.RESET_ALL}"
                        f"{Fore.MAGENTA}\u2551{Style.RESET_ALL}")
    lines.append(f"  {Fore.MAGENTA}\u255a{_DLINE * inner}\u255d{Style.RESET_ALL}")
    _write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════
# Tool output
# ═══════════════════════════════════════════════════════════════

_TOOL_ICONS = {
    "shell": ("$", Fore.YELLOW),
    "web_search": ("\U0001f50d", Fore.CYAN),
    "read_file": ("\U0001f4d6", Fore.GREEN),
    "write_file": ("\u270f\ufe0f", Fore.GREEN),
    "list_directory": ("\U0001f4c1", Fore.GREEN),
    "patch_file": ("\U0001f527", Fore.YELLOW),
    "diff": ("\U0001f4ca", Fore.BLUE),
    "python_run": ("\U0001f40d", Fore.CYAN),
    "syntax_check": ("\u2705", Fore.GREEN),
    "memory": ("\U0001f9e0", Fore.YELLOW),
    "knowledge": ("\U0001f50e", Fore.CYAN),
    "get_knowledge": ("\U0001f50e", Fore.CYAN),
    "add_to_knowledge": ("\U0001f9e0", Fore.YELLOW),
    "read": ("\U0001f4d6", Fore.GREEN),
    "write": ("\u270f\ufe0f", Fore.GREEN),
    "exec": ("$", Fore.YELLOW),
    "ls": ("\U0001f4c1", Fore.GREEN),
    "info": ("\u2139", Fore.CYAN),
    "success": ("\u2713", Fore.GREEN),
    "warning": ("\u26a0", Fore.YELLOW),
    "error": ("\u2717", Fore.RED),
}


def _tool_icon(name: str):
    return _TOOL_ICONS.get(name, ("\u25cf", Fore.WHITE))


def tool_call(name: str, args: dict = None):
    icon, color = _tool_icon(name)
    if args:
        parts = []
        for k, v in args.items():
            vs = str(v)
            if len(vs) > 60:
                vs = vs[:57] + "..."
            parts.append(f"{k}={vs!r}")
        args_str = ", ".join(parts)
        _write(f"  {color}{icon}{Style.RESET_ALL} {Fore.WHITE}{Style.BRIGHT}{name}{Style.RESET_ALL}({args_str})")
    else:
        _write(f"  {color}{icon}{Style.RESET_ALL} {Fore.WHITE}{Style.BRIGHT}{name}{Style.RESET_ALL}()")


def tool_result(name: str, preview: str, elapsed_ms: int = 0,
                is_error: bool = False, full_len: int = 0):
    icon, color = _tool_icon(name)
    status = f"{Fore.RED}\u2717{Style.RESET_ALL}" if is_error else f"{Fore.GREEN}\u2713{Style.RESET_ALL}"
    if elapsed_ms:
        el = f" ({elapsed_ms}ms)" if elapsed_ms < 2000 else f" ({elapsed_ms/1000:.1f}s)"
    else:
        el = ""
    label = f"{color}{icon}{Style.RESET_ALL} {name}{el}"
    display = preview
    if full_len > 500:
        display = preview[:500]
        display += f"\n    ... [{full_len:,} total chars]"
    _write(f"  {status} {label}")
    for line in display.strip().split('\n'):
        _write(f"    {line}")


# ═══════════════════════════════════════════════════════════════
# Agent response display
# ═══════════════════════════════════════════════════════════════

def agent_response(content: str, label: str = "Agent"):
    if not content:
        return
    is_thought = "thought" in label.lower()
    if is_thought:
        _write(f"  {Fore.MAGENTA}{Style.BRIGHT}{label}{Style.RESET_ALL}")
        _write(f"  {Fore.MAGENTA}{_HLINE * min(_term_w() - 4, 50)}{Style.RESET_ALL}")
        for line in content.strip().split("\n"):
            _write(f"  {line.strip()}")
    else:
        _write(f"  {Fore.GREEN}{Style.BRIGHT}\U0001f4ac Response{Style.RESET_ALL}")
        for line in content.strip().split("\n"):
            _write(f"  {Fore.WHITE}{line.rstrip()}{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════
# Key-value display
# ═══════════════════════════════════════════════════════════════

def kv(key: str, value: str, col1: int = 20):
    k = f"{Fore.CYAN}{key}{Style.RESET_ALL}"
    v = f"{Fore.WHITE}{value}{Style.RESET_ALL}"
    _write(f"  {k:>{col1}} : {v}")


# ═══════════════════════════════════════════════════════════════
# Table
# ═══════════════════════════════════════════════════════════════

def table(rows: list[list], headers: list[str] = None):
    if not rows:
        info("(empty)")
        return
    col_count = max(len(r) for r in rows)
    if headers:
        col_count = max(col_count, len(headers))
    widths = [0] * col_count
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], ansi_len(str(cell)))
    if headers:
        for i, h in enumerate(headers):
            widths[i] = max(widths[i], ansi_len(h))
    sep = "\u2500" * (sum(widths) + (col_count - 1) * 3 + 2)
    if headers:
        hdr = " \u2502 ".join(
            f"{Fore.WHITE}{Style.BRIGHT}{h:<{widths[i]}}{Style.RESET_ALL}"
            for i, h in enumerate(headers)
        )
        _write(f"  {hdr}")
        _write(f"  {Fore.MAGENTA}{sep}{Style.RESET_ALL}")
    for row in rows:
        r = " \u2502 ".join(
            f"{str(c):<{widths[i]}}" for i, c in enumerate(row)
        )
        _write(f"  {r}")


# ═══════════════════════════════════════════════════════════════
# Code block
# ═══════════════════════════════════════════════════════════════

def code(text: str, lang: str = ""):
    _write(f"  {Fore.YELLOW}{_HLINE * (_term_w() - 4)}{Style.RESET_ALL}")
    for line in text.strip().split('\n'):
        _write(f"  {Fore.YELLOW}{line}{Style.RESET_ALL}")
    _write(f"  {Fore.YELLOW}{_HLINE * (_term_w() - 4)}{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════
# Progress bar
# ═══════════════════════════════════════════════════════════════

def progress_bar(value: float, width: int = 20, label: str = "") -> str:
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    empty = width - filled
    pct = int(value * 100)
    if pct >= 100:
        color = Fore.GREEN
    elif pct >= 80:
        color = Fore.YELLOW
    else:
        color = Fore.CYAN
    bar = f"{color}{_BLOCK_FULL * filled}{Style.RESET_ALL}{_BLOCK_LIGHT * empty}"
    result = f"  {bar} {color}{pct}%{Style.RESET_ALL}"
    if label:
        result += f"  {Fore.WHITE}{label}{Style.RESET_ALL}"
    return result


def print_progress(value: float, width: int = 20, label: str = ""):
    _write(progress_bar(value, width, label))


# ═══════════════════════════════════════════════════════════════
# Animated gradient spinner
# ═══════════════════════════════════════════════════════════════

_GRADIENT_COLORS = [
    Fore.MAGENTA, Fore.BLUE, Fore.CYAN,
    Fore.GREEN, Fore.YELLOW, Fore.RED,
]


class GradientSpinner:
    def __init__(self, msg: str = "Working..."):
        self.msg = msg
        self._running = False
        self._thread = None
        self._chars = "\u281b\u2819\u2839\u2838\u283c\u2834\u2826\u2827\u2807\u280f"
        if sys.platform == "win32":
            self._chars = "|/-\\"

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            sys.stdout.write('\r' + ' ' * (ansi_len(self.msg) + 4) + '\r')
            sys.stdout.flush()
        except Exception:
            pass

    def update(self, msg: str):
        self.msg = msg

    def _spin(self):
        i = 0
        color_idx = 0
        g_len = len(_GRADIENT_COLORS) or 1
        c_len = len(self._chars) or 1
        while self._running:
            char = self._chars[i % c_len]
            color = _GRADIENT_COLORS[color_idx % g_len] if HAS_COLORAMA else Fore.CYAN
            try:
                sys.stdout.write(f'\r  {color}{char}{Style.RESET_ALL} {self.msg}')
                sys.stdout.flush()
            except Exception:
                pass
            i += 1
            color_idx += 1
            time.sleep(0.03)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


Spinner = GradientSpinner


# ═══════════════════════════════════════════════════════════════
# Ghost text (inline completion preview)
# ═══════════════════════════════════════════════════════════════

def ghost_text(prefix: str, completion: str) -> str:
    if not completion:
        return prefix
    if completion.startswith(prefix):
        ghost = completion[len(prefix):]
        if ghost:
            return f"{prefix}{Style.DIM}{Fore.WHITE}{ghost}{Style.RESET_ALL}"
    return prefix


# ═══════════════════════════════════════════════════════════════
# Paste collapsing
# ═══════════════════════════════════════════════════════════════

_LARGE_PASTE_LINE_THRESHOLD = 5
_LARGE_PASTE_CHAR_THRESHOLD = 500


def is_large_paste(text: str) -> bool:
    if not text:
        return False
    line_count = text.count('\n') + 1
    return line_count > _LARGE_PASTE_LINE_THRESHOLD or len(text) > _LARGE_PASTE_CHAR_THRESHOLD


def collapse_paste(text: str) -> str:
    if not is_large_paste(text):
        return text
    lines = text.count('\n') + 1
    return f"[Pasted Text: {lines} lines, {len(text)} chars]"


# ═══════════════════════════════════════════════════════════════
# Status row
# ═══════════════════════════════════════════════════════════════

def status_row(loading_msg: str = "", tip: str = "",
               context: str = "", is_loading: bool = False) -> str:
    parts = []
    if is_loading and loading_msg:
        parts.append(f"{Fore.CYAN}\u280b{Style.RESET_ALL} {loading_msg}")
    elif loading_msg:
        parts.append(loading_msg)
    if tip:
        parts.append(f"{Fore.WHITE}{Style.DIM}{tip}{Style.RESET_ALL}")
    if context:
        parts.append(f"{Fore.CYAN}{context}{Style.RESET_ALL}")
    return "  " + "  \u00b7  ".join(parts)


def print_status_row(loading_msg: str = "", tip: str = "",
                     context: str = "", is_loading: bool = False):
    _write(status_row(loading_msg, tip, context, is_loading))


# ═══════════════════════════════════════════════════════════════
# Session resume banner
# ═══════════════════════════════════════════════════════════════

def session_resume_banner(msg_count: int, knowledge_count: int = 0, is_truncated: bool = False):
    w = _term_w()
    inner = w - 4
    subtitle = f"Resumed session ({msg_count} messages)"
    if knowledge_count:
        subtitle += f", {knowledge_count} knowledge facts"
    if is_truncated:
        subtitle += f" (showing last {msg_count})"
    lines = [
        f"  {Fore.GREEN}\u2554{_DLINE * inner}\u2557{Style.RESET_ALL}",
        f"  {Fore.GREEN}\u2551{Style.RESET_ALL}"
        f"{Fore.WHITE}{Style.BRIGHT}{subtitle:^{inner}}{Style.RESET_ALL}"
        f"{Fore.GREEN}\u2551{Style.RESET_ALL}",
        f"  {Fore.GREEN}\u255a{_DLINE * inner}\u255d{Style.RESET_ALL}",
    ]
    _write('\n'.join(lines))


# ═══════════════════════════════════════════════════════════════
# Knowledge retrieval indicator
# ═══════════════════════════════════════════════════════════════

def knowledge_hint(count: int):
    if count > 0:
        _write(f"  {Fore.YELLOW}\U0001f9e0{Style.RESET_ALL} Retrieved {count} relevant fact{'s' if count != 1 else ''}")


# ═══════════════════════════════════════════════════════════════
# Startup tip
# ═══════════════════════════════════════════════════════════════

def startup_tip(model: str = ""):
    model_str = f" \u2014 {model}" if model else ""
    _write(f"  {Fore.CYAN}{Style.DIM}Type /help for commands.  Esc+Enter cancels.  Ctrl+C twice exits.{Style.RESET_ALL}")


# ═══════════════════════════════════════════════════════════════
# Backward compat aliases
# ═══════════════════════════════════════════════════════════════

print_info = info
print_success = success
print_warning = warning
print_error = error
print_header = header
print_rule = rule
print_banner = banner


def _unicode_safe(text: str) -> str:
    """Strip or replace Unicode chars that Windows consoles can't display."""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            return ''
    if sys.platform != 'win32':
        try:
            text.encode('utf-8')
            return text
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    safe = []
    for ch in text:
        cp = ord(ch)
        if cp < 32 and cp not in (10, 13, 9, 27):
            continue
        if 0x1F300 <= cp <= 0x1F9FF:
            safe.append('?')
        elif 0xFE00 <= cp <= 0xFE0F:
            continue
        elif 0x200D == cp:
            continue
        else:
            try:
                ch.encode('utf-8')
                safe.append(ch)
            except (UnicodeEncodeError, UnicodeDecodeError):
                safe.append('?')
    return ''.join(safe)


# ═══════════════════════════════════════════════════════════════
# Rich-based Panel + Markdown rendering (optional, graceful fallback)
# ═══════════════════════════════════════════════════════════════

_HAS_RICH = False
try:
    from rich.console import Console as _RichConsole
    from rich.panel import Panel as _RichPanel
    from rich.markdown import Markdown as _RichMarkdown
    from rich import box as _RichBox
    from rich.style import Style as _RichStyle
    _RICH_CONSOLE = _RichConsole(highlight=False)
    _HAS_RICH = True
except Exception:
    _HAS_RICH = False

# ── Panel border characters (fallback when Rich unavailable) ──
_PANEL_CHARS = {
    "round":   ("╭", "╮", "╰", "╯", "─", "│"),
    "single":  ("┌", "┐", "└", "┘", "─", "│"),
    "double":  ("╔", "╗", "╚", "╝", "═", "║"),
    "heavy":   ("┏", "┓", "┗", "┛", "━", "┃"),
}


def _apply_md_simple(text: str) -> str:
    """Simple Markdown rendering for terminal: bold, italic, code, links, headers."""
    if not text:
        return text
    if not HAS_COLORAMA:
        # Strip md markers when no color
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'`(.+?)`', r'\1', text)
        text = re.sub(r'#{1,6}\s+', '', text)
        text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
        return text
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', lambda m: f"{Style.BRIGHT}{Fore.WHITE}{m.group(1)}{Style.RESET_ALL}", text)
    # Italic
    text = re.sub(r'\*(.+?)\*', lambda m: f"\033[3m{m.group(1)}\033[23m", text)
    # Inline code
    text = re.sub(r'`([^`]+)`', lambda m: f"{Fore.YELLOW}{m.group(1)}{Style.RESET_ALL}", text)
    # Links
    text = re.sub(r'\[(.+?)\]\((.+?)\)', lambda m: f"{Fore.CYAN}{m.group(1)}{Style.RESET_ALL}({Fore.BLUE}{m.group(2)}{Style.RESET_ALL})", text)
    # Headers
    text = re.sub(r'^#{1,6}\s+(.+)$', lambda m: f"{Style.BRIGHT}{Fore.MAGENTA}{m.group(1)}{Style.RESET_ALL}", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r'^---+\s*$', lambda m: f"{Fore.MAGENTA}{'─' * (_term_w() - 6)}{Style.RESET_ALL}", text, flags=re.MULTILINE)
    return text


def _render_markdown(text: str) -> str:
    """Render Markdown text. Uses Rich if available, otherwise simple ANSI rendering."""
    if _HAS_RICH:
        # Rich handles it internally — we return the text as-is and render via Rich
        return text
    return _apply_md_simple(text)


def panel(title: str = "", content: str = "", border_style: str = "green",
          subtitle: str = "", width: int = None, render_md: bool = False):
    """
    Display a panel box with title, content, and optional Markdown rendering.
    Uses Rich if available, otherwise falls back to simple box-drawing chars.
    """
    w = width or (_term_w() - 2)
    if w < 20:
        w = 20

    panel_content = content

    if _HAS_RICH:
        # Rich-based rendering
        style_map = {
            "green": "green",
            "blue": "blue",
            "cyan": "cyan",
            "yellow": "yellow",
            "red": "red",
            "magenta": "magenta",
            "white": "white",
        }
        bs = style_map.get(border_style, border_style)
        
        if render_md and panel_content:
            inner = _RichMarkdown(panel_content, code_theme="monokai")
        else:
            inner = panel_content

        p = _RichPanel(
            inner,
            title=title or None,
            subtitle=subtitle or None,
            border_style=bs,
            box=_RichBox.ROUNDED,
            padding=(0, 1),
            width=min(w, _term_w()),
        )
        with _OUTPUT_LOCK:
            _RICH_CONSOLE.print(p)
            _RICH_CONSOLE.print()
    else:
        # Fallback: simple box-drawing
        chars = _PANEL_CHARS.get("single", _PANEL_CHARS["single"])
        tl, tr, bl, br, h, v = chars
        
        if render_md:
            panel_content = _apply_md_simple(panel_content)
        
        lines = panel_content.strip().split('\n')
        
        # Build title line
        title_str = f" {title} " if title else ""
        top = f"{tl}{h}{title_str}{h * (w - len(title_str) - 3)}{tr}" if title_str else f"{tl}{h * (w - 2)}{tr}"
        
        result_lines = [f"  {top}"]
        for line in lines:
            visible_len = ansi_len(line)
            padding = max(0, w - visible_len - 4)
            result_lines.append(f"  {v} {line}{' ' * padding}{v}")
        
        bottom = f"{bl}{h * (w - 2)}{br}"
        result_lines.append(f"  {bottom}")
        
        with _OUTPUT_LOCK:
            try:
                sys.stdout.write('\n'.join(result_lines) + '\n\n')
                sys.stdout.flush()
            except Exception:
                pass


def md_print(text: str):
    """Print text with Markdown formatting directly to the terminal."""
    if _HAS_RICH:
        with _OUTPUT_LOCK:
            _RICH_CONSOLE.print(_RichMarkdown(text, code_theme="monokai"))
    else:
        _write(_apply_md_simple(text))


# ── Backward compat alias ──
print_md = md_print
