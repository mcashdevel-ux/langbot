"""Unit tests for input.py — prompt sanitising, paste draining and read_input.

readline and stdin draining are stubbed so tests are deterministic and never
block on real terminal input.
"""

import io
import sys

import pytest

import input as inp


# ---------------------------------------------------------------------------
# _readline_safe_prompt
# ---------------------------------------------------------------------------

class TestSafePrompt:
    def test_empty_prompt_unchanged(self):
        assert inp._readline_safe_prompt("") == ""

    def test_wraps_ansi_when_readline_present(self, monkeypatch):
        monkeypatch.setattr(inp, "_HAS_READLINE", True)
        out = inp._readline_safe_prompt("\033[31m> \033[0m")
        assert "\001" in out and "\002" in out

    def test_no_readline_returns_prompt_asis(self, monkeypatch):
        monkeypatch.setattr(inp, "_HAS_READLINE", False)
        prompt = "\033[31m> \033[0m"
        assert inp._readline_safe_prompt(prompt) == prompt


# ---------------------------------------------------------------------------
# _drain_stdin
# ---------------------------------------------------------------------------

class TestDrainStdin:
    def test_dispatches_to_unix(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "linux")
        monkeypatch.setattr(inp, "_drain_stdin_unix", lambda: "unix-drained")
        assert inp._drain_stdin() == "unix-drained"

    def test_dispatches_to_windows(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(inp, "_drain_stdin_windows", lambda: "win-drained")
        assert inp._drain_stdin() == "win-drained"

    def test_unix_drain_handles_no_fileno(self, monkeypatch):
        # A stdin without a usable fileno must not raise.
        monkeypatch.setattr(sys, "stdin", io.StringIO("data"))
        assert inp._drain_stdin_unix() == ""

    def test_unix_drain_reads_pending_lines(self, monkeypatch):
        import os

        r, w = os.pipe()
        os.write(w, b"extra1\nextra2\n")
        os.close(w)
        fake_stdin = os.fdopen(r, "r")
        monkeypatch.setattr(sys, "stdin", fake_stdin)
        try:
            out = inp._drain_stdin_unix()
        finally:
            fake_stdin.close()
        assert out == "extra1\nextra2"

    def test_windows_drain_reads_via_msvcrt(self, monkeypatch):
        import types

        lines = iter(["pasted\n", ""])
        fake_msvcrt = types.SimpleNamespace(kbhit=lambda: True)
        monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
        monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(readline=lambda: next(lines)))
        out = inp._drain_stdin_windows()
        assert out == "pasted"


# ---------------------------------------------------------------------------
# read_input
# ---------------------------------------------------------------------------

class TestReadInput:
    @pytest.fixture(autouse=True)
    def no_drain(self, monkeypatch):
        # Default: nothing extra queued on stdin.
        monkeypatch.setattr(inp, "_drain_stdin", lambda: "")

    def test_basic_line(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "hello world")
        assert inp.read_input("> ") == "hello world"

    def test_strips_whitespace(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "  padded  ")
        assert inp.read_input() == "padded"

    def test_escape_prefix_cancels(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "\x1bcancel")
        assert inp.read_input() == ""

    def test_eof_propagates(self, monkeypatch):
        def raise_eof(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        with pytest.raises(EOFError):
            inp.read_input()

    def test_keyboardinterrupt_propagates(self, monkeypatch):
        def raise_kbi(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", raise_kbi)
        with pytest.raises(KeyboardInterrupt):
            inp.read_input()

    def test_multiline_paste_appended(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "first")
        monkeypatch.setattr(inp, "_drain_stdin", lambda: "second\nthird")
        assert inp.read_input() == "first\nsecond\nthird"

    def test_caps_at_5000_chars(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt="": "y" * 6000)
        assert len(inp.read_input()) == 5000

    def test_large_paste_prints_hint(self, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda prompt="": "line")
        monkeypatch.setattr(inp, "_drain_stdin", lambda: "\n".join(f"l{i}" for i in range(10)))
        inp.read_input()
        # A display hint is written to stdout for large pastes.
        assert "Pasted Text" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# setup_readline
# ---------------------------------------------------------------------------

class TestSetupReadline:
    def test_noop_without_readline(self, monkeypatch):
        monkeypatch.setattr(inp, "_HAS_READLINE", False)
        # Should simply return without touching the filesystem.
        assert inp.setup_readline() is None

    @pytest.mark.skipif(not inp._HAS_READLINE, reason="readline not available")
    def test_configures_history_and_completer(self, tmp_path):
        import readline

        histfile = tmp_path / "hist"
        inp.setup_readline(histfile=str(histfile), history_length=50)
        assert readline.get_history_length() == 50
        # The command completer should offer slash-commands for a matching prefix.
        completer = readline.get_completer()
        assert completer is not None
        first = completer("/he", 0)
        assert first == "/help"
        # Non-slash text yields no completion.
        assert completer("plain", 0) is None
