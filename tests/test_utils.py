"""Unit tests for utils.py — the parts touched by the console/UX work.

Focus on ``suppress_native_output``: it must silence writes to the underlying
stdout/stderr file descriptors (how ML progress bars leak) and always restore
them, including on error.
"""

import io
import os
import sys

import pytest

import components.utils as utils


class TestSuppressNativeOutput:
    def test_suppresses_fd_level_writes(self, capfd):
        # capfd captures at the fd level (like the real terminal would receive).
        print("before", flush=True)
        with utils.suppress_native_output():
            os.write(sys.stdout.fileno(), b"HIDDEN-STDOUT\n")
            os.write(sys.stderr.fileno(), b"HIDDEN-STDERR\n")
        print("after", flush=True)

        out, err = capfd.readouterr()
        assert "before" in out and "after" in out
        assert "HIDDEN-STDOUT" not in out
        assert "HIDDEN-STDERR" not in err

    def test_restores_fds_on_exception(self, capfd):
        with pytest.raises(RuntimeError):
            with utils.suppress_native_output():
                raise RuntimeError("boom")
        # After the block the fds are restored and writing works again.
        print("restored", flush=True)
        assert "restored" in capfd.readouterr().out

    def test_falls_back_without_real_fds(self, monkeypatch):
        # Streams without a usable fileno (e.g. StringIO) must not raise.
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        with utils.suppress_native_output():
            print("swallowed")
        # No exception == pass.
