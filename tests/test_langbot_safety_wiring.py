"""Integration tests: the catastrophic-command guard is wired into both shell
surfaces (``execute_shell_command`` and ``task_start``) at the @tool choke
point in langbot.py.

These import the real langbot module (heavy: langchain/langgraph/chromadb),
so they're kept in their own file and can be skipped in constrained
environments if needed. They deliberately do NOT re-test the detection logic
itself (see tests/test_safety.py for the full catastrophic/safe matrix) —
only that the tools refuse without ever reaching subprocess/task manager code
for a catastrophic command, and are unaffected otherwise.
"""

import os

import pytest

os.environ.setdefault("LANGBOT_VAULT_PASSWORD", "test-only-password")

langbot = pytest.importorskip(
    "langbot", reason="requires the full runtime dependency set (langchain/langgraph/chromadb)"
)


class TestExecuteShellCommandGuard:
    def test_catastrophic_command_is_refused_without_running(self, monkeypatch):
        called = {}

        def _fake_run(*args, **kwargs):
            called["ran"] = True
            raise AssertionError("subprocess.run must not be called for a catastrophic command")

        monkeypatch.setattr(langbot.subprocess, "run", _fake_run)
        result = langbot.execute_shell_command.func("rm -rf /")
        assert "Refused" in result
        assert "ran" not in called

    def test_safe_command_runs_unchanged(self):
        result = langbot.execute_shell_command.func("echo safety-wiring-ok")
        assert "safety-wiring-ok" in result
        assert "Refused" not in result

    def test_refusal_wording_matches_across_both_surfaces(self):
        shell_result = langbot.execute_shell_command.func("rm -rf /home")
        task_result = langbot.task_start.func("rm -rf /home")
        assert shell_result.startswith("Refused: command not executed (")
        assert task_result.startswith("Refused: command not executed (")


class TestTaskStartGuard:
    def test_catastrophic_command_never_reaches_task_manager(self, monkeypatch):
        called = {}

        def _fake_task_start(*args, **kwargs):
            called["ran"] = True
            raise AssertionError("_tasks.task_start must not be called for a catastrophic command")

        monkeypatch.setattr(langbot._tasks, "task_start", _fake_task_start)
        result = langbot.task_start.func("dd if=/dev/zero of=/dev/sda")
        assert "Refused" in result
        assert "ran" not in called

    def test_safe_command_reaches_task_manager_unchanged(self, monkeypatch):
        received = {}

        def _fake_task_start(command, cwd=None):
            received["command"] = command
            received["cwd"] = cwd
            return "Started task_deadbeef (pid 1). Use task_list / task_output / task_kill."

        monkeypatch.setattr(langbot._tasks, "task_start", _fake_task_start)
        result = langbot.task_start.func("sleep 1", cwd="/tmp")
        assert received == {"command": "sleep 1", "cwd": "/tmp"}
        assert "Started" in result
