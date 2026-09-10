"""Track F — confirmation-gate wiring in langbot.py (config ``tools.confirm_mutating``).

These import the real langbot module (heavy), so they're kept in their own file
and skipped in constrained environments if needed.  They test the gate
helpers (``_check_confirm_gate``, ``_gate_tool_calls``, ``_note_user_input``)
and the ``tools_node`` integration — NOT the SafetyGate verdict logic itself
(see tests/test_safety_gate.py for that matrix).

Default behaviour (``confirm_mutating: false``) must be byte-for-byte the
pre-Track-F flow: every non-catastrophic call runs immediately, no refusals.

"""

import os

import pytest

os.environ.setdefault("LANGBOT_VAULT_PASSWORD", "test-only-password")

langbot = pytest.importorskip(
    "langbot", reason="requires the full runtime dependency set (langchain/langgraph/chromadb)"
)


def _call(name, command, call_id="c1"):
    return {"name": name, "args": {"command": command}, "id": call_id}


class TestCheckConfirmGate:
    def test_read_tools_always_pass(self):
        allowed, reason = langbot._check_confirm_gate("read_any_file", {"args": {"file_path": "/x"}})
        assert allowed
        assert reason == ""

    def test_gate_off_passes_mutating_tools(self):
        """Default: autonomous — mutating calls run immediately."""
        langbot._CONFIRM_MUTATING = False
        try:
            allowed, _ = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf ./build"))
            assert allowed
        finally:
            langbot._CONFIRM_MUTATING = False

    def test_gate_on_refuses_gray_zone_shell(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        allowed, reason = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf ./build"))
        assert not allowed
        assert "requires confirmation" in reason
        assert langbot._pending_confirm == "rm -rf ./build"

    def test_gate_on_fast_allows_trusted_readonly(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        allowed, reason = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "ls -la"))
        assert allowed
        assert reason == ""

    def test_gate_on_hard_blocks_catastrophic(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        allowed, reason = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf /"))
        assert not allowed
        assert "BLOCKED" in reason

    def test_approved_call_runs_once_and_clears(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        # First call: refused, pending set.
        allowed, _ = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf ./build"))
        assert not allowed
        # User approves; re-issued identical call runs once.

        langbot._pending_approved = True
        allowed, _ = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf ./build"))
        assert allowed
        assert langbot._pending_confirm is None

    def test_different_command_after_approval_still_refused(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf ./build"))
        langbot._pending_approved = True
        allowed, _ = langbot._check_confirm_gate("execute_shell_command", _call("execute_shell_command", "rm -rf ./other"))
        assert not allowed
        assert langbot._pending_confirm == "rm -rf ./other"

    def test_non_shell_mutating_tool_gated(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        allowed, reason = langbot._check_confirm_gate("write_any_file", {"args": {"file_path": "/x", "content": "y"}})
        assert not allowed
        assert "requires confirmation" in reason


class TestNoteUserInput:
    def test_approval_approves_pending(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        langbot._pending_confirm = "rm -rf ./build"
        langbot._note_user_input("yes")
        assert langbot._pending_approved is True

    def test_non_approval_clears_pending(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        langbot._pending_confirm = "rm -rf ./build"
        langbot._note_user_input("actually, do something else")
        assert langbot._pending_confirm is None

        assert langbot._pending_approved is False

    def test_gate_off_is_noop(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", False)
        langbot._clear_pending_confirm()
        langbot._pending_confirm = "rm -rf ./build"
        langbot._note_user_input("yes")
        assert langbot._pending_approved is False

    def test_approval_with_no_pending_is_noop(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        langbot._note_user_input("yes")
        assert langbot._pending_approved is False


class TestGateToolCalls:
    def test_refused_calls_get_tool_messages(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        to_run = [
            _call("execute_shell_command", "ls -la", "c1"),
            _call("execute_shell_command", "rm -rf ./build", "c2"),
        ]
        runnable, refused = langbot._gate_tool_calls(to_run)
        assert [c["id"] for c in runnable] == ["c1"]
        assert len(refused) == 1
        assert refused[0].tool_call_id == "c2"
        assert "requires confirmation" in refused[0].content

    def test_gate_off_runs_everything(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", False)
        langbot._clear_pending_confirm()
        to_run = [_call("execute_shell_command", "rm -rf ./build", "c1")]
        runnable, refused = langbot._gate_tool_calls(to_run)
        assert len(runnable) == 1
        assert refused == []