"""Track G — KernelContext dependency-injection wiring in langbot.py.

These import the real langbot module (heavy), so they're kept in their own file
and skipped in constrained environments if needed.  They verify:

  - ``_ctx_for_run`` builds a per-run ``KernelContext`` stashed into
    ``config["configurable"]["kernel_context"]``, with a per-thread task manager
    (same thread reuses its manager across turns; different threads get isolated
    managers, so two concurrent sessions don't clobber each other's task lists).
  - ``_resolve_ctx`` round-trips the stashed context back out of a tool's config.

  - ``tools_node`` passes its context to the confirmation gate, so pending
    approval state lands on the per-run context (not the module-level defaults),
    giving per-session isolation for concurrent runs.

  - The task tools resolve the per-thread manager when a context is present, and
    fall back to the module-level singleton when absent (direct ``.func()`` calls,
    REPL slash handlers).
"""

import os

import pytest

os.environ.setdefault("LANGBOT_VAULT_PASSWORD", "test-only-password")

langbot = pytest.importorskip(
    "langbot", reason="requires the full runtime dependency set (langchain/langgraph/chromadb)"
)


class TestCtxForRun:
    def test_builds_and_stashes_context(self):
        config = {"configurable": {"thread_id": "t-ctx-1"}}
        ctx = langbot._ctx_for_run(config)
        assert ctx.thread_id == "t-ctx-1"
        assert config["configurable"]["kernel_context"] is ctx
        assert langbot._resolve_ctx(config) is ctx

    def test_same_thread_reuses_task_manager(self):
        c1 = {"configurable": {"thread_id": "t-shared"}}
        c2 = {"configurable": {"thread_id": "t-shared"}}
        ctx1 = langbot._ctx_for_run(c1)
        ctx2 = langbot._ctx_for_run(c2)
        assert ctx1.tasks is ctx2.tasks

    def test_different_threads_get_isolated_managers(self):
        c1 = {"configurable": {"thread_id": "t-iso-a"}}
        c2 = {"configurable": {"thread_id": "t-iso-b"}}
        ctx1 = langbot._ctx_for_run(c1)
        ctx2 = langbot._ctx_for_run(c2)
        assert ctx1.tasks is not ctx2.tasks
        assert ctx1.thread_id != ctx2.thread_id


class TestResolveCtx:
    def test_none_when_no_config(self):
        assert langbot._resolve_ctx(None) is None
        assert langbot._resolve_ctx({}) is None

    def test_none_when_no_kernel_context(self):
        assert langbot._resolve_ctx({"configurable": {"thread_id": "x"}}) is None


class TestGateContextIsolation:
    def test_pending_state_lands_on_ctx_not_module(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        ctx = langbot._KernelContext(thread_id="t-gate")
        allowed, reason = langbot._check_confirm_gate(
            "execute_shell_command", {"name": "execute_shell_command", "args": {"command": "rm -rf ./build"}, "id": "c1"}, ctx
        )
        assert not allowed
        assert ctx.pending_confirm == "rm -rf ./build"
        assert langbot._pending_confirm is None   # module-level untouched

    def test_approval_via_ctx_clears_ctx(self, monkeypatch):
        monkeypatch.setattr(langbot, "_CONFIRM_MUTATING", True)
        langbot._clear_pending_confirm()
        ctx = langbot._KernelContext(thread_id="t-gate2")
        langbot._check_confirm_gate(
            "execute_shell_command", {"name": "execute_shell_command", "args": {"command": "rm -rf ./build"}, "id": "c1"}, ctx

        )
        langbot._note_user_input("yes", ctx=ctx)
        assert ctx.pending_approved is True
        allowed, _ = langbot._check_confirm_gate(
            "execute_shell_command", {"name": "execute_shell_command", "args": {"command": "rm -rf ./build"}, "id": "c1"}, ctx)
        assert allowed
        assert ctx.pending_confirm is None


class TestTaskToolsResolveCtx:
    def test_task_list_uses_per_thread_manager(self):
        config = {"configurable": {"thread_id": "t-tools"}}
        ctx = langbot._ctx_for_run(config)
        out = langbot.task_list.func(config=config)
        assert "No background tasks." in out
        # The per-thread manager is empty; the module singleton may have tasks
        # from other tests, but that must not leak into this thread's view.


    def test_task_list_falls_back_to_singleton_without_ctx(self):
        out = langbot.task_list.func()
        assert isinstance(out, str)
