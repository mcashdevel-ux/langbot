"""Unit tests for tasks.py — the background task manager."""

import sys
import time

import pytest

import components.tasks as tasks


def _wait_until(pred, timeout=5.0, interval=0.02):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return False


@pytest.fixture
def mgr(tmp_path):
    return tasks.BackgroundTaskManager(tasks_dir=str(tmp_path / "tasks"))


class TestBackgroundTaskManager:
    def test_start_and_complete(self, mgr):
        t = mgr.start("echo hello-task")
        assert t.id.startswith("task_")
        assert _wait_until(lambda: mgr.get(t.id).status != "running")
        done = mgr.get(t.id)
        assert done.status == "exited"
        assert done.returncode == 0
        assert "hello-task" in mgr.output(t.id)

    def test_failing_command_marked_failed(self, mgr):
        t = mgr.start("exit 3")
        assert _wait_until(lambda: mgr.get(t.id).status != "running")
        done = mgr.get(t.id)
        assert done.status == "failed"
        assert done.returncode == 3

    def test_list_returns_tasks(self, mgr):
        mgr.start("echo a")
        mgr.start("echo b")
        assert len(mgr.list()) == 2

    def test_output_paging(self, mgr):
        t = mgr.start("printf 'abcdefgh'")
        assert _wait_until(lambda: mgr.get(t.id).status != "running")
        page = mgr.output(t.id, offset=0, max_chars=4)
        assert "abcd" in page
        assert "offset=4" in page

    def test_output_unknown_task(self, mgr):
        assert mgr.output("task_nope") is None

    def test_empty_command_rejected(self, mgr):
        with pytest.raises(ValueError):
            mgr.start("   ")

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal semantics")
    def test_kill_running_task(self, mgr):
        t = mgr.start("sleep 30")
        assert _wait_until(lambda: mgr.get(t.id).pid > 0)
        killed = mgr.kill(t.id)
        assert killed.status == "killed"
        # monitor thread should still finalize (ended_at set) after the kill.
        assert _wait_until(lambda: mgr.get(t.id).ended_at is not None)
        assert mgr.get(t.id).status == "killed"

    def test_kill_unknown_task(self, mgr):
        assert mgr.kill("task_nope") is None


class TestStringWrappers:
    def test_wrappers_use_singleton(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tasks, "manager",
                            tasks.BackgroundTaskManager(tasks_dir=str(tmp_path / "t")))
        out = tasks.task_start("echo wrapped")
        assert "Started task_" in out
        tid = out.split("Started ")[1].split(" ")[0]
        assert _wait_until(lambda: tasks.manager.get(tid).status != "running")
        assert "wrapped" in tasks.task_output(tid)
        assert tid in tasks.task_list()
        assert tid in tasks.task_status(tid)

    def test_unknown_ids(self, tmp_path, monkeypatch):
        monkeypatch.setattr(tasks, "manager",
                            tasks.BackgroundTaskManager(tasks_dir=str(tmp_path / "t")))
        assert "No such task" in tasks.task_output("task_x")
        assert "No such task" in tasks.task_status("task_x")
        assert "No such task" in tasks.task_kill("task_x")
