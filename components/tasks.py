"""In-process background task manager for long-running shell commands.

Unlike fire-and-forget ``cmd &``, tasks started here are actively tracked: each
runs under its own process group with output streamed to a log file, and a
per-task monitor thread updates status the moment the process exits
(event-driven, no polling by the caller). The agent can list, inspect output,
and kill tasks by id.

Logs live under ``./memory/agent_tasks`` (override with ``AGENT_TASKS_DIR``) to
stay within the repo's memory policy. All state is in-process, so tasks do not
survive a restart of the agent — a restart's monitor threads are gone, so only
tasks started in the current run are managed.
"""

import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

TASKS_DIR = os.environ.get("AGENT_TASKS_DIR", "./memory/agent_tasks")

_IS_WINDOWS = sys.platform == "win32"


@dataclass
class Task:
    """A tracked background process."""

    id: str
    command: str
    cwd: Optional[str]
    log_path: str
    pid: int
    started_at: float
    status: str = "running"  # running | exited | failed | killed
    returncode: Optional[int] = None
    ended_at: Optional[float] = None

    def elapsed(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.time()
        return end - self.started_at


class BackgroundTaskManager:
    """Start, track, inspect, and kill background shell commands."""

    def __init__(self, tasks_dir: str = TASKS_DIR):
        self._dir = tasks_dir
        self._tasks: dict[str, Task] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()

    # ── lifecycle ──
    def start(self, command: str, cwd: Optional[str] = None,
              env: Optional[dict] = None) -> Task:
        """Launch ``command`` in the background and begin tracking it."""
        if not command or not command.strip():
            raise ValueError("empty command")
        os.makedirs(self._dir, exist_ok=True)
        tid = "task_" + uuid.uuid4().hex[:8]
        log_path = os.path.join(self._dir, f"{tid}.log")
        merged = os.environ.copy()
        if env:
            merged.update(env)
        log = open(log_path, "wb")
        popen_kwargs = dict(
            shell=True, cwd=cwd or None, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=merged,
        )
        if _IS_WINDOWS:
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_kwargs["start_new_session"] = True  # own process group for clean kill
        try:
            proc = subprocess.Popen(command, **popen_kwargs)
        except Exception:
            log.close()
            raise
        task = Task(
            id=tid, command=command, cwd=cwd or None, log_path=log_path,
            pid=proc.pid, started_at=time.time(),
        )
        with self._lock:
            self._tasks[tid] = task
            self._procs[tid] = proc
        threading.Thread(target=self._monitor, args=(task, proc, log),
                         daemon=True, name=f"monitor-{tid}").start()
        return task

    def _monitor(self, task: Task, proc: subprocess.Popen, log) -> None:
        """Wait for ``proc`` to exit, then finalize the task's status."""
        try:
            rc = proc.wait()
        finally:
            try:
                log.flush()
                log.close()
            except OSError:
                pass
        with self._lock:
            task.returncode = rc
            task.ended_at = time.time()
            if task.status != "killed":
                task.status = "exited" if rc == 0 else "failed"

    # ── queries ──
    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.started_at)

    def output(self, task_id: str, offset: int = 0, max_chars: int = 4000) -> Optional[str]:
        """Return a slice of a task's captured output, or ``None`` if unknown."""
        task = self.get(task_id)
        if task is None:
            return None
        offset = max(0, offset)
        try:
            with open(task.log_path, "rb") as f:
                f.seek(offset)
                raw = f.read(max(0, max_chars))
            total = os.path.getsize(task.log_path)
        except OSError as e:
            return f"(could not read task log: {e})"
        end = offset + len(raw)
        text = raw.decode("utf-8", errors="replace")
        more = f"\n...(more output, call with offset={end})" if end < total else ""
        return f"[{task_id} output bytes {offset}-{end}/{total}]\n{text}{more}"

    def kill(self, task_id: str) -> Optional[Task]:
        """Terminate a running task (whole process group). Idempotent."""
        task = self.get(task_id)
        if task is None:
            return None
        with self._lock:
            proc = self._procs.get(task_id)
            if task.status != "running":
                return task
            task.status = "killed"
        if proc is not None:
            _terminate(proc)
        return task


def _terminate(proc: subprocess.Popen) -> None:
    """Best-effort terminate a process (and its group on POSIX)."""
    try:
        if _IS_WINDOWS:
            proc.terminate()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, OSError):
        try:
            proc.terminate()
        except OSError:
            pass


# ── module-level singleton + string-returning convenience wrappers ──
manager = BackgroundTaskManager()


def _fmt(task: Task) -> str:
    line = f"{task.id}  {task.status:<7}  {task.elapsed():6.1f}s  pid={task.pid}"
    if task.returncode is not None:
        line += f"  rc={task.returncode}"
    return f"{line}  {task.command[:80]}"


def task_start(command: str, cwd: str = "") -> str:
    """Start a background task; returns its id."""
    try:
        task = manager.start(command, cwd=cwd or None)
    except ValueError as e:
        return f"Error: {e}"
    except OSError as e:
        return f"Error: failed to start task: {e}"
    return f"Started {task.id} (pid {task.pid}). Use task_list / task_output / task_kill."


def task_list() -> str:
    """List all background tasks and their status."""
    tasks = manager.list()
    if not tasks:
        return "No background tasks."
    return "\n".join(_fmt(t) for t in tasks)


def task_status(task_id: str) -> str:
    """Show one task's status and command."""
    task = manager.get(task_id)
    if task is None:
        return f"No such task: {task_id}"
    return _fmt(task)


def task_output(task_id: str, offset: int = 0) -> str:
    """Read captured output for a task, paged by byte offset."""
    out = manager.output(task_id, offset=offset)
    if out is None:
        return f"No such task: {task_id}"
    return out


def task_kill(task_id: str) -> str:
    """Terminate a running background task."""
    task = manager.kill(task_id)
    if task is None:
        return f"No such task: {task_id}"
    if task.status == "killed":
        return f"Killed {task_id}."
    return f"Task {task_id} was not running (status: {task.status})."
