"""Run slow first-use initialisers on a background thread.

The embedding model and the Chroma collection each cost seconds to construct, and
``memory_store`` builds them lazily under a lock, so whoever touches memory first
pays for both. Doing that at import time makes every start-up slow; leaving it to
the first ``remember``/``recall`` makes one turn slow instead.

This runs them on a daemon thread while the REPL is already usable. It is purely
an optimisation: nothing here gates the tools, because ``memory_store``'s own lock
means a tool call arriving mid-warmup simply waits for the load in flight rather
than starting a second one or being told to come back later.

A job that raises is recorded rather than retried forever, and ``start()`` runs the
failed ones again on the next call, so a transient failure is not permanent.
"""

import logging
import threading

logger = logging.getLogger(__name__)

PENDING = "pending"
RUNNING = "running"
READY = "ready"
FAILED = "failed"


class Warmup:
    """Runs named initialisers once, in the background, in the given order."""

    def __init__(self, jobs):
        self._jobs = dict(jobs)
        self._status = {name: PENDING for name in self._jobs}
        self._errors = {}
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._running = False

    def start(self) -> bool:
        """Warm everything not ready yet; return whether a run was started.

        The "already started" flag is set *before* the thread is spawned and under
        the lock, so concurrent callers cannot each kick off their own copy of a
        multi-second model load.
        """
        with self._lock:
            if self._running:
                return False
            pending = [n for n, s in self._status.items() if s != READY]
            if not pending:
                return False
            self._running = True
            self._done.clear()
            for name in pending:
                self._status[name] = PENDING
                self._errors.pop(name, None)
        threading.Thread(target=self._run, args=(pending,),
                         name="langbot-warmup", daemon=True).start()
        return True

    def _run(self, names) -> None:
        try:
            for name in names:
                self._set(name, RUNNING)
                try:
                    self._jobs[name]()
                except Exception as e:  # one broken job must not strand the rest
                    self._set(name, FAILED, e)
                    logger.warning("warmup: %s failed: %s", name, e, exc_info=True)
                else:
                    self._set(name, READY)
                    logger.info("warmup: %s ready", name)
        finally:
            with self._lock:
                self._running = False
            self._done.set()

    def _set(self, name: str, status: str, error=None) -> None:
        with self._lock:
            self._status[name] = status
            if error is not None:
                self._errors[name] = error

    def wait(self, timeout=None) -> bool:
        """Block until the current run finishes; return False if it timed out."""
        return self._done.wait(timeout)

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def errors(self) -> dict:
        with self._lock:
            return dict(self._errors)

    def is_ready(self) -> bool:
        with self._lock:
            return all(s == READY for s in self._status.values())

    def summary(self) -> str:
        """One line for `/health`, e.g. ``embeddings ready, memory store running``."""
        status = self.status()
        errors = self.errors()
        parts = []
        for name, state in status.items():
            if state == FAILED and name in errors:
                parts.append(f"{name} failed ({errors[name]})")
            else:
                parts.append(f"{name} {state}")
        return ", ".join(parts) if parts else "nothing to warm"
