"""Reclaim disk left behind by finished sessions.

Two stores grow without bound in normal use, and neither failure is visible until
the disk is full:

- ``paths.scratch_dir`` keeps every offloaded tool result forever, and saves are
  deliberately uncapped (see ``scratch.py``), so a handful of large fetches is
  hundreds of megabytes.
- the LangGraph SQLite checkpointer mints a new ``thread_id`` on every start and
  every ``/new``, and nothing ever deletes the rows of a thread nobody will resume.

So a sweep runs once per start, on the warmup thread (never on the interactive
loop). Both halves are deliberately conservative: the scratch sweep keeps anything
recent regardless of size, and the checkpoint sweep keeps the newest threads plus
the one in use, because deleting state the user still wants is worse than keeping
bytes they don't.

Recency for checkpoints comes from ``rowid`` order rather than a timestamp: the
checkpointer's schema stores none, and ``thread_id`` is only sometimes derived from
the clock (``session_<date>`` at start, a random hex id after ``/new``).
"""

import logging
import os
import sqlite3
import time

from .config import config

logger = logging.getLogger(__name__)

ENABLED = config.get("housekeeping.enabled", True)
# A week of paging history is far more than any thread survives, and the size cap
# is the real backstop: age alone cannot stop one session writing 10 GB.
SCRATCH_MAX_AGE_DAYS = config.get("housekeeping.scratch_max_age_days", 7)
SCRATCH_MAX_TOTAL_MB = config.get("housekeeping.scratch_max_total_mb", 512)
# Threads to keep besides the active one. Resuming an older thread is not
# something the REPL offers today, so this is purely a safety margin.
CHECKPOINT_KEEP_THREADS = config.get("housekeeping.checkpoint_keep_threads", 20)

DAY = 86400.0


def prune_scratch(directory, max_age_days=None, max_total_mb=None, now=None) -> dict:
    """Delete scratch entries older than ``max_age_days``, then oldest-first until
    the directory fits ``max_total_mb``. Returns counts and bytes freed."""
    max_age_days = SCRATCH_MAX_AGE_DAYS if max_age_days is None else max_age_days
    max_total_mb = SCRATCH_MAX_TOTAL_MB if max_total_mb is None else max_total_mb
    now = time.time() if now is None else now
    result = {"removed": 0, "bytes": 0, "kept": 0}
    if not os.path.isdir(directory):
        return result

    entries = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            stat = os.stat(path)
        except OSError:                      # vanished under us; nothing to do
            continue
        if os.path.isdir(path):
            continue
        entries.append((stat.st_mtime, stat.st_size, path))
    entries.sort()                           # oldest first

    def _remove(path, size):
        try:
            os.remove(path)
        except OSError as e:
            logger.warning("housekeeping: could not remove %s: %s", path, e)
            return False
        result["removed"] += 1
        result["bytes"] += size
        return True

    survivors = []
    cutoff = now - max_age_days * DAY if max_age_days else None
    for mtime, size, path in entries:
        if cutoff is not None and mtime < cutoff:
            if _remove(path, size):
                continue
        survivors.append((mtime, size, path))

    if max_total_mb:
        budget = max_total_mb * 1024 * 1024
        total = sum(size for _, size, _ in survivors)
        while survivors and total > budget:
            _, size, path = survivors.pop(0)
            if _remove(path, size):
                total -= size
    result["kept"] = len(survivors)
    return result


def _thread_id_tables(conn) -> "list[str]":
    """Tables carrying a ``thread_id`` column, so a schema change cannot leave rows
    of a deleted thread behind in a table this module has never heard of."""
    tables = []
    for (name,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall():
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{name}")')}
        if "thread_id" in columns:
            tables.append(name)
    return tables


def prune_checkpoints(db_path, keep_threads=None, active_thread_id=None) -> dict:
    """Delete rows of all but the ``keep_threads`` most recent threads (plus the
    active one) from every checkpointer table. Returns threads/rows removed."""
    keep_threads = CHECKPOINT_KEEP_THREADS if keep_threads is None else keep_threads
    result = {"threads": 0, "rows": 0}
    if not db_path or not os.path.exists(db_path):
        return result

    # Short timeout: the checkpointer may hold the write lock, and a start-up
    # sweep must never be the reason the first turn waits.
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        tables = _thread_id_tables(conn)
        if not tables:
            return result

        recency = {}
        for table in tables:
            for thread_id, last in conn.execute(
                f'SELECT thread_id, MAX(rowid) FROM "{table}" GROUP BY thread_id'
            ):
                if last is not None and last > recency.get(thread_id, -1):
                    recency[thread_id] = last

        keep = {t for t, _ in sorted(recency.items(), key=lambda kv: -kv[1])[:keep_threads]}
        if active_thread_id:
            keep.add(active_thread_id)
        doomed = [t for t in recency if t not in keep]
        if not doomed:
            return result

        for table in tables:
            for thread_id in doomed:
                cursor = conn.execute(
                    f'DELETE FROM "{table}" WHERE thread_id = ?', (thread_id,)
                )
                result["rows"] += cursor.rowcount or 0
        conn.commit()
        result["threads"] = len(doomed)

        # Deleting rows only frees pages inside the file; the point of the sweep is
        # the disk back. A lock contention here is not a failure worth reporting.
        try:
            conn.execute("VACUUM")
        except sqlite3.Error as e:
            logger.info("housekeeping: skipped VACUUM (%s)", e)
    finally:
        conn.close()
    return result


def sweep(scratch_dir, checkpoint_db, active_thread_id=None) -> str:
    """Run both sweeps, log and return a one-line summary."""
    if not ENABLED:
        return "disabled"

    scratch = {"removed": 0, "bytes": 0}
    checkpoints = {"threads": 0, "rows": 0}
    try:
        scratch = prune_scratch(scratch_dir)
    except Exception as e:                   # never let a sweep break start-up
        logger.warning("housekeeping: scratch sweep failed: %s", e, exc_info=True)
    try:
        checkpoints = prune_checkpoints(checkpoint_db, active_thread_id=active_thread_id)
    except Exception as e:
        logger.warning("housekeeping: checkpoint sweep failed: %s", e, exc_info=True)

    summary = (
        f"{scratch['removed']} scratch entries "
        f"({scratch['bytes'] / (1024 * 1024):.1f} MB), "
        f"{checkpoints['threads']} threads ({checkpoints['rows']} rows)"
    )
    if scratch["removed"] or checkpoints["threads"]:
        logger.info("housekeeping: freed %s", summary)
    return summary
