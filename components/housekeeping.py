"""Reclaim disk left behind by finished sessions.

Three things grow without bound in normal use, and none of the failures is visible
until the disk is full:

- ``paths.scratch_dir`` keeps every offloaded tool result forever, and saves are
  deliberately uncapped (see ``scratch.py``), so a handful of large fetches is
  hundreds of megabytes.
- the LangGraph SQLite checkpointer mints a new ``thread_id`` on every start and
  every ``/new``, and nothing ever deletes the rows of a thread nobody will resume.
- the same checkpointer stores a **full snapshot of the whole message history on
  every super-step**, so one long session is O(N^2) on its own — measured here at
  180 MB in 50 rows, 87% of a 10 GB store in a single 18-hour session (see
  ``docs/sitrep-2026-09-23-checkpoint-db-lock.md``).

So a sweep runs once per start, on the warmup thread (never on the interactive
loop). Each part is deliberately conservative: the scratch sweep keeps anything
recent regardless of size, the thread sweep keeps the newest threads plus the one
in use, and the history sweeps trim only the *oldest* checkpoints of a thread —
resuming replays from the newest one, so deleting older snapshots costs no state
the user can reach. Deleting state the user still wants is worse than keeping
bytes they don't.

Three bounds, not two, and the third is the one that matters most. The thread
sweep bounds how many threads survive but keeps them whatever their size; the
live-thread sweep bounds the size of the thread in use. Neither touches a thread
that is *abandoned yet still recent enough to keep* — so one runaway session could
sit at 9 GB indefinitely, which is exactly what happened. The per-thread ceiling
(``prune_fat_threads``) closes that: every thread over ``checkpoint_max_thread_mb``
is trimmed to its newest checkpoints, oldest first.

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
# Checkpoints to keep *within* the live thread. The checkpointer snapshots the
# whole message history on every super-step, so one long session is O(N^2) on its
# own: the largest thread measured here held 180 MB in 50 rows. Pruning whole
# threads (above) cannot bound that, so the live thread's history is trimmed too.
CHECKPOINT_KEEP_PER_THREAD = config.get("housekeeping.checkpoint_keep_per_thread", 20)
# Size backstop for the live thread. If it still exceeds this after trimming to
# CHECKPOINT_KEEP_PER_THREAD rows, keep halving the keep-count until it fits or
# one row remains. 0 disables the backstop.
CHECKPOINT_MAX_MB = config.get("housekeeping.checkpoint_max_mb", 1024)
# Per-thread size budget, applied to *every* thread rather than only the live one.
# Neither bound above can stop one abandoned thread from holding gigabytes: the
# thread sweep keeps the newest CHECKPOINT_KEEP_THREADS threads whatever their size,
# and the history sweep only looks at the active thread. So a fat thread that is
# neither active nor old enough to be pruned is never trimmed at all. Measured: a
# single 18-hour session reached 9 GB (87% of a 10 GB store, see
# docs/sitrep-2026-09-23-checkpoint-db-lock.md), and 263 MB here in a thread that
# was neither in use nor old enough to drop. This is the backstop that makes "one
# thread eats the disk" impossible. 0 disables it.
CHECKPOINT_MAX_THREAD_MB = config.get("housekeeping.checkpoint_max_thread_mb", 256)

DAY = 86400.0


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


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

        # Deleting rows only frees pages inside the file, so the file does not
        # shrink on its own. A full VACUUM would reclaim it, but it rewrites the
        # whole database into a temp file while holding an exclusive lock for the
        # entire duration — on a multi-GB checkpoint DB that is minutes, and every
        # checkpointer write in the meantime fails with "database is locked".
        # So the sweep never VACUUMs. Reclaiming space is an offline operation
        # (see scripts/vacuum_checkpoints.py) run when no session is live.
        #
        # incremental_vacuum is safe here: it only touches the freelist, takes no
        # long exclusive lock, and is a no-op unless auto_vacuum=INCREMENTAL.
        try:
            conn.execute("PRAGMA incremental_vacuum")
        except sqlite3.Error as e:
            logger.info("housekeeping: skipped incremental_vacuum (%s)", e)
    finally:
        conn.close()
    return result


def _trim_thread(conn, tables, thread_id, keep_n, by_checkpoint_id) -> int:
    """Keep the ``keep_n`` newest checkpoints of ``thread_id``, delete the rest.

    Rows are ranked by ``rowid``: the checkpointer schema stores no timestamp, and
    rowid order is insertion order, which is what recency means here. The ``writes``
    rows are deleted with their checkpoint — a write whose checkpoint is gone is
    unreachable state. A table carrying ``thread_id`` but no ``checkpoint_id``
    (schema drift) is trimmed by rowid instead, so the "sweep every table that
    carries thread state" rule holds.
    """
    if by_checkpoint_id:
        safe = [
            row[0] for row in conn.execute(
                'SELECT checkpoint_id FROM "checkpoints" '
                "WHERE thread_id = ? ORDER BY rowid DESC LIMIT ?",
                (thread_id, keep_n),
            )
        ]
        if not safe:
            return 0
        placeholders = ", ".join("?" for _ in safe)
    deleted = 0
    for table in tables:
        columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if by_checkpoint_id and "checkpoint_id" in columns:
            cursor = conn.execute(
                f'DELETE FROM "{table}" WHERE thread_id = ? '
                f"AND checkpoint_id NOT IN ({placeholders})",
                (thread_id, *safe),
            )
        else:                               # schema drift: trim by recency instead
            cursor = conn.execute(
                f'DELETE FROM "{table}" WHERE thread_id = ? AND rowid NOT IN ('
                f'SELECT rowid FROM "{table}" WHERE thread_id = ? '
                "ORDER BY rowid DESC LIMIT ?)",
                (thread_id, thread_id, keep_n),
            )
        deleted += cursor.rowcount or 0
    conn.commit()
    return deleted


def _thread_bytes(conn, thread_id) -> int:
    """Bytes of checkpoint blobs held by one thread, on an open connection."""
    row = conn.execute(
        'SELECT COALESCE(SUM(LENGTH(checkpoint)), 0) FROM "checkpoints" '
        "WHERE thread_id = ?", (thread_id,),
    ).fetchone()
    return row[0] if row else 0


def thread_blob_bytes(db_path) -> dict:
    """Map ``thread_id`` → bytes of checkpoint blobs it holds, biggest first.

    Used to find the threads worth trimming: the total file size is dominated by a
    handful of threads, and a thread's *row count* is a poor proxy for its size
    because every row snapshots the whole history (so a long session's rows are far
    fatter than a short one's).
    """
    sizes = {}
    if not db_path or not os.path.exists(db_path):
        return sizes
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        tables = _thread_id_tables(conn)
        if "checkpoints" not in tables:
            return sizes
        for thread_id, total in conn.execute(
            'SELECT thread_id, COALESCE(SUM(LENGTH(checkpoint)), 0) '
            'FROM "checkpoints" GROUP BY thread_id'
        ):
            sizes[thread_id] = total or 0
    finally:
        conn.close()
    return dict(sorted(sizes.items(), key=lambda kv: -kv[1]))


def prune_fat_threads(db_path, max_thread_mb=None, keep=None,
                      active_thread_id=None) -> dict:
    """Trim every thread whose checkpoint blobs exceed ``max_thread_mb``.

    This closes the gap the other two sweeps leave. ``prune_checkpoints`` bounds how
    many threads survive but keeps them regardless of size; ``prune_thread_history``
    bounds size but only for the active thread. So a thread that is abandoned yet
    still inside the newest ``checkpoint_keep_threads`` is bounded by nothing, and
    one runaway session can hold gigabytes indefinitely.

    Deleting the *oldest* checkpoints of such a thread is not lossy in any way the
    REPL can observe: it only ever resumes the newest checkpoint, and the rolling
    summary and the journal hold what the user needs to remember. The newest
    checkpoint is always kept, so a resume still replays from a valid state.

    ``keep`` is how many checkpoints to leave in a fat thread (defaults to
    ``CHECKPOINT_KEEP_PER_THREAD``); the newest ``keep`` rows always survive. If the
    thread is *still* over budget after that — possible when a single snapshot is
    huge, since one row can carry a whole history — ``keep`` is halved and the trim
    repeated until it fits or one row remains, so the budget is a real ceiling
    rather than a row count that assumes rows are small.

    Returns ``{"threads", "rows", "trimmed"}`` where ``trimmed`` maps thread_id →
    rows deleted.
    """
    max_thread_mb = (CHECKPOINT_MAX_THREAD_MB if max_thread_mb is None
                     else max_thread_mb)
    keep = CHECKPOINT_KEEP_PER_THREAD if keep is None else keep
    result = {"threads": 0, "rows": 0, "trimmed": {}}
    if not max_thread_mb or keep <= 0:
        return result
    if not db_path or not os.path.exists(db_path):
        return result

    budget = max_thread_mb * 1024 * 1024
    fat = [
        (thread_id, size)
        for thread_id, size in thread_blob_bytes(db_path).items()
        if size > budget and thread_id != active_thread_id
    ]
    if not fat:
        return result

    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        tables = _thread_id_tables(conn)
        ranked = [row[1] for row in conn.execute('PRAGMA table_info("checkpoints")')]
        by_checkpoint_id = "checkpoints" in tables and "checkpoint_id" in ranked
        for thread_id, _size in fat:
            keep_n = keep
            rows = _trim_thread(conn, tables, thread_id, keep_n, by_checkpoint_id)
            # Size backstop: one snapshot can be huge, so a row count is not a
            # bound. Halve until the thread fits or a single row is left.
            while keep_n > 1 and _thread_bytes(conn, thread_id) > budget:
                keep_n = max(1, keep_n // 2)
                rows += _trim_thread(conn, tables, thread_id, keep_n, by_checkpoint_id)
            if rows:
                result["trimmed"][thread_id] = rows
                result["rows"] += rows
                result["threads"] += 1
                logger.info(
                    "housekeeping: trimmed %d checkpoint row(s) of abandoned thread "
                    "%s (kept %d)", rows, thread_id, keep_n,
                )
    finally:
        conn.close()
    return result


def prune_thread_history(db_path, active_thread_id=None, keep=None, max_mb=None,
                         vacuum=False) -> dict:
    """Bound the history of the *live* thread by keeping its newest checkpoints.

    ``prune_checkpoints`` removes abandoned threads, but the checkpointer stores a
    full snapshot of the entire message history on every super-step, so a single
    long session grows O(N^2) all by itself (measured: 180 MB in 50 rows, and 87%
    of a 10 GB store in one 18-hour session). Deleting the older rows of the live
    thread is safe — resuming replays from the newest checkpoint, and the rolling
    summary / journal hold what the user needs to remember.

    Only the active thread is touched, because an older thread being resumed is
    not something the REPL offers; trimming those would be lossy for no gain.

    ``max_mb`` is a backstop for a thread that is fat even when short: if the file
    still exceeds it after trimming to ``keep`` rows, ``keep`` is halved and the
    trim repeated. ``vacuum=True`` rewrites the DB afterwards to actually shrink
    the file — it takes an exclusive lock for the whole rewrite, so it must only
    be used when no session is live (see ``scripts/vacuum_checkpoints.py``).

    Returns ``{"rows", "bytes", "kept", "freed_mb"}``.
    """
    result = {"rows": 0, "bytes": 0, "kept": 0, "freed_mb": 0.0}
    if not db_path or not os.path.exists(db_path) or not active_thread_id:
        return result
    keep = CHECKPOINT_KEEP_PER_THREAD if keep is None else keep
    max_mb = CHECKPOINT_MAX_MB if max_mb is None else max_mb
    if keep <= 0:
        return result

    before = _size(db_path) + _size(db_path + "-wal")
    # Short timeout, like prune_checkpoints: the live session holds this DB, and
    # the sweep must never be the reason a turn waits.
    conn = sqlite3.connect(db_path, timeout=5.0)
    try:
        tables = _thread_id_tables(conn)
        if not tables:
            return result
        # Rank by rowid: the checkpointer schema stores no timestamp, and rowid
        # order is insertion order, which is what recency means here. Only the
        # checkpointer's own table has a checkpoint_id to rank by; a table this
        # module has never heard of is trimmed by rowid instead (below).
        ranked = [
            row[1] for row in conn.execute('PRAGMA table_info("checkpoints")')
        ]
        by_checkpoint_id = "checkpoints" in tables and "checkpoint_id" in ranked

        def _trim(keep_n: int) -> int:
            return _trim_thread(conn, tables, active_thread_id, keep_n,
                                by_checkpoint_id)

        result["rows"] = _trim(keep)
        result["kept"] = min(
            keep,
            conn.execute(
                'SELECT count(*) FROM "checkpoints" WHERE thread_id = ?',
                (active_thread_id,),
            ).fetchone()[0] if by_checkpoint_id else keep,
        )

        # Size backstop: a thread can be fat even when short (one snapshot of a
        # huge history). Halve the keep-count until it fits or one row is left.
        if max_mb:
            budget = max_mb * 1024 * 1024
            keep_n = keep
            while _size(db_path) > budget and keep_n > 1:
                keep_n = max(1, keep_n // 2)
                result["rows"] += _trim(keep_n)
                result["kept"] = keep_n

        if vacuum:
            try:
                conn.execute("VACUUM")
            except sqlite3.Error as e:      # never let reclamation break the sweep
                logger.info("housekeeping: skipped vacuum (%s)", e)
    finally:
        conn.close()

    after = _size(db_path) + _size(db_path + "-wal")
    result["bytes"] = max(0, before - after)
    result["freed_mb"] = result["bytes"] / (1024 * 1024)
    if result["rows"]:
        logger.info(
            "housekeeping: trimmed %d checkpoint row(s) of thread %s "
            "(kept %d, %.1f MB freed)",
            result["rows"], active_thread_id, result["kept"], result["freed_mb"],
        )
    return result


def sweep(scratch_dir, checkpoint_db, active_thread_id=None,
          checkpoint_vacuum=False) -> str:
    """Run all sweeps — scratch, checkpoints, and memory — log and return a one-line summary."""
    if not ENABLED:
        return "disabled"

    scratch = {"removed": 0, "bytes": 0}
    checkpoints = {"threads": 0, "rows": 0}
    history = {"rows": 0, "bytes": 0, "kept": 0, "freed_mb": 0.0}
    fat = {"threads": 0, "rows": 0, "trimmed": {}}
    memories = {"removed": 0, "kept": 0}
    try:
        scratch = prune_scratch(scratch_dir)
    except Exception as e:                   # never let a sweep break start-up  # noqa: BLE001
        logger.warning("housekeeping: scratch sweep failed: %s", e, exc_info=True)
    try:
        checkpoints = prune_checkpoints(checkpoint_db, active_thread_id=active_thread_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("housekeeping: checkpoint sweep failed: %s", e, exc_info=True)
    # Runs after the thread sweep, so it only looks at the threads that survived —
    # the gap it closes is precisely "abandoned but still recent enough to keep".
    try:
        fat = prune_fat_threads(checkpoint_db, active_thread_id=active_thread_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("housekeeping: fat-thread trim failed: %s", e, exc_info=True)
    try:
        history = prune_thread_history(
            checkpoint_db, active_thread_id=active_thread_id,
            vacuum=checkpoint_vacuum,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("housekeeping: thread-history trim failed: %s", e, exc_info=True)
    try:
        memories = prune_memories()
    except Exception as e:  # noqa: BLE001
        logger.warning("housekeeping: memory prune failed: %s", e, exc_info=True)

    summary = (
        f"{scratch['removed']} scratch entries "
        f"({scratch['bytes'] / (1024 * 1024):.1f} MB), "
        f"{checkpoints['threads']} threads ({checkpoints['rows']} rows)"
    )
    if history["rows"]:
        summary += (
            f", {history['rows']} checkpoint rows of this session "
            f"({history['freed_mb']:.1f} MB)"
        )
    if fat["rows"]:
        summary += (
            f", {fat['rows']} checkpoint rows from {fat['threads']} oversized "
            f"thread(s)"
        )
    if memories["removed"]:
        summary += (
            f", {memories['removed']} stale memories pruned "
            f"({memories['kept']} kept)"
        )
    return summary


def prune_memories() -> dict:
    """Remove stale low-confidence facts from the memory store.

    Only distilled facts (confidence < 1.0) older than ``memory.prune_age_days``
    are eligible; manual facts (/save) are always kept.  Low-confidence facts
    that have been recalled at least once are also kept — "never recalled" is a
    stronger signal of staleness than "recalled rarely".

    Returns a dict with ``removed`` and ``kept`` counts, or both zero when
    pruning is disabled or the store is empty.
    """
    from datetime import datetime, timezone, timedelta

    from .config import config as _cfg
    from .memory_store import (
        PRUNE_AGE_DAYS, PRUNE_CONFIDENCE_THRESHOLD,
        get_collection,
    )

    prune_age_days = _cfg.get("memory.prune_age_days", PRUNE_AGE_DAYS)
    if prune_age_days <= 0:
        return {"removed": 0, "kept": 0}

    collection = get_collection()
    total = collection.count()
    if total == 0:
        return {"removed": 0, "kept": 0}

    cutoff = (datetime.now(timezone.utc) - timedelta(days=prune_age_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    doomed = []
    limit = 1000
    offset = 0
    while offset < total:
        batch = collection.get(
            offset=offset,
            limit=limit,
            include=["metadatas"],
        )
        ids = batch.get("ids") or []
        metas = batch.get("metadatas") or []
        if not ids:
            break
        for mem_id, meta in zip(ids, metas):
            if meta is None:
                continue
            source = (meta.get("source") or "").strip()
            timestamp = (meta.get("timestamp") or "").strip()
            confidence_str = (meta.get("confidence") or "1.0").strip()
            try:
                confidence = float(confidence_str)
            except ValueError:
                confidence = 0.7

            if source in ("manual", "supabase", ""):
                continue
            if confidence >= PRUNE_CONFIDENCE_THRESHOLD:
                continue
            if not timestamp or timestamp >= cutoff:
                continue
            doomed.append(mem_id)
        offset += limit
        if not ids or len(ids) < limit:
            break

    if doomed:
        collection.delete(ids=doomed)
        logger.info(
            "housekeeping: pruned %d stale memory fact(s) "
            "(older than %d days, confidence < %.1f)",
            len(doomed), prune_age_days, PRUNE_CONFIDENCE_THRESHOLD,
        )

    return {"removed": len(doomed), "kept": total - len(doomed)}
