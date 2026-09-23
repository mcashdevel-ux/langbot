# Situation Report — checkpoint DB "database is locked"

**Date:** 2026-09-23 09:47 UTC
**Repo:** `~/ai/repos/langbot` @ `8b465b3` (main)
**Severity:** High — turns fail and are not persisted; ~30 GB of disk consumed
**Status:** Lock cause fixed (uncommitted). Growth cause identified, **not fixed**.

---

## 1. Symptom

`sqlite3.OperationalError: database is locked`, logged as
`ERROR __main__: Error while processing turn`. The turn dies and its state is
not saved. Observed 3 times:

| When | Log line |
|---|---|
| 2026-09-22 21:18:32 | `memory/langbot.log:9728` |
| 2026-09-23 08:34:42 | `memory/langbot.log:9809` |
| 2026-09-23 15:22:26 | `memory/langbot.log:10193` |

Origin: `langgraph/checkpoint/sqlite/__init__.py:425`, the
`INSERT OR REPLACE INTO checkpoints` in `SqliteSaver.put()`.

## 2. Root cause A — the start-up sweep ran a full `VACUUM` (FIXED)

`components/housekeeping.py:151` issued a full `VACUUM` on the checkpoint DB
during the start-up sweep. A full VACUUM rewrites the entire database into a
temp file while holding an **exclusive lock for the whole duration**.

Verified live: the `langbot-warmup` thread was blocked in
`ext4_block_write_begin` with a 10.4 GB temp file open at
`/var/tmp/etilqs_117a4f43de52d518 (deleted)` — a full copy of the DB. It ran
for minutes; every checkpointer write in that window failed with
"database is locked". The old comment claimed lock contention "is not a failure
worth reporting", but the VACUUM *causes* the contention.

**Fix:** replaced the full `VACUUM` with `PRAGMA incremental_vacuum`
(freelist-only, no long exclusive lock, no-op unless `auto_vacuum=INCREMENTAL`).
Offline reclamation moved to `scripts/vacuum_checkpoints.py`, which refuses to
run unless it can take the write lock immediately.

## 3. Root cause B — O(N²) checkpoint growth (NOT FIXED)

LangGraph's SQLite checkpointer stores a **full snapshot of the entire message
history on every super-step** — it is not incremental. Every tool-call round
rewrites the whole conversation.

Measured on `session_20260922_023236` (the 18-hour session):

| Checkpoint | Messages | Blob size |
|---|---|---|
| #0 | 1 | 0.001 MB |
| #433 | 259 | 0.622 MB |
| #2169 | 1144 | 2.186 MB |
| #4339 | 2240 | 3.621 MB |

Sum of a linear ramp over N rows = N × max / 2 = 4340 × 3.62 MB / 2 ≈ 7.9 GB
predicted, **9.0 GB actual** — the O(N²) signature. A long session costs O(N²),
not O(N).

DB totals: 21 threads, 8,587 checkpoint rows, **10.35 GB** of checkpoint blobs,
of which **9.0 GB (87%) is that one session**.

## 4. Root cause C — WAL pinned by a stale process (NOT FIXED)

`memory/agent_checkpoints.db-wal` is **20 GB** and cannot be reclaimed.
`PRAGMA wal_checkpoint(TRUNCATE)` returns `(1, -1, -1)` — blocked.

Cause: PID **613741**, a `python langbot.py` started **2026-09-22 21:11** on
`pts/4` (SSH `100.112.9.124`), still holding the DB, WAL and SHM open with a
read transaction. In WAL mode every write appends a new version; the WAL is only
reclaimed at checkpoint, so it holds every version since that process started
and can never be truncated while it lives. It is idle (55s CPU total, no I/O).

## 5. Current state

| Item | Value |
|---|---|
| `agent_checkpoints.db` | 10.5 GB |
| `agent_checkpoints.db-wal` | 21.1 GB |
| Disk | 152G total, 60G used, 85G free (42%) |
| PID 613741 (stale, pins WAL) | alive, 18h35m |
| PID 1274407 (current session) | alive, 29m |

## 6. Changes made (uncommitted at time of writing)

- `components/housekeeping.py` — sweep no longer VACUUMs; uses
  `PRAGMA incremental_vacuum`.
- `tests/test_housekeeping.py` — `test_never_vacuums` regression test.
  **16 passed.**
- `scripts/vacuum_checkpoints.py` — new offline reclamation tool.

## 7. Recommended next steps

1. **Reclaim disk now:** `kill 613741 1274407` then
   `python scripts/vacuum_checkpoints.py`.
2. **Bound the growth (the real fix):** add intra-thread checkpoint pruning to
   the sweep — keep the last K checkpoints per thread. The current sweep only
   prunes whole threads, so a single live session can still reach ~9 GB.
3. Consider capping session length / auto-`/new` after N turns, or a
   delta-storing checkpointer.
