# Situation Report — checkpoint DB growth, follow-up

**Date:** 2026-09-24 09:05 +06
**Repo:** `~/ai/repos/langbot` @ `b524b6e` (main) + uncommitted work
**Severity:** resolved for new sessions; the historical DB is back to a sane size
**Follows:** `docs/sitrep-2026-09-23-checkpoint-db-lock.md`

---

## 1. What was still open

The 2026-09-23 sitrep closed root cause A (the start-up sweep ran a full `VACUUM`)
and left two open:

- **B — O(N²) growth inside one thread.** The LangGraph `SqliteSaver` stores a full
  snapshot of the entire message history on every super-step. The sweep only pruned
  whole *threads*, so a single long session was unbounded: `session_20260922_023236`
  held 180 MB in 50 rows and accounted for 87% of a 10 GB store.
- **C — the WAL could not be reclaimed while a session held the DB.**

## 2. What changed

`components/housekeeping.py` gained `prune_thread_history()`, and `sweep()` now runs
it after the thread sweep:

| New setting | Default | Effect |
|---|---|---|
| `checkpoint_keep_per_thread` | `20` | checkpoints kept in the **active** thread; older snapshots are deleted |
| `checkpoint_max_mb` | `1024` | backstop — a thread still over budget halves its history until it fits |
| `checkpoint_vacuum_on_start` | `true` | `VACUUM` once at start-up, before the checkpointer is busy |

Design notes:

- **Only the active thread is trimmed.** Resuming an older thread is not something the
  REPL offers, so trimming those would be lossy for no gain. Deleting the *oldest*
  snapshots of the live thread is safe: a resume replays from the newest checkpoint.
- **`writes` rows are deleted with their checkpoint** (`checkpoint_id NOT IN (kept)`),
  otherwise a write whose checkpoint is gone is unreachable state. A table with a
  `thread_id` but no `checkpoint_id` (schema drift) is trimmed by `rowid` instead, so
  the same "sweep every table carrying thread state" rule as `prune_checkpoints` holds.
- **`VACUUM` is opt-in and start-up only.** It takes the write lock for the whole
  rewrite, which is exactly what caused the original "database is locked" failures. It
  is safe on the warmup thread (it runs before `main()` opens the checkpointer), and it
  is required for the file to actually shrink — deleting rows only frees pages inside
  the file. A running session's overhead is reclaimed offline by
  `scripts/vacuum_checkpoints.py`, which now also takes `--thread <id> --keep N`.

## 3. Verified live

Ran `prune_checkpoints(keep_threads=1, active_thread_id=session_20260924_025528)` on the
real `memory/agent_checkpoints.db`, with a session live and holding the DB:

| | before | after |
|---|---|---|
| threads | 21 | 1 |
| checkpoint rows | 2,693 | 159 |
| blob bytes | 365 MB | 19.5 MB |
| file | 379 MB | 382 MB (freed pages inside the file) |

`PRAGMA quick_check` → `ok`. `PRAGMA wal_checkpoint(TRUNCATE)` then succeeded (`0,0,0`)
and the WAL went from 367 MB back to 0 — the trim released enough WAL frames that the
live reader no longer pinned the whole log. The 360 MB of freed pages stays inside the
file until the next start-up `VACUUM` (or the offline script).

Tests: `tests/test_housekeeping.py` + `tests/test_housekeeping_history.py` — 29 passed.
Full suite green.

## 4. Residual

- **Historical file size.** `agent_checkpoints.db` is still ~380 MB with ~360 MB of
  freelist pages; the next `langbot.py` start reclaims it. Run
  `python scripts/vacuum_checkpoints.py` if you would rather not wait.
- **Not the same as context compaction.** `context_budget` caps what is *sent* to the
  model; this caps what is *stored*. They are independent, and a long session needs both.
- The dead-session deletion above used `keep_threads=1` deliberately (to hand back
  disk); the shipped default is 20.
