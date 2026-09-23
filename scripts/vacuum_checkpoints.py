#!/usr/bin/env python3
"""Reclaim disk from the LangGraph checkpoint DB — offline only.

The start-up sweep (components/housekeeping.py) deletes abandoned threads but
deliberately never VACUUMs: a full VACUUM rewrites the whole database into a
temp file under an exclusive lock, which on a multi-GB checkpoint DB means
minutes of "database is locked" for every live session.

So reclaiming the space is a manual, offline operation. Run this only when no
langbot process is running:

    pgrep -f 'python langbot.py' && echo 'stop langbot first'
    python scripts/vacuum_checkpoints.py

It refuses to run if it cannot take the write lock immediately, so it can never
be the reason a live session stalls.
"""

import argparse
import os
import sqlite3
import sys

DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "memory",
    "agent_checkpoints.db",
)


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def vacuum(db_path: str, timeout: float = 0.0) -> int:
    """VACUUM ``db_path``. Returns bytes freed. Raises if the DB is busy."""
    before = _size(db_path) + _size(db_path + "-wal")
    conn = sqlite3.connect(db_path, timeout=timeout)
    try:
        # Fail fast rather than block a live session: if anyone holds the write
        # lock, this is not the moment to vacuum.
        conn.execute("BEGIN IMMEDIATE")
        conn.rollback()
        conn.execute("VACUUM")
    finally:
        conn.close()
    after = _size(db_path) + _size(db_path + "-wal")
    return before - after


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", nargs="?", default=DEFAULT_DB, help="checkpoint DB path")
    args = ap.parse_args()

    if not os.path.exists(args.db):
        print(f"no such database: {args.db}", file=sys.stderr)
        return 1

    try:
        freed = vacuum(args.db)
    except sqlite3.OperationalError as e:
        print(
            f"refusing to vacuum: {e}\n"
            "A langbot session is probably running. Stop it first.",
            file=sys.stderr,
        )
        return 2

    print(f"freed {freed / (1024 * 1024):.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
