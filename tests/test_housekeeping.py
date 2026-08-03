"""Tests for housekeeping — the start-up sweep that reclaims disk.

The risk being tested is asymmetric: failing to delete costs disk, but deleting
the wrong thing costs the user their state. So most of these assert what survives.
"""

import os
import sqlite3
import time

from components import housekeeping

DAY = 86400.0


def _write(directory, name, content="x", age_days=0.0, now=None):
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    if age_days:
        when = (time.time() if now is None else now) - age_days * DAY
        os.utime(path, (when, when))
    return path


def _checkpoint_db(path, threads):
    """A checkpointer-shaped DB: the two tables LangGraph's SqliteSaver creates."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
    conn.execute("CREATE TABLE writes (thread_id TEXT, channel TEXT)")
    for thread_id in threads:                # insertion order == recency (rowid)
        conn.execute("INSERT INTO checkpoints VALUES (?, ?)", (thread_id, "c"))
        conn.execute("INSERT INTO writes VALUES (?, ?)", (thread_id, "messages"))
    conn.commit()
    conn.close()
    return path


def _threads(path):
    conn = sqlite3.connect(path)
    try:
        return {row[0] for row in conn.execute("SELECT thread_id FROM checkpoints")}
    finally:
        conn.close()


class TestPruneScratch:
    def test_old_entries_go_and_recent_ones_stay(self, tmp_path):
        old = _write(tmp_path, "doc_old.txt", age_days=30)
        fresh = _write(tmp_path, "doc_new.txt")
        result = housekeeping.prune_scratch(str(tmp_path), max_age_days=7)
        assert result["removed"] == 1 and result["kept"] == 1
        assert not os.path.exists(old)
        assert os.path.exists(fresh)

    def test_size_cap_evicts_oldest_first(self, tmp_path):
        # Three 1 MB entries, all recent, with a 2 MB budget: the oldest goes.
        paths = [_write(tmp_path, f"doc_{i}.txt", content="x" * (1024 * 1024),
                        age_days=3 - i) for i in range(3)]
        result = housekeeping.prune_scratch(
            str(tmp_path), max_age_days=7, max_total_mb=2
        )
        assert result["removed"] == 1
        assert not os.path.exists(paths[0])
        assert all(os.path.exists(p) for p in paths[1:])

    def test_age_cap_alone_never_evicts_for_size(self, tmp_path):
        big = _write(tmp_path, "doc_big.txt", content="x" * (1024 * 1024))
        housekeeping.prune_scratch(str(tmp_path), max_age_days=7, max_total_mb=0)
        assert os.path.exists(big)

    def test_bytes_freed_is_reported(self, tmp_path):
        _write(tmp_path, "doc_old.txt", content="y" * 2048, age_days=30)
        result = housekeeping.prune_scratch(str(tmp_path), max_age_days=7)
        assert result["bytes"] == 2048

    def test_missing_directory_is_not_an_error(self, tmp_path):
        result = housekeeping.prune_scratch(str(tmp_path / "nope"))
        assert result == {"removed": 0, "bytes": 0, "kept": 0}

    def test_subdirectories_are_left_alone(self, tmp_path):
        nested = tmp_path / "sub"
        nested.mkdir()
        os.utime(nested, (time.time() - 90 * DAY,) * 2)
        housekeeping.prune_scratch(str(tmp_path), max_age_days=7)
        assert nested.is_dir()


class TestPruneCheckpoints:
    def test_keeps_the_newest_threads(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), ["t1", "t2", "t3", "t4"])
        result = housekeeping.prune_checkpoints(db, keep_threads=2)
        assert result["threads"] == 2
        assert _threads(db) == {"t3", "t4"}

    def test_active_thread_survives_even_when_oldest(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), ["old", "t2", "t3"])
        housekeeping.prune_checkpoints(db, keep_threads=1, active_thread_id="old")
        assert _threads(db) == {"old", "t3"}

    def test_every_thread_id_table_is_swept(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), ["t1", "t2"])
        housekeeping.prune_checkpoints(db, keep_threads=1)
        conn = sqlite3.connect(db)
        try:
            assert conn.execute(
                "SELECT count(*) FROM writes WHERE thread_id = 't1'"
            ).fetchone()[0] == 0
        finally:
            conn.close()

    def test_nothing_to_prune_leaves_the_db_alone(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), ["t1", "t2"])
        result = housekeeping.prune_checkpoints(db, keep_threads=20)
        assert result == {"threads": 0, "rows": 0}
        assert _threads(db) == {"t1", "t2"}

    def test_missing_db_is_not_an_error(self, tmp_path):
        result = housekeeping.prune_checkpoints(str(tmp_path / "nope.db"))
        assert result == {"threads": 0, "rows": 0}

    def test_unknown_schema_without_thread_id_is_left_alone(self, tmp_path):
        db = str(tmp_path / "other.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE something (id INTEGER)")
        conn.execute("INSERT INTO something VALUES (1)")
        conn.commit()
        conn.close()
        assert housekeeping.prune_checkpoints(db, keep_threads=0) == {
            "threads": 0, "rows": 0,
        }
        conn = sqlite3.connect(db)
        try:
            assert conn.execute("SELECT count(*) FROM something").fetchone()[0] == 1
        finally:
            conn.close()


class TestSweep:
    def test_reports_both_halves(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir(exist_ok=True)
        _write(scratch, "doc_old.txt", content="z" * 1024, age_days=30)
        db = _checkpoint_db(str(tmp_path / "ck.db"), ["t1", "t2"])
        summary = housekeeping.sweep(str(scratch), db, active_thread_id="t2")
        assert "1 scratch entries" in summary
        assert "threads" in summary

    def test_a_broken_half_does_not_take_down_the_sweep(self, tmp_path, monkeypatch):
        def boom(*_args, **_kwargs):
            raise OSError("disk on fire")

        monkeypatch.setattr(housekeeping, "prune_scratch", boom)
        db = _checkpoint_db(str(tmp_path / "ck.db"), ["t1", "t2", "t3"])
        summary = housekeeping.sweep(str(tmp_path), db, active_thread_id="t3")
        assert "0 scratch entries" in summary

    def test_disabled_does_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(housekeeping, "ENABLED", False)
        old = _write(tmp_path, "doc_old.txt", age_days=90)
        assert housekeeping.sweep(str(tmp_path), None) == "disabled"
        assert os.path.exists(old)
