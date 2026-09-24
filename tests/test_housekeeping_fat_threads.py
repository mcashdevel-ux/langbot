"""Tests for the per-thread size bound (``prune_fat_threads``).

The gap this closes is narrow and was measured live: a thread that is *abandoned*
but still inside the newest ``checkpoint_keep_threads`` is bounded by neither
existing sweep. ``prune_checkpoints`` keeps it because it is recent enough, and
``prune_thread_history`` never looks at it because it is not the active thread. So
one runaway session can hold gigabytes indefinitely — 9 GB (87% of a 10 GB store)
in the 18-hour session recorded in docs/sitrep-2026-09-23-checkpoint-db-lock.md,
and 263 MB in a thread observed in a live DB here.

The asymmetry is the usual one: trimming too little costs disk, trimming too much
costs the user their conversation. So the newest checkpoint must always survive.
"""

import os
import sqlite3

from components import housekeeping


def _checkpoint_db(path, threads):
    """A checkpointer-shaped DB.

    ``threads`` maps thread_id -> number of checkpoints. Each row's blob grows with
    its index, which is what makes the real store O(N^2): the checkpointer writes a
    full snapshot of the whole message history on every super-step.
    """
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, "
        "checkpoint_id TEXT, parent_checkpoint_id TEXT, type TEXT, "
        "checkpoint BLOB, metadata BLOB, PRIMARY KEY (thread_id, checkpoint_ns, "
        "checkpoint_id))"
    )
    conn.execute(
        "CREATE TABLE writes (thread_id TEXT, checkpoint_ns TEXT, "
        "checkpoint_id TEXT, task_id TEXT, idx INTEGER, channel TEXT, type TEXT, "
        "value BLOB, PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, "
        "task_id, idx))"
    )
    for thread_id, rows in threads.items():
        for i in range(rows):
            cid = f"cp{i:04d}"
            conn.execute(
                "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
                (thread_id, "", cid, None, "json", b"x" * (100 * (i + 1)), b"{}"),
            )
            conn.execute(
                "INSERT INTO writes VALUES (?,?,?,?,?,?,?,?)",
                (thread_id, "", cid, "t", 0, "messages", "json", b"y" * 10),
            )
    conn.commit()
    conn.close()
    return path


def _ids(path, thread_id):
    conn = sqlite3.connect(path)
    try:
        return [
            row[0] for row in conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ? "
                "ORDER BY rowid", (thread_id,),
            )
        ]
    finally:
        conn.close()


def _assert_bounded(db, thread_id, budget_mb):
    """The documented guarantee: the thread fits the budget, or is down to one row.

    A single snapshot can be larger than the budget on its own, so "fits" cannot be
    absolute — one row is the floor, and that row is the newest, which is the one a
    resume replays from.
    """
    blob = housekeeping.thread_blob_bytes(db)[thread_id]
    if blob > budget_mb * 1024 * 1024:
        assert len(_ids(db, thread_id)) == 1


class TestThreadBlobBytes:
    def test_reports_each_thread_biggest_first(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"small": 2, "big": 50})
        sizes = housekeeping.thread_blob_bytes(db)
        assert list(sizes) == ["big", "small"]
        assert sizes["big"] > sizes["small"]

    def test_missing_db_is_empty(self, tmp_path):
        assert housekeeping.thread_blob_bytes(str(tmp_path / "nope.db")) == {}


class TestPruneFatThreads:
    def test_trims_an_abandoned_thread_over_budget(self, tmp_path):
        # The exact failure: fat, not active, recent enough to survive the thread
        # sweep — so neither existing sweep would ever touch it.
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 200, "live": 3})
        result = housekeeping.prune_fat_threads(
            db, max_thread_mb=0.01, keep=5, active_thread_id="live",
        )
        assert result["threads"] == 1
        assert result["trimmed"]["fat"] > 0
        # The budget is a real ceiling, not a row count that assumes rows are small:
        # 200 rows of growing blobs are ~2 MB, far over the 0.01 MB budget, so the
        # halving backstop drives it down to a single row.
        _assert_bounded(db, "fat", 0.01)
        assert _ids(db, "fat")[-1] == "cp0199"  # newest survives

    def test_the_newest_checkpoint_always_survives(self, tmp_path):
        # The one invariant a resume depends on.
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 100})
        housekeeping.prune_fat_threads(
            db, max_thread_mb=0.001, keep=1, active_thread_id="live",
        )
        assert _ids(db, "fat") == ["cp0099"]

    def test_threads_under_budget_are_untouched(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"small": 5, "live": 3})
        result = housekeeping.prune_fat_threads(
            db, max_thread_mb=10, keep=1, active_thread_id="live",
        )
        assert result == {"threads": 0, "rows": 0, "trimmed": {}}
        assert len(_ids(db, "small")) == 5

    def test_the_active_thread_is_left_to_prune_thread_history(self, tmp_path):
        # Two functions trimming the same thread on the same pass would fight;
        # the active thread has its own sweep with its own (halving) backstop.
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"live": 200})
        result = housekeeping.prune_fat_threads(
            db, max_thread_mb=0.01, keep=5, active_thread_id="live",
        )
        assert result["rows"] == 0
        assert len(_ids(db, "live")) == 200

    def test_writes_are_trimmed_with_their_checkpoints(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 50})
        housekeeping.prune_fat_threads(
            db, max_thread_mb=0.001, keep=2, active_thread_id="live",
        )
        conn = sqlite3.connect(db)
        try:
            left = {
                row[0] for row in conn.execute(
                    "SELECT checkpoint_id FROM writes WHERE thread_id = 'fat'"
                )
            }
        finally:
            conn.close()
        # A write whose checkpoint is gone is unreachable state; it must not stay.
        # Whatever survives in `writes` must correspond to a surviving checkpoint.
        assert left == set(_ids(db, "fat"))
        assert left

    def test_only_the_oversized_threads_are_touched(self, tmp_path):
        db = _checkpoint_db(
            str(tmp_path / "ck.db"), {"fat": 100, "ok": 4, "tiny": 1}
        )
        result = housekeeping.prune_fat_threads(
            db, max_thread_mb=0.05, keep=2, active_thread_id="live",
        )
        assert result["trimmed"] == {"fat": 196}
        assert len(_ids(db, "ok")) == 4
        assert len(_ids(db, "tiny")) == 1

    def test_zero_budget_disables_the_pass(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 100})
        result = housekeeping.prune_fat_threads(
            db, max_thread_mb=0, keep=2, active_thread_id="live",
        )
        assert result["rows"] == 0
        assert len(_ids(db, "fat")) == 100

    def test_keep_zero_disables_the_pass(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 100})
        result = housekeeping.prune_fat_threads(
            db, max_thread_mb=0.001, keep=0, active_thread_id="live",
        )
        assert result["rows"] == 0
        assert len(_ids(db, "fat")) == 100

    def test_missing_db_is_not_an_error(self, tmp_path):
        assert housekeeping.prune_fat_threads(
            str(tmp_path / "nope.db"), max_thread_mb=1, active_thread_id="live",
        ) == {"threads": 0, "rows": 0, "trimmed": {}}

    def test_unknown_schema_is_left_alone(self, tmp_path):
        db = str(tmp_path / "other.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE something (id INTEGER)")
        conn.execute("INSERT INTO something VALUES (1)")
        conn.commit()
        conn.close()
        assert housekeeping.prune_fat_threads(
            db, max_thread_mb=1, active_thread_id="live",
        ) == {"threads": 0, "rows": 0, "trimmed": {}}


class TestSweepBoundsOneThread:
    def test_sweep_trims_a_fat_abandoned_thread(self, tmp_path, monkeypatch):
        # End to end: with the other sweeps as they are, a fat abandoned thread
        # survives them — this is the pass that catches it.
        monkeypatch.setattr(housekeeping, "CHECKPOINT_MAX_THREAD_MB", 0.01)
        scratch = tmp_path / "scratch"
        scratch.mkdir(exist_ok=True)
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 200, "live": 3})
        summary = housekeeping.sweep(
            str(scratch), db, active_thread_id="live",
        )
        assert "oversized thread" in summary
        _assert_bounded(db, "fat", 0.01)
        assert _ids(db, "fat")[-1] == "cp0199"  # newest survives

    def test_the_abandoned_thread_survives_the_other_sweeps_alone(
        self, tmp_path, monkeypatch
    ):
        # Guards the premise: without the fat-thread pass, nothing bounds it.
        monkeypatch.setattr(housekeeping, "CHECKPOINT_MAX_THREAD_MB", 0)
        scratch = tmp_path / "scratch"
        scratch.mkdir(exist_ok=True)
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 200, "live": 3})
        housekeeping.sweep(str(scratch), db, active_thread_id="live")
        assert len(_ids(db, "fat")) == 200

    def test_a_broken_fat_trim_does_not_take_down_the_sweep(
        self, tmp_path, monkeypatch
    ):
        def boom(*_a, **_k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(housekeeping, "prune_fat_threads", boom)
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 5, "live": 2})
        summary = housekeeping.sweep(str(tmp_path), db, active_thread_id="live")
        assert "scratch entries" in summary

    def test_trimming_does_not_shrink_the_file_without_a_vacuum(self, tmp_path):
        # Documents the second half of the problem: deleting rows frees pages
        # inside the file, so reclamation is still an offline step.
        db = _checkpoint_db(str(tmp_path / "ck.db"), {"fat": 400})
        before = os.path.getsize(db)
        housekeeping.prune_fat_threads(
            db, max_thread_mb=0.001, keep=1, active_thread_id="live",
        )
        assert os.path.getsize(db) >= before * 0.9
