"""Tests for the intra-thread checkpoint history trim.

``prune_checkpoints`` bounds how many *threads* survive; this bounds how large one
thread can get. The failure it prevents is O(N^2) growth inside a single long
session, because the LangGraph checkpointer writes a full snapshot of the whole
message history on every super-step.

The asymmetry under test is the same as for the rest of housekeeping: trimming too
little costs disk, trimming too much costs the user their conversation. So the
newest checkpoint must always survive — that is the one a resume replays from.
"""

import sqlite3

from components import housekeeping


def _checkpoint_db(path, thread_rows, other_thread="abandoned"):
    """A checkpointer-shaped DB: ``thread_rows`` checkpoints for the live thread.

    Each row's blob grows linearly, which is what makes the real store O(N^2).
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
    for i in range(thread_rows):             # insertion order == rowid == recency
        cid = f"cp{i:04d}"
        conn.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
            ("live", "", cid, None, "json", b"x" * (100 * (i + 1)), b"{}"),
        )
        conn.execute(
            "INSERT INTO writes VALUES (?,?,?,?,?,?,?,?)",
            ("live", "", cid, "t", 0, "messages", "json", b"y" * 10),
        )
    if other_thread:
        conn.execute(
            "INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?)",
            (other_thread, "", "cpX", None, "json", b"z" * 500, b"{}"),
        )
        conn.execute(
            "INSERT INTO writes VALUES (?,?,?,?,?,?,?,?)",
            (other_thread, "", "cpX", "t", 0, "messages", "json", b"y"),
        )
    conn.commit()
    conn.close()
    return path


def _ids(path, thread_id="live"):
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


class TestPruneThreadHistory:
    def test_keeps_only_the_newest_checkpoints(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 12)
        result = housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=3, max_mb=0,
        )
        assert _ids(db) == ["cp0009", "cp0010", "cp0011"]
        # rows counts every table, so 9 checkpoints plus their 9 write rows.
        assert result["rows"] == 18
        assert result["kept"] == 3

    def test_newest_checkpoint_always_survives(self, tmp_path):
        # The one invariant a resume depends on, checked at every keep-count.
        for keep in (1, 2, 5, 11):
            db = _checkpoint_db(str(tmp_path / f"ck{keep}.db"), 12)
            housekeeping.prune_thread_history(
                db, active_thread_id="live", keep=keep, max_mb=0,
            )
            assert _ids(db)[-1] == "cp0011"
            assert len(_ids(db)) == min(keep, 12)

    def test_writes_are_trimmed_with_their_checkpoints(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 10)
        housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=2, max_mb=0,
        )
        conn = sqlite3.connect(db)
        try:
            left = {
                row[0] for row in conn.execute(
                    "SELECT checkpoint_id FROM writes WHERE thread_id = 'live'"
                )
            }
        finally:
            conn.close()
        # A write whose checkpoint is gone is unreachable state; it must not stay.
        assert left == {"cp0008", "cp0009"}

    def test_other_threads_are_untouched(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 10)
        housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=2, max_mb=0,
        )
        assert _ids(db, "abandoned") == ["cpX"]

    def test_no_active_thread_is_a_no_op(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 10)
        assert housekeeping.prune_thread_history(db, active_thread_id=None) == {
            "rows": 0, "bytes": 0, "kept": 0, "freed_mb": 0.0,
        }
        assert len(_ids(db)) == 10

    def test_keep_zero_disables_the_trim(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 10)
        housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=0, max_mb=0,
        )
        assert len(_ids(db)) == 10

    def test_missing_db_is_not_an_error(self, tmp_path):
        assert housekeeping.prune_thread_history(
            str(tmp_path / "nope.db"), active_thread_id="live"
        ) == {"rows": 0, "bytes": 0, "kept": 0, "freed_mb": 0.0}

    def test_short_history_is_left_alone(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 3)
        result = housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=20, max_mb=0,
        )
        assert result["rows"] == 0
        assert len(_ids(db)) == 3

    def test_size_backstop_halves_until_it_fits(self, tmp_path):
        # 400 rows of growing blobs are ~8 MB; a 1 MB budget forces extra halving.
        db = _checkpoint_db(str(tmp_path / "ck.db"), 400)
        result = housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=400, max_mb=1,
        )
        assert result["kept"] < 400
        assert _ids(db)[-1] == "cp0399"

    def test_schema_without_checkpoint_id_falls_back_to_rowid(self, tmp_path):
        db = str(tmp_path / "odd.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE events (thread_id TEXT, kind TEXT)")
        for i in range(10):
            conn.execute("INSERT INTO events VALUES ('live', ?)", (f"e{i}",))
        conn.commit()
        conn.close()
        housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=4, max_mb=0,
        )
        conn = sqlite3.connect(db)
        try:
            kinds = [r[0] for r in conn.execute("SELECT kind FROM events")]
        finally:
            conn.close()
        assert kinds == ["e6", "e7", "e8", "e9"]

    def test_vacuum_shrinks_the_file(self, tmp_path):
        db = _checkpoint_db(str(tmp_path / "ck.db"), 200)
        import os
        before = os.path.getsize(db)
        housekeeping.prune_thread_history(
            db, active_thread_id="live", keep=2, max_mb=0, vacuum=True,
        )
        # Without a VACUUM the freed pages stay in the file.
        assert os.path.getsize(db) < before


class TestSweepReportsHistory:
    def test_summary_mentions_the_trim(self, tmp_path):
        scratch = tmp_path / "scratch"
        scratch.mkdir(exist_ok=True)
        db = _checkpoint_db(str(tmp_path / "ck.db"), 30)
        summary = housekeeping.sweep(
            str(scratch), db, active_thread_id="live",
        )
        assert "checkpoint rows of this session" in summary
        assert len(_ids(db)) == housekeeping.CHECKPOINT_KEEP_PER_THREAD

    def test_a_broken_trim_does_not_take_down_the_sweep(self, tmp_path, monkeypatch):
        def boom(*_a, **_k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(housekeeping, "prune_thread_history", boom)
        db = _checkpoint_db(str(tmp_path / "ck.db"), 5)
        summary = housekeeping.sweep(str(tmp_path), db, active_thread_id="live")
        assert "scratch entries" in summary
