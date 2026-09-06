"""Unit tests for components/journal.py — durable journaled sessions (Track B).

Ported from neo's journal tests (adapted to langbot's config-driven sessions
dir and LANGBOT_SESSION_ENCRYPT env).
"""

import json
import os
import time

import pytest

import components.journal as journal_mod
from components.journal import (
    Event,
    EventLog,
    Journal,
    heuristic_title,
    resolve_session_key,
)


@pytest.fixture(autouse=True)
def sessions_dir(tmp_path, monkeypatch):
    """Redirect the journal root into a temp dir for every test."""
    d = tmp_path / "sessions"
    d.mkdir()
    monkeypatch.setattr(journal_mod, "journals_dir", lambda: d)
    return d


class TestEventLog:
    def test_append_read_roundtrip(self, tmp_path):
        log = EventLog(tmp_path / "e.jsonl")
        e1 = log.append(Event(type="user_message", data={"text": "hi"}, source="user"))
        e2 = log.append(Event(type="llm_response", data={"content": "hello"}, source="agent"))
        assert e1.seq == 1
        assert e2.seq == 2
        log.close()

        log2 = EventLog(tmp_path / "e.jsonl")
        evs = log2.read_all()
        assert [e.type for e in evs] == ["user_message", "llm_response"]
        assert evs[0].data == {"text": "hi"}
        assert evs[0].source == "user"
        assert evs[1].seq == 2

    def test_torn_tail_truncated_on_load(self, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text('{"type": "a", "seq": 1}\n'
                     '{"type": "b", "seq": 2}\n'
                     '{"type": "c", "seq": 3}\n'
                     '{"type": "par')
        log = EventLog(p)
        assert log._seq == 3
        evs = log.read_all()
        assert [e.type for e in evs] == ["a", "b", "c"]
        assert p.read_text().endswith("\n")

    def test_seq_recovers_after_reopen(self, tmp_path):
        p = tmp_path / "e.jsonl"
        EventLog(p).append(Event(type="a"))
        log2 = EventLog(p)
        e = log2.append(Event(type="b"))
        assert e.seq == 2

    def test_unparseable_lines_skipped(self, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text('{"type": "a", "seq": 1}\ngarbage\n{"type": "b", "seq": 2}\n')
        log = EventLog(p)
        evs = log.read_all()
        assert [e.type for e in evs] == ["a", "b"]

    def test_unknown_event_types_are_forward_tolerant(self, tmp_path):
        p = tmp_path / "e.jsonl"
        p.write_text('{"type": "future_thing", "seq": 1, "data": {"x": 1}}\n')
        log = EventLog(p)
        evs = log.read_all()
        assert len(evs) == 1
        assert evs[0].type == "future_thing"


class TestJournal:
    def test_create_load_roundtrip(self, sessions_dir):
        j = Journal.create(name="session:t1")
        assert j.id
        assert (sessions_dir / j.id / "meta.json").exists()
        j.log.append(Event(type="user_message", data={"text": "hi"}, source="user"))
        j.set_title("Hello world")
        j.release()

        j2 = Journal.load(j.id)
        assert j2.meta()["title"] == "Hello world"
        assert j2.meta()["name"] == "session:t1"
        evs = j2.events()
        assert evs[0].type == "session_start"
        assert evs[1].type == "user_message"

    def test_list_sorts_by_activity_desc(self, sessions_dir):
        j1 = Journal.create(name="a")
        j2 = Journal.create(name="b")
        j1.log.append(Event(type="user_message", data={"text": "x"}, source="user"))

        # Force distinct mtimes — filesystem timestamp granularity varies, so
        # relying on real write timing would be flaky.

        now = time.time()
        os.utime(sessions_dir / j1.id / "events.jsonl", (now, now))
        os.utime(sessions_dir / j2.id / "events.jsonl", (now - 10, now - 10))

        metas = Journal.list()
        assert metas[0]["id"] == j1.id   # j1 touched most recently
        assert metas[1]["id"] == j2.id

    def test_list_skips_corrupt_meta_without_crashing(self, sessions_dir):
        j = Journal.create(name="a")
        (sessions_dir / j.id / "meta.json").write_text("{not json")
        metas = Journal.list()
        assert all(m["id"] != j.id for m in metas)
        assert isinstance(metas, list)

    def test_acquire_release_lock(self, sessions_dir):
        j = Journal.create(name="a")
        assert j.acquire() is True
        # Same-process second acquire: the lock is legitimately held → False
        assert Journal.load(j.id).acquire() is False
        j.release()
        assert Journal.load(j.id).acquire() is True

    def test_stale_lock_reclaimed(self, sessions_dir):
        j = Journal.create(name="a")
        lock = j.dir / "session.lock"
        lock.write_text("999999")   # a pid that cannot be alive
        assert j.acquire() is True   # stale lock reclaimed
        j.release()

    def test_resume_point(self, sessions_dir):
        j = Journal.create(name="a")
        j.log.append(Event(type="user_message", data={"text": "hi"}, source="user", turn=1))
        rp = j.resume_point()
        assert rp["journal_id"] == j.id
        assert rp["events"] == 2   # session_start + user_message
        assert rp["turn"] == 1
        assert rp["finished"] is False
        j.log.append(Event(type="finish", data={"message": "done"}, source="agent", turn=1))
        assert j.resume_point()["finished"] is True


class TestEncryption:
    def test_ephemeral_key(self, monkeypatch, sessions_dir):
        monkeypatch.setenv("LANGBOT_SESSION_ENCRYPT", "1")
        key, extra = resolve_session_key()
        assert key is not None
        assert extra == {"ephemeral_key": True}

        j = Journal.create(name="e", encrypt_key=key, meta_extra=extra)
        j.log.append(Event(type="user_message", data={"text": "secret"}, source="user"))
        j.release()

        # Without the key, the log is unreadable (undecryptable lines skipped)
        j2 = Journal.load(j.id)
        assert j2.events() == []

        # With the key, it round-trips
        j3 = Journal.load(j.id, encrypt_key=key)
        assert j3.events()[1].data["text"] == "secret"

    def test_passphrase_key_roundtrip(self, monkeypatch, sessions_dir):
        monkeypatch.setenv("LANGBOT_SESSION_ENCRYPT", "hunter2")
        key, extra = resolve_session_key()
        assert "kdf_salt" in extra

        j = Journal.create(name="p", encrypt_key=key, meta_extra=extra)
        j.log.append(Event(type="user_message", data={"text": "secret"}, source="user"))
        j.release()

        # Reopen with the same passphrase: salt read from meta.json → same key
        key2, extra2 = resolve_session_key(journal_id=j.id)
        assert key2 == key
        j2 = Journal.load(j.id, encrypt_key=key2)
        assert j2.events()[1].data["text"] == "secret"

    def test_encryption_off(self, monkeypatch, sessions_dir):
        monkeypatch.delenv("LANGBOT_SESSION_ENCRYPT", raising=False)
        key, extra = resolve_session_key()
        assert key is None
        assert extra == {}


class TestHeuristicTitle:
    def test_strips_common_prefixes(self):
        assert heuristic_title("please help me fix the build") == "fix the build"

    def test_takes_first_clause(self):
        assert heuristic_title("How do I deploy the server? Then what?") == "deploy the server"

    def test_truncates_long_titles(self):
        t = heuristic_title("can you " + "x" * 60)
        assert len(t) <= 50
        assert t.endswith("...")


class TestStreamTurnJournaling:
    """Integration: a synthetic turn through langbot._stream_turn journals
    user/ai/tool/finish events in order (Track B plan's integration test)."""

    def test_turn_journals_events_in_order(self, sessions_dir, monkeypatch):
        import os
        os.environ.setdefault("LANGBOT_VAULT_PASSWORD", "test-only-password")
        langbot = pytest.importorskip(
            "langbot", reason="requires the full runtime dependency set (langchain/langgraph/chromadb)"
        )
        from langchain_core.messages import AIMessage, ToolMessage

        j = Journal.create(name="session:t")
        monkeypatch.setattr(langbot, "_current_journal", j)
        monkeypatch.setattr(langbot, "_turn", 0)

        class FakeApp:
            def __init__(self, chunks):
                self._chunks = chunks

            def stream(self, *args, **kwargs):
                for c in self._chunks:
                    yield c

        chunks = [
            {"agent": {"messages": [AIMessage(
                content="",
                tool_calls=[{"name": "execute_shell_command",
                             "args": {"command": "echo hi"}, "id": "1"}],
            )]}},
            {"tools": {"messages": [ToolMessage(
                content="hi", tool_call_id="1", name="execute_shell_command",
            )]}},
            {"agent": {"messages": [AIMessage(content="Done.")]}},
        ]
        langbot._stream_turn(FakeApp(chunks), {}, "hello")

        evs = j.events()
        assert [e.type for e in evs] == [
            "session_start",
            "llm_response",
            "tool_call",
            "tool_result",
            "llm_response",
            "finish",
        ]
        assert evs[1].data["tool_calls"][0]["name"] == "execute_shell_command"
        assert evs[3].data["tool"] == "execute_shell_command"
        assert evs[3].data["preview"] == "hi"
        assert evs[5].data["message"] == "Done."
        assert all(e.turn == 0 for e in evs[1:])