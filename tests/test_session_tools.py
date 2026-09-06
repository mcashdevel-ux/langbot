"""Unit tests for components/session_tools.py — session history tools (Track E.

These cover the three tools ported from neo (``session_list``,
``session_search``, ``journal_search``), adapted to langbot's journal
layout.  They verify:

  - ``session_list`` lists journals newest-by-activity with event counts and
    finished status;
  - ``session_search`` finds keyword matches in a past session, respects
    ``n``, and returns a clean error for an unknown session id;
  - ``journal_search`` searches the current session's journal (via the module-
    level ``_current_journal`` set by ``set_current_journal``), including
    events that were demoted/summarized;
  - encrypted sessions are searched transparently (passphrase re-derived from
    the stored salt).

The journal root is redirected to a temp dir for every test so the real
``./memory/sessions/`` is never touched.
"""

import os

import pytest

import components.journal as journal_mod
from components.journal import Event, Journal
import components.session_tools as st


@pytest.fixture(autouse=True)
def sessions_dir(tmp_path, monkeypatch):
    """Redirect the journal root into a temp dir for every test."""
    d = tmp_path / "sessions"
    d.mkdir()
    # Patch both namespaces:the alias imported into session_tools,and the
    # canonical one in components.journal (Journal.list/load call it
    # internally).
    monkeypatch.setattr(st, "journals_dir", lambda: d)
    monkeypatch.setattr(journal_mod, "journals_dir", lambda: d)
    return d


def _seed_session(sid="session:a", title="disk space investigation", events=None):
    """Create a journal with a known id and a few known events; return it."""
    j = Journal.create(name=f"session:{sid}", journal_id=sid)
    if title:
        j.set_title(title)
    for ev in (events or [
        Event(type="user_message", data={"text": "check disk space"}, source="user", turn=1),
        Event(type="tool_call", data={"name": "execute_shell_command",
                                     "args": {"command": "df -h"}}, source="agent", turn=1),
        Event(type="tool_result", data={"tool": "execute_shell_command",
                                        "preview": "Filesystem Size Used Avail Use% Mounted on"}, source="tool", turn=1),
        Event(type="llm_response", data={"content": "Disk usage looks fine."}, source="agent", turn=1),
        Event(type="finish", data={"message": "Disk usage looks fine."}, source="agent", turn=1),
    ]):
        j.log.append(ev)
    return j


class TestSessionList:
    def test_lists_sessions_newest_first(self, sessions_dir):
        a = _seed_session(sid="session:a", title="older session")
        b = _seed_session(sid="session:b", title="newer session")
        out = st.session_list()
        assert "2 total" in out
        # b was created after a → newest first
        assert out.index("session:b") < out.index("session:a")
        assert "disk space investigation" in out or "older session" in out
        assert "6 events" in out
        assert "done" in out  # finish event present

    def test_limit(self, sessions_dir):
        _seed_session(sid="session:a")
        _seed_session(sid="session:b")
        out = st.session_list(limit=1)
        assert "showing 1" in out
        assert "session:b" in out
        assert "session:a" not in out

    def test_empty(self, sessions_dir):
        assert st.session_list() == "(no sessions found)"


class TestSessionSearch:
    def test_finds_matching_events(self, sessions_dir):
        _seed_session()
        out = st.session_search("session:a", "disk")
        assert "3 matches" in out
        assert "user_message" in out
        assert "check disk space" in out
        assert "llm_response" in out
        assert "Disk usage looks fine." in out
        # Nested args participate in search (tool_call's command "df -h")
        out2 = st.session_search("session:a", "df -h")
        assert "tool_call" in out2
        assert "execute_shell_command" in out2

    def test_respects_n(self, sessions_dir):
        _seed_session()
        out = st.session_search("session:a", "disk", n=1)
        assert "showing first 1" in out
        assert out.count("[") == 1  # one match line

    def test_unknown_session(self, sessions_dir):
        out = st.session_search("session:nope", "disk")
        assert out == "Session session:nope not found"

    def test_no_match(self, sessions_dir):
        _seed_session()
        out = st.session_search("session:a", "zzzznope")
        assert "No events matching" in out

    def test_encrypted_session_searchable(self, sessions_dir, monkeypatch):
        monkeypatch.setenv("LANGBOT_SESSION_ENCRYPT", "test-passphrase")
        key, meta_extra = journal_mod.resolve_session_key()
        j = Journal.create(name="session:session:enc", journal_id="session:enc",
                          encrypt_key=key, meta_extra=meta_extra)
        j.log.append(Event(type="user_message", data={"text": "secret plan"}, source="user"))
        j.log.close()
        out = st.session_search("session:enc", "secret")
        assert "secret plan" in out


class TestJournalSearch:
    def test_searches_current_journal(self, sessions_dir):
        j = _seed_session()
        st.set_current_journal(j)
        try:
            out = st.journal_search("disk")
            assert "Journal search: 3 matches" in out
            assert "check disk space" in out
        finally:
            st.set_current_journal(None)

    def test_includes_demoted_events(self, sessions_dir):
        j = _seed_session()
        j.log.append(Event(type="condensation", data={"kind": "demote",
                                                      "summary": "old tool result about nginx config"}, source="system"))
        st.set_current_journal(j)
        try:
            out = st.journal_search("nginx")
            assert "condensation" in out
            assert "nginx config" in out
        finally:
            st.set_current_journal(None)

    def test_no_journal(self):
        st.set_current_journal(None)
        assert st.journal_search("anything") == "(no journal available)"

    def test_no_match(self, sessions_dir):
        j = _seed_session()
        st.set_current_journal(j)
        try:
            assert st.journal_search("zzzznope") == "No events matching 'zzzznope' in current journal"
        finally:
            st.set_current_journal(None)
