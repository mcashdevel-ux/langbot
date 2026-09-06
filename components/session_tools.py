"""Session history tools — ``session_list`` / ``session_search`` / ``journal_search``.

Ported from neo's ``neo/tools/essential.py`` (Track E of docs/neo-port-plan.md),
adapted to langbot's journal layout (``components/journal.py``, one directory
per session under ``./memory/sessions/<id>/``) and message format.

These tools cover *events* — what happened in past sessions — where
``remember``/``recall`` cover *facts*.  They let the agent answer "what did
we do last week?" without the human grepping ``./memory/sessions/``.

Design rules (mirroring neo):
  - ``session_list`` lists journals, newest by activity, with event counts and
    finished status (a ``finish`` event in the last 5 entries marks done).
  - ``session_search`` searches one past session's events by keyword, with
    ``n`` limiting how many matches are returned.  Unknown session id →
    clean error, no exception.

  - ``journal_search`` searches the *current* session's full journal (including
    events demoted/summarized away from the view) — the journal is threaded
    through a module-level ``_current_journal`` set by ``_stream_turn`` (or the
    graph config's thread_id, looked up by name).

All three are read-only and best-effort: a corrupt/missing journal yields a
clean message, never an exception.  Encrypted sessions are handled
transparently via ``resolve_session_key`` (exactly like the ``/session`` slash
command.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .journal import Journal, journals_dir, resolve_session_key

logger = logging.getLogger(__name__)

# The current turn's journal, set by langbot._stream_turn (or looked up by
# thread_id from the graph config).  Module-level so the tools can reach it
# without threading a ctx parameter through every tool wrapper (Track G is
# the structural fix for that; this is the pragmatic interim).
_current_journal: Optional[Journal] = None


def set_current_journal(journal: Optional[Journal]) -> None:
    """Point the session tools at the current turn's journal.



    Called by ``_stream_turn`` when a turn starts (and cleared when it ends..
    """
    global _current_journal
    _current_journal = journal


def _event_text(ev: Any) -> str:
    """Flatten an event's text-bearing fields into one lowercase-able string.



    Nested dicts (e.g. a tool_call's ``args``) are recursed so their values
    participate in keyword search; lists are joined element-wise..
    """
    d = ev.data if isinstance(ev.data, dict) else {}
    parts = []

    def _walk(v: Any) -> None:
        if isinstance(v, dict):
            for val in v.values():
                _walk(val)
        elif isinstance(v, (list, tuple)):
            for val in v:
                _walk(val)
        elif isinstance(v, (str, int, float)):
            parts.append(str(v))

    _walk(d)
    return " ".join(parts)


def _format_event(ev: Any, width: int) -> str:
    """Render one event as a single line for search output."""
    d = ev.data if isinstance(ev.data, dict) else {}
    seq = ev.seq
    if ev.type == "user_message":
        content = d.get("text", "")[:width]
    elif ev.type == "tool_result":
        content = f"[{d.get('tool', '?')}] {str(d.get('preview', ''))[:width]}"
    elif ev.type == "llm_response":
        content = d.get("content", "")[:width]
        calls = d.get("tool_calls", [])
        if calls:
            names = [c.get("function", {}).get("name", "?") for c in calls]
            content += f" calls={names}"
    elif ev.type == "agent_message":
        content = d.get("text", "")[:width]
    elif ev.type == "condensation":
        kind = d.get("kind", "?")
        summary = d.get("summary", "")
        content = f"[{kind}] {summary[:width]}" if summary else f"[{kind}]"
    else:
        content = str(d)[:width]
    return f"  [{seq}] {ev.type}: {content}"


def session_list(limit: int = 20) -> str:
    """List past sessions with metadata (id, title, created_at, event count,
    finished status).  Use to find past work and get a session ID for
    ``session_search``.  Pass ``limit`` to control how many sessions are
    returned (default 20, newest first)."""
    root = journals_dir()
    if not root.exists():
        return "(no sessions found)"
    metas = Journal.list()
    if not metas:
        return "(no sessions found)"
    lines = []
    for m in metas[:limit]:
        sid = m.get("id", "?")
        title = m.get("title") or "(untitled)"
        created = m.get("created_at", "?")
        name = m.get("name", "?")
        n_events = 0
        finished = False
        ev_path = root / sid / "events.jsonl"
        if ev_path.exists():
            try:
                with open(ev_path, "r") as f:
                    for line in f:
                        n_events += 1
                        if '"finish"' in line:
                            finished = True
            except Exception:  # noqa: BLE001 — a corrupt log must not break the list
                pass
        status = "done" if finished else "active/incomplete"
        lines.append(f"  {sid}  {title[:50]:50s}  {created}  "
                     f"{n_events:5d} events  [{status}]  ({name})")
    return (f"Sessions ({len(metas)} total, showing {min(limit, len(metas))}):\n"
            + "\n".join(lines))


def session_search(session_id: str, query: str, n: int = 10) -> str:
    """Search events in a past session by keyword. Returns matching events
    (user messages,, tool calls,, tool results,, agent messages) with context.

    Use ``session_list`` first to get the session ID.  Pass ``n`` to limit
    results (default 10).  Unknown session id → clean error, no
    exception."""
    try:
        key, _ = resolve_session_key(journal_id=session_id)
        j = Journal.load(session_id, encrypt_key=key)
        evs = j.events()
    except FileNotFoundError:
        return f"Session {session_id} not found"
    except Exception as e:  # noqa: BLE001 — search errors are user-facing
        return f"[error reading session] {e}"
    query_lower = query.lower()
    matches = [ev for ev in evs
                if query_lower in _event_text(ev).lower()
                or query_lower in ev.type.lower()]
    if not matches:
        return f"No events matching '{query}' in session {session_id}"
    lines = [_format_event(ev, 200) for ev in matches[:n]]
    header = f"Session {session_id}: {len(matches)} matches for '{query}'"
    if len(matches) > n:
        header += f" (showing first {n})"
    return header + "\n" + "\n".join(lines)


def journal_search(query: str, n: int = 10) -> str:
    """Search the CURRENT session's full journal (including events that were
    demoted or summarized away from the view).  Returns matching events with
    their content.  Use this to recall details that were lost during context
    condensation."""
    j = _current_journal
    if j is None:
        return "(no journal available)"
    query_lower = query.lower()
    try:
        evs = j.events()
    except Exception as e:  # noqa: BLE001 — search errors are user-facing
        return f"[error reading journal] {e}"
    matches = [ev for ev in evs
                if query_lower in _event_text(ev).lower()
                or query_lower in ev.type.lower()]
    if not matches:
        return f"No events matching '{query}' in current journal"
    lines = [_format_event(ev, 300) for ev in matches[:n]]
    header = f"Journal search: {len(matches)} matches for '{query}'"
    if len(matches) > n:
        header += f" (showing first {n})"
    return header + "\n" + "\n".join(lines)