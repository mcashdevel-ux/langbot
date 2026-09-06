"""Durable journaled sessions — append-only JSONL event logs per session.

Ported from neo's ``neo/core/journal.py`` (Track B of docs/neo-port-plan.md),
adapted to langbot:

- one directory per session under ``./memory/sessions/<id>/``:
    events.jsonl   the source of truth — one Event per line, monotonic ``seq``
    meta.json      id, name, created_at, log_version, title, encrypted
    session.lock   single-writer lock (O_EXCL creation; see Journal.acquire)

Design rules (mirroring neo):
  - append-only: events are never mutated in place; condensation is an event
  - torn-write recovery: a trailing non-JSON line (crash mid-write) is
    truncated on load
  - forward-tolerant readers: unknown event types and fields are ignored
  - previews, not payloads: large tool results are truncated (head+tail)

The journal is an *audit/observability* layer, not the source of truth — langbot's
conversation history still lives in the LangGraph checkpoint DB.  The ``View``
class from neo is deliberately not ported (the message list comes from the graph
state, so folding the log back into messages would duplicate it).

Tier 0: stdlib only (optional ``cryptography`` for Fernet encryption, exactly
like neo).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import config

LOG_VERSION = 1

# Pre-salt logs were derived with this fixed salt. New sessions get a random
# per-session salt stored in meta.json; the constant remains only so those
# legacy logs stay readable.
LEGACY_SALT = b"langbot-session-v1"


def journals_dir() -> Path:
    """Return the root directory for all journal sessions.

    Overridable via ``AGENT_SESSIONS_DIR`` (or ``paths.sessions_dir`` in the
    config file); defaults to ``./memory/sessions`` per MEMORY_POLICY.md.
    """
    return Path(config.get("paths.sessions_dir", "./memory/sessions",
                           env="AGENT_SESSIONS_DIR"))


def new_salt() -> bytes:
    """Random 16-byte KDF salt for a new encrypted session."""
    return os.urandom(16)


def derive_key(passphrase: str, salt: Optional[bytes] = None) -> bytes:
    """Derive a 32-byte Fernet key from a passphrase using PBKDF2.

    ``salt=None`` uses the legacy fixed salt — kept only to decrypt logs
    written before per-session salts. New sessions must pass ``new_salt()``
    and persist it in journal meta (see ``resolve_session_key``).
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt or LEGACY_SALT, iterations=100_000)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode()))


def generate_key() -> bytes:
    """Generate a random 32-byte Fernet key (for per-session encryption)."""
    return base64.urlsafe_b64encode(os.urandom(32))


def resolve_session_key(journal_id: Optional[str] = None
                        ) -> Tuple[Optional[bytes], Dict[str, Any]]:
    """Resolve LANGBOT_SESSION_ENCRYPT to (fernet_key, meta_extra).

    Returns (None, {}) when encryption is off. ``meta_extra`` carries fields
    the caller must persist into journal meta on create (the KDF salt, so the
    same passphrase can reopen the session later).

    Values:
      "1"         — ephemeral random key, held in memory only. The log is
                    permanently unreadable once the process exits..
      passphrase  — PBKDF2 key. For an existing ``journal_id`` the salt is
                    read from its meta.json (logs without one fall back to
                    the legacy fixed salt); for a new session a random salt
                    is generated and returned in ``meta_extra``..
    """
    env = os.environ.get("LANGBOT_SESSION_ENCRYPT")
    if not env:
        return None, {}
    if env == "1":
        warnings.warn(
            "LANGBOT_SESSION_ENCRYPT=1 uses an ephemeral in-memory key: the "
            "session log will be unreadable after this process exits and "
            "cannot be resumed. Set LANGBOT_SESSION_ENCRYPT to a passphrase to "
            "keep encrypted sessions resumable.", stacklevel=2)
        return generate_key(), {"ephemeral_key": True}
    if journal_id is not None:
        salt: Optional[bytes] = None
        try:
            meta = json.loads(
                (journals_dir() / journal_id / "meta.json").read_text())
            b64 = meta.get("kdf_salt")
            if b64:
                salt = base64.b64decode(b64)
        except (OSError, ValueError, KeyError):
            pass  # missing/corrupt meta → legacy salt
        return derive_key(env, salt), {}
    salt = new_salt()
    return derive_key(env, salt), {"kdf_salt": base64.b64encode(salt).decode()}


# ── Event ───────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """One journaled event. JSON-serializable wire schema (mirrors neo's)."""
    type: str
    data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    source: str = "agent"                # agent | user | tool | system
    turn: int = 0                         # which human turn this belongs to
    seq: int = 0                         # log-assigned monotonic id (0 = pre-append)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Event":
        e = cls(**{k: d.get(k) for k in
                   ("type", "data", "id", "ts", "source", "turn", "seq")})
        # forward tolerance: malformed/partial events must never crash
        # readers — normalize the fields readers depend on
        if not isinstance(e.data, dict):
            e.data = {}
        if not isinstance(e.seq, int):
            try:
                e.seq = int(e.seq)
            except (TypeError, ValueError):
                e.seq = 0
        return e


# ── EventLog ────────────────────────────────────────────────────────────────

class EventLog:
    """Append-only JSONL log with torn-tail recovery.

    Writes are single ``write()`` calls with a flush; readers tolerate and
    truncate a trailing partial line from a crash mid-write..

    When ``encrypt_key`` is set, each line is a Fernet-encrypted JSON event
    (base64.). The key is derived from a passphrase or generated per-session
    and held in memory only..
    """

    def __init__(self, path: Path, encrypt_key: Optional[bytes] = None):
        """Open (or create) an append-only JSONL event log at *path*..

        If *encrypt_key* is provided, each line is Fernet-encrypted..
        Torn-tail recovery runs on construction..
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._fh = None
        self._fernet = None
        if encrypt_key:
            from cryptography.fernet import Fernet
            self._fernet = Fernet(encrypt_key)
        self._check_tail()

    def _encrypt(self, data: str) -> str:
        """Encrypt *data* with Fernet if a key is set; otherwise return as-is."""
        if self._fernet is None:
            return data
        return self._fernet.encrypt(data.encode()).decode()

    def _decrypt(self, data: str) -> str:
        """Decrypt *data* with Fernet if a key is set; otherwise return as-is."""
        if self._fernet is None:
            return data
        return self._fernet.decrypt(data.encode()).decode()

    def _check_tail(self) -> None:
        """Truncate a torn final line; recover the seq counter."""
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        if not raw:
            return
        if not raw.endswith(b"\n"):
            # crash mid-write — drop the partial line
            good = raw.rfind(b"\n")
            with self.path.open("r+b") as f:
                f.truncate(good + 1 if good >= 0 else 0)
            raw = raw[: good + 1] if good >= 0 else b""
        for line in raw.splitlines():
            try:
                plain = self._decrypt(line.decode("utf-8", errors="replace"))
                self._seq = max(self._seq, int(json.loads(plain).get("seq", 0)))
            except Exception:
                # unparseable or undecryptable line — seq recovery is
                # best-effort; read_all applies the same tolerance
                continue

    def append(self, event: Event, fsync: bool = False) -> Event:
        """Append *event* to the log, assigning the next monotonic ``seq``..

        The write is a single ``write()`` + ``flush()``; pass ``fsync=True``
        for durability (e.g. session-start). Returns the event with ``seq``
        set..
        """
        self._seq += 1
        event.seq = self._seq
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
            try:
                os.chmod(self.path, 0o600)  # owner-only
            except OSError:
                pass
        line = json.dumps(event.to_json(), default=str)
        self._fh.write(self._encrypt(line) + "\n")
        self._fh.flush()
        if fsync:
            os.fsync(self._fh.fileno())
        return event

    def read_all(self) -> List[Event]:
        """Read and parse all events from the log. Unparseable or
        undecryptable lines are skipped (forward-tolerant.."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8",
                                        errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                line = self._decrypt(line)
                out.append(Event.from_json(json.loads(line)))
            except Exception:
                continue  # forward-tolerant: skip anything unparseable
        return out

    def tail(self, n: int = 50) -> List[Event]:
        """Return the last *n* events from the log."""
        return self.read_all()[-n:]

    def close(self) -> None:
        """Close the underlying file handle if open."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# ── Journal ─────────────────────────────────────────────────────────────────

class Journal:
    """A durable agent session: event log + metadata + single-writer lock.

    Usage:
        j = Journal.create(name="session:t1")
        j.log.append(Event(type="user_message", data={"text": "hi"}, source="user"))
        ...
        j2 = Journal.load(j.id)      # after restart — read the log
    """

    def __init__(self, journal_id: str, directory: Path,
                 encrypt_key: Optional[bytes] = None):
        """Open an existing journal at *directory* with the given *journal_id*..

        Use ``Journal.create()`` for new sessions and ``Journal.load()`` for
        resume. *encrypt_key* is the Fernet key for encrypted logs..
        """
        self.id = journal_id
        self.dir = directory
        self.log = EventLog(directory / "events.jsonl",
                            encrypt_key=encrypt_key)
        self._meta_path = directory / "meta.json"
        self._lock_fd: Optional[int] = None
        self._encrypt_key = encrypt_key

    # ── lifecycle ──

    @classmethod
    def create(cls, name: str = "agent",
               journal_id: Optional[str] = None,
               encrypt_key: Optional[bytes] = None,
               meta_extra: Optional[Dict[str, Any]] = None) -> "Journal":
        """Create a new journal session: allocate a directory, write
        ``meta.json``, emit a ``session_start`` event, and return the Journal..

        *meta_extra* fields (e.g. ``kdf_salt``) are merged into meta.json..
        """
        sid = journal_id or uuid.uuid4().hex[:12]
        root = journals_dir()
        root.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(root, 0o700)
        except OSError:
            pass
        d = root / sid
        d.mkdir(parents=True, exist_ok=True)
        # owner-only permissions (local multi-user isolation)
        d.chmod(0o700)
        meta = {
            "id": sid,
            "name": name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "log_version": LOG_VERSION,
            "title": None,
            "encrypted": encrypt_key is not None,
        }
        if meta_extra:
            meta.update(meta_extra)
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        (d / "meta.json").chmod(0o600)
        j = cls(sid, d, encrypt_key=encrypt_key)
        j.log.append(Event(type="session_start",
                           data={"name": name},
                           source="system"), fsync=True)
        return j

    @classmethod
    def load(cls, journal_id: str,
             encrypt_key: Optional[bytes] = None) -> "Journal":
        """Open an existing journal by id for resume. Raises FileNotFoundError
        if the session directory or events.jsonl does not exist.."""
        d = journals_dir() / journal_id
        if not (d / "events.jsonl").exists():
            raise FileNotFoundError(
                f"no journal {journal_id!r} in {journals_dir()}")
        return cls(journal_id, d, encrypt_key=encrypt_key)

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        """All journals, newest first, from their meta.json."""
        root = journals_dir()
        if not root.exists():
            return []
        metas = []
        for d in root.iterdir():
            mp = d / "meta.json"
            if mp.exists():
                try:
                    m = json.loads(mp.read_text())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                # Track last activity via events.jsonl mtime
                ep = d / "events.jsonl"
                if ep.exists():
                    m["updated_at"] = ep.stat().st_mtime
                metas.append(m)
        # Sort by last activity (updated_at) descending; fall back to created_at
        return sorted(metas,
                      key=lambda m: m.get("updated_at", 0) or m.get("created_at", ""),
                      reverse=True)



    def meta(self) -> Dict[str, Any]:
        """Read and return this journal's metadata dict. Returns ``{'id': …}``
        if meta.json is missing or corrupt.."""
        try:
            return json.loads(self._meta_path.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
            return {"id": self.id}

    def set_title(self, title: str) -> None:
        """Set the session title in meta.json (overwrites any existing title.."""
        m = self.meta()
        m["title"] = title
        self._meta_path.write_text(json.dumps(m, indent=2))

    # ── single-writer lock ──

    def acquire(self) -> bool:
        """Take the single-writer lock. Returns False if another live process
        holds it (that process is the writer; everyone else is a read-only
        tail.. Stale locks from dead processes are reclaimed.."""
        lock_path = self.dir / "session.lock"
        try:
            self._lock_fd = os.open(str(lock_path),
                                    os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.fchmod(self._lock_fd, 0o600)
            except OSError:
                pass
            os.write(self._lock_fd, str(os.getpid()).encode())
            return True
        except FileExistsError:
            try:
                pid = int(lock_path.read_text().strip())
                os.kill(pid, 0)          # alive?
                return False             # legitimately held
            except (ValueError, ProcessLookupError, PermissionError):
                lock_path.unlink(missing_ok=True)   # stale — reclaim
                return self.acquire()

    def release(self) -> None:
        """Release the single-writer lock and close the event log file handle."""
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
        (self.dir / "session.lock").unlink(missing_ok=True)
        self.log.close()

    # ── state folding ──

    def events(self) -> List[Event]:
        """Return all events in the journal (folds the entire log.."""
        return self.log.read_all()

    def turn(self) -> int:
        """Return the highest turn number across all events (0 if empty.."""
        return max((e.turn for e in self.events()), default=0)

    def finished(self) -> bool:
        """True if a ``finish`` event appears in the last 5 log entries."""
        return any(e.type == "finish" for e in self.log.tail(5))

    def resume_point(self) -> Dict[str, Any]:
        """Return a summary dict for resume: journal id,, event count,, current
        turn,, and whether the run finished.."""
        evs = self.events()
        return {
            "journal_id": self.id,
            "events": len(evs),
            "turn": max((e.turn for e in evs), default=0),
            "finished": self.finished(),
        }


# ── title heuristic ──────────────────────────────────────────────────────────

def heuristic_title(text: str) -> str:
    """Short title from the first user message — no LLM call needed..

    Takes the first meaningful phrase, truncates to ~50 chars.  Ported from
    neo's ``_heuristic_title`` (neo/core/kernel.py)..
    """
    # skip common prefixes (loop to catch nested ones like "please help me")
    for _ in range(3):
        for prefix in ("please ", "can you ", "could you ", "help me ",
                       "i need ", "i want ", "how do i ", "how to "):
            if text.lower().startswith(prefix):
                text = text[len(prefix):]
                break
        else:
            break
    # take first sentence or clause
    for sep in (". ", "? ", "! ", "; ", "\n"):
        if sep in text:
            text = text.split(sep)[0]
            break
    title = text.strip()
    if len(title) > 50:
        title = title[:47] + "..."
    return title