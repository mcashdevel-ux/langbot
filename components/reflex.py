"""Procedural reflex — distilled 'when X → do Y' rules (manus-style).

Ported from neo's ``neo/core/reflex.py`` (Track D of docs/neo-port-plan.md),
adapted to langbot's config-driven paths:

- rules persist as JSON under ``./memory/reflexes.json`` (configurable via
  ``paths.reflexes_file`` or ``AGENT_REFLEXES_FILE``), per MEMORY_POLICY.md.
- ``match()`` fires on keyword overlap (≥ ``min_overlap`` trigger words appear
  in the query), best-first, before the LLM is ever called — deterministic,
  free, and fast.
- Safety still applies to the command a reflex runs: the ``reflex`` node
  dispatches through ``execute_shell_command``, which applies the catastrophic
  denylist and blast-radius warning logging exactly as it does for any other
  shell call.

A reflex entry::

    {"id", "trigger", "command", "uses", "disabled", "created_at"}

Near-leaf module by design — it depends only on ``config``, so any tool module
can import it without risking an import cycle..
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config

# Where rules persist.  Per MEMORY_POLICY.md: under ./memory/, configurable
# via ``paths.reflexes_file`` (or ``AGENT_REFLEXES_FILE``).
REFLEXES_FILE = config.get("paths.reflexes_file", "./memory/reflexes.json",
                             env="AGENT_REFLEXES_FILE")

# A rule fires when at least this many trigger words appear in the query.
MIN_OVERLAP = config.get("reflex.min_overlap", 2)


def _words(text: str) -> List[str]:
    """Lowercased alphanumeric words longer than 2 chars — the trigger
    vocabulary.  Short words (``df``, ``ls``) are deliberately excluded so
    a one-word overlap can't fire a rule by accident; ``min_overlap`` is
    still the gate on top of this."""
    return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) > 2]


class ReflexStore:
    """JSON-persisted procedural rules with atomic tmp+rename saves.

    Thread-safe via a lock (the ``reflex`` node and the ``reflex_distill`` tool
    can be called from different threads in principle)."""

    def __init__(self, path: Optional[Path] = None, min_overlap: int = MIN_OVERLAP):
        self.path = Path(path or REFLEXES_FILE)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.min_overlap = max(1, min_overlap)
        self._lock = threading.Lock()
        self._rules: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except (json.JSONDecodeError, UnicodeDecodeError):
                return []
        return []

    def _save(self) -> None:
        """Atomic write: write to a tmp file, then rename over the real one,
        so a crash mid-save never leaves a truncated rules file."""
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._rules, indent=2))
        os.replace(tmp, self.path)

    # ── write ──
    def distill(self, trigger: str, command: str) -> str:
        """Store (or update in place) a rule. Returns its id, or "" if
        either field is empty.  Updating an existing rule keeps its id and
        use count — the procedure improved, not replaced."""
        trigger, command = (trigger or "").strip(), (command or "").strip()
        if not trigger or not command:
            return ""
        norm = re.sub(r"\s+", " ", trigger.lower())
        with self._lock:
            for r in self._rules:
                if re.sub(r"\s+", " ", r["trigger"].lower()) == norm:
                    r["command"] = command  # update the procedure in place
                    self._save()
                    return r["id"]
            rid = uuid.uuid4().hex[:12]
            self._rules.append({
                "id": rid, "trigger": trigger, "command": command,
                "uses": 0, "disabled": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            self._save()
            return rid

    # ── read ──
    def match(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """Keyword-overlap match: a rule fires if enough trigger words appear
        in the query. Returns best-first rules with their commands."""
        qwords = set(_words(query))
        if not qwords:
            return []
        hits = []
        for r in self._rules:
            if r.get("disabled"):
                continue
            overlap = len(qwords & set(_words(r["trigger"])))
            if overlap >= self.min_overlap:
                hits.append((overlap, r))
        hits.sort(key=lambda h: -h[0])
        return [r for _, r in hits[:n]]

    def use(self, rule_id: str) -> None:
        """Increment a rule's use counter and persist."""
        with self._lock:
            for r in self._rules:
                if r["id"] == rule_id:
                    r["uses"] = r.get("uses", 0) + 1
                    self._save()
                    return

    def disable(self, rule_id: str) -> bool:
        """Disable a rule so it no longer matches. Returns True if found."""
        with self._lock:
            for r in self._rules:
                if r["id"] == rule_id:
                    r["disabled"] = True
                    self._save()
                    return True
        return False

    def enable(self, rule_id: str) -> bool:
        """Re-enable a disabled rule. Returns True if found."""
        with self._lock:
            for r in self._rules:
                if r["id"] == rule_id:
                    r["disabled"] = False
                    self._save()
                    return True
        return False

    def all(self) -> List[Dict[str, Any]]:
        """All rules, in insertion order."""
        with self._lock:
            return list(self._rules)

    def count(self) -> int:
        return len(self._rules)