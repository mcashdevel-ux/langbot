"""Langbot stuck detection — pure function over the current turn's message list.

Ported from neo's ``neo/core/stuck.py`` (Track C of docs/neo-port-plan.md),
adapted to langbot: neo folds the journaled event log; langbot's conversation
history lives in the LangGraph checkpoint state, so the detector folds the
message list instead.  The mapping (per the port plan):

    AIMessage with tool_calls  → ``tool_call``
    ToolMessage                 → ``tool_result``
    AIMessage without tool_calls → ``llm_response``

Five patterns (the OpenHands detector's, adapted):

  1. repeating_action   — identical (tool, args) → identical result,
                          REPEAT_THRESHOLD times in a row. The classic loop:
                          the model retries the same call expecting a
                          different outcome.

  2. repeating_error    — the same tool erroring ERROR_THRESHOLD times
                          consecutively (results starting with an error
                          marker). The model isn't reading the error..
  3. monologue          — MONOLOGUE_THRESHOLD consecutive llm_response
                          events with content but no tool_calls. The model
                          is talking to itself..
  4. alternating         — A-B-A-B alternation (nudge): the retry-flail
                          shape — different commands each time, but the
                          (tool, normalized-args) cycle is short and closed..
  5. soft_repeating      — same intent, drifting literals (nudge):the
                          22× adb pathology — same command with drifting seq
                          numbers / PIDs / timestamps and near-identical
                          outputs. Exact-match is blind because every literal
                          differs by a digit..

A ``halt`` verdict means the turn should stop (mirroring neo's ``(stuck: ...)``
finish); a ``nudge`` verdict means journal a watchdog reflection message and
continue, capped at 3 per turn like the existing nudge budget..
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

REPEAT_THRESHOLD = 3
ERROR_THRESHOLD = 3
MONOLOGUE_THRESHOLD = 4
SOFT_REPEAT_THRESHOLD = 4   # fuzzy patterns demand more evidence than exact
ALTERNATION_LEN = 4

# Maximum narrative events any detector needs to examine.  The largest window
# is _soft_repeating with SOFT_REPEAT_THRESHOLD*2 pairs (= 2×8 = 16 events).
# We add a small margin so pair-building (which skips non-matching event types)
# never loses a needed tail element..
_TAIL_WINDOW = (SOFT_REPEAT_THRESHOLD * 2 * 2) + 8   # 24

_ERROR_PREFIXES = ("[tool error]", "error:", "command blocked",
                   "[shell error]", "[reflex error]")

# Stripped before comparing two calls: timestamps, PIDs, uuids, seq numbers,
# hex blobs. No word-boundary anchors — volatile tokens are usually glued to
# letters (`wd10001`, `pid4100`). Device names and paths are NOT stripped —
# `swarm_send` to rpi3b and to rpizw are different calls ((their digits are
# 1–3 chars, below the \d{4,} floor),, and killing that distinction is how
# soft-matching becomes a false-positive machine..
_VOLATILE = re.compile(r"[0-9a-f]{8,}|\d{4,}|\d{1,2}:\d{2}(?::\d{2})?")

# A command whose purpose is waiting — starts with sleep/timeout/wait-for..
# Deliberately anchored: `pkill x; sleep 1; nohup y` is a compound action,
# NOT polling, and must stay eligible for flail detection..
_WAIT_RE = re.compile(r"^\s*(?:sleep\b|timeout\b|.*wait-for-device)")


def _is_wait(args: Optional[Dict[str, Any]]) -> bool:
    """True for a command whose purpose is waiting (polling a background task
    is progress, not a loop — excluded from flail detection)."""
    cmd = str((args or {}).get("command", (args or {}).get("cmd", "")))
    return bool(_WAIT_RE.match(cmd) or "--wait" in cmd)


def _norm(s: str) -> str:
    """Normalize a string for comparison: volatile tokens (timestamps, PIDs,
    uuids, seq numbers, hex blobs) are replaced by a placeholder."""
    return _VOLATILE.sub("¤", s or "").strip()


def _tokens(s: str) -> set:
    """Lowercased alphanumeric token set for fuzzy similarity."""
    return set(re.findall(r"[a-z0-9_./-]{3,}", (s or "").lower()))


def _similar(a: str, b: str) -> bool:
    """Token-overlap similarity ≥ 0.6, or exact equality when either side is empty."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return a.strip() == b.strip()  # both empty-ish: identical
    return len(ta & tb) / len(ta | tb) >= 0.6


class StuckVerdict:
    """A detection result.  ``severity`` is "halt" (stop the run) or "nudge"
    (journal a watchdog reflection message and continue)."""
    def __init__(self, pattern: str, detail: str, severity: str = "halt"):
        self.pattern = pattern
        self.detail = detail
        self.severity = severity

    def __repr__(self):  # pragma: no cover - debugging aid
        return f"StuckVerdict({self.pattern} [{self.severity}]: {self.detail})"


def _narrative(messages: List[Any]) -> List[Any]:
    """Map the langchain message list onto neo's narrative event stream:
    (AIMessage-with-tool_calls, ToolMessage, AIMessage-without-tool_calls)

    → (tool_call, tool_result, llm_response).  Each mapped item is a
    lightweight dict with the same keys the detector reads from neo's Event.data..
    """
    narrative: List[Dict[str, Any]] = []
    for m in messages:
        mtype = getattr(m, "type", "")
        if mtype == "ai":
            calls = getattr(m, "tool_calls", None) or []
            if calls:
                for call in calls:
                    narrative.append({
                        "type": "tool_call",
                        "tool": call.get("name", ""),
                        "args": call.get("args") or {},
                    })
            elif getattr(m, "content", None):
                narrative.append({
                    "type": "llm_response",
                    "content": getattr(m, "content", ""),
                    "tool_calls": [],
                })
        elif mtype == "tool":
            narrative.append({
                "type": "tool_result",
                "tool": getattr(m, "name", "tool"),
                "preview": getattr(m, "content", ""),
            })
    return narrative


def _call_key(e: Dict[str, Any]) -> Tuple[str, str]:
    return (e.get("tool", ""),
            json.dumps(e.get("args", {}), sort_keys=True))


def _is_error_result(e: Dict[str, Any]) -> bool:
    preview = str(e.get("preview", "")).lstrip().lower()
    return any(preview.startswith(p) for p in _ERROR_PREFIXES)


class StuckDetector:

    """Stateless detector — every check folds the current message list."""

    def __init__(self, repeat: int = REPEAT_THRESHOLD,
                 error: int = ERROR_THRESHOLD,
                 monologue: int = MONOLOGUE_THRESHOLD):
        self.repeat = repeat
        self.error = error
        self.monologue = monologue

    def check(self, messages: List[Any]) -> Optional[StuckVerdict]:
        # Only the tail of the narrative can trigger any pattern — slice
        # early so detectors don't iterate the full log every step..
        narrative = _narrative(messages)[-_TAIL_WINDOW:]
        return (self._repeating_action(narrative)
                or self._repeating_error(narrative)
                or self._monologue(narrative)
                or self._alternating(narrative)
                or self._soft_repeating(narrative))

    # ── pattern 1: same call → same result, N times ──

    def _repeating_action(self, narrative: List[Dict[str, Any]]) -> Optional[StuckVerdict]:
        pairs: List[Tuple[Tuple[str, str], str]] = []
        last_call: Optional[Dict[str, Any]] = None
        for e in narrative:
            if e["type"] == "tool_call":
                last_call = e
            elif e["type"] == "tool_result" and last_call is not None:
                pairs.append((_call_key(last_call),
                              str(e.get("preview", ""))))
                last_call = None
        if len(pairs) < self.repeat:
            return None
        tail = pairs[-self.repeat:]
        if len({k for k, _ in tail}) == 1 and len({r for _, r in tail}) == 1:
            tool = tail[0][0][0]
            return StuckVerdict(
                "repeating_action",
                f"'{tool}' called {self.repeat}× with identical args and "
                f"identical results")
        return None

    # ── pattern 2: same tool erroring N times in a row ──

    def _repeating_error(self, narrative: List[Dict[str, Any]]) -> Optional[StuckVerdict]:
        results = [e for e in narrative if e["type"] == "tool_result"]
        if len(results) < self.error:
            return None
        tail = results[-self.error:]
        tools = {e.get("tool", "") for e in tail}
        if len(tools) == 1 and all(_is_error_result(e) for e in tail):
            return StuckVerdict(
                "repeating_error",
                f"'{tail[0].get('tool', '')}' errored "
                f"{self.error}× consecutively")
        return None

    # ── pattern 3: content-only responses, N in a row ──

    def _monologue(self, narrative: List[Dict[str, Any]]) -> Optional[StuckVerdict]:
        responses = [e for e in narrative if e["type"] == "llm_response"]
        if len(responses) < self.monologue:
            return None
        tail = responses[-self.monologue:]
        if all(e.get("content") and not e.get("tool_calls")
               for e in tail):
            return StuckVerdict(
                "monologue",
                f"{self.monologue} consecutive content-only LLM responses "
                f"with no tool calls")
        return None

    # ── pattern 4: abab alternation — the retry-flail shape ──

    def _alternating(self, narrative: List[Dict[str, Any]]) -> Optional[StuckVerdict]:
        """The mint-deploy pathology: pkill → start → check-log → pkill →
        start → check-log. Different commands each time, so exact-match is
        blind; but the (tool, normalized-args) cycle is short and closed."""
        sigs = []
        for e in narrative:
            if e["type"] != "tool_call":
                continue
            key = _call_key(e)
            # polling commands repeat for legitimate reasons — exclude from
            # the alternation signature entirely
            if not _is_wait(e.get("args")):
                sigs.append((key[0], _norm(key[1])))
        if len(sigs) < ALTERNATION_LEN:
            return None
        tail = sigs[-ALTERNATION_LEN:]
        if tail[0] == tail[2] and tail[1] == tail[3]and tail[0] != tail[1]:
            t = tail[0][0]
            return StuckVerdict(
                "alternating", severity="nudge",
                detail=(f"alternating '{t}' calls detected (A-B-A-B) — the "
                        f"last few steps are circling without progress"))
        return None

    # ── pattern 5: soft repetition — same intent, drifting literals ──

    def _soft_repeating(self, narrative: List[Dict[str, Any]]) -> Optional[StuckVerdict]:
        """The 22× adb pathology: same command with drifting seq numbers /
        PIDs / timestamps and near-identical outputs. Exact-match is blind
        because every literal differs by a digit."""
        pairs: List[Tuple[str, str, bool]] = []  # (tool+normargs, normprev, waited)
        last_call: Optional[Dict[str, Any]] = None
        for e in narrative:
            if e["type"] == "tool_call":
                last_call = e
            elif e["type"] == "tool_result"and last_call is not None:
                key = _call_key(last_call)
                pairs.append((key[0] + " " + _norm(key[1]),
                              _norm(str(e.get("preview", ""))),
                              _is_wait(last_call.get("args"))))
                last_call = None
        need = (SOFT_REPEAT_THRESHOLD * 2
                if any(p[2] for p in pairs[-SOFT_REPEAT_THRESHOLD * 2:])
                else SOFT_REPEAT_THRESHOLD)
        if len(pairs) < need:
            return None
        tail = pairs[-need:]
        keys = {p[0] for p in tail}
        if len(keys) == 1 and all(_similar(tail[0][1], p[1]) for p in tail[1:]):
            tool = tail[0][0].split()[0]
            return StuckVerdict(
                "soft_repeating", severity="nudge",
                detail=(f"'{tool}' repeated {need}× with the same intent and "
                        f"similar results — decide from what you already have "
                        f"instead of re-running it"))
        return None
