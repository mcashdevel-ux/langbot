"""Hard, non-interactive refusal for a small set of catastrophic shell commands.

This is deliberately narrow. It exists to catch whole-system, irreversible
actions — wipe the filesystem root, wipe every user's home directory,
overwrite a raw block device, fork-bomb the process table — and refuse them
outright, synchronously, with no confirmation prompt and no pause for a
human. Everything else the agent does, including the softer "blast radius"
patterns already logged elsewhere (force-push, DROP TABLE, an ordinary
`rm -rf` on a project directory), still runs immediately and without
interruption.

This is a denylist, not a sandbox. It cannot see through arbitrary
obfuscation (base64-encoded payloads piped to `bash`, octal-escaped
strings, a command built up across several tool calls). Treat it as one
cheap layer under an agent that is otherwise trusted to act autonomously —
not as a security boundary against a genuinely adversarial actor. That
threat (e.g. an instruction smuggled in via `fetch_url` content) needs a
different mitigation: constraining what the model can do with vault
secrets and outbound network access, not a bigger denylist here.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass

# A dangerous sub-command chained after something innocuous
# (`echo hi; rm -rf /`) should still be caught, so the full string is split
# on common shell separators before each piece is checked individually.
# This is best-effort text splitting, not a real shell parser.
_SEPARATORS = re.compile(r"&&|\|\||;|\n|\|(?!\|)")

# Path tokens that mean "wipe everything" for an rm -rf-style call.
# Trailing slashes and a trailing /* are normalized away before comparison
# (see _normalize_rm_target), so this set only needs the canonical form.
_CATASTROPHIC_RM_TARGETS = {
    "/",         # filesystem root
    "/home",     # every user's home directory
    "/root",     # root's home directory
    "~",         # shell shorthand for the invoking user's home
    "$home",     # $HOME, compared case-insensitively
    "${home}",   # ${HOME}, compared case-insensitively
}

_RM_RECURSIVE_FLAGS = {"-r", "-R", "--recursive"}
_RM_FORCE_FLAGS = {"-f", "--force"}

# Classic bash fork bomb, e.g. `:(){ :|:& };:`, generalized to any function
# name (backreference \1 requires the same token to open, appear twice in
# the body, and be called at the end).
_FORK_BOMB_RE = re.compile(
    r"([^\s();{}|&]{1,32})\s*\(\)\s*\{\s*\1\s*\|\s*\1\s*&\s*\}\s*;\s*\1"
)

# Raw-disk destruction: `dd ... of=/dev/sdX`, `mkfs.* /dev/sdX`,
# `wipefs ... /dev/sdX`, or a bare redirect onto a device node. Deliberately
# does NOT match `dd if=/dev/sda of=backup.img` (reading a device is normal;
# writing to one is not).
_DISK_DESTROY_RE = re.compile(
    r"\bdd\b[^;&|\n]*\bof=/dev/(?:sd|hd|nvme|vd|xvd)\w*"
    r"|\bmkfs\b\S*[^;&|\n]*/dev/(?:sd|hd|nvme|vd|xvd)\w*"
    r"|\bwipefs\b[^;&|\n]*/dev/(?:sd|hd|nvme|vd|xvd)\w*"
    r"|>\s*/dev/(?:sd|hd|nvme|vd|xvd)\w*",
    re.IGNORECASE,
)


def _normalize_rm_target(token: str) -> str:
    t = token.strip().strip('"').strip("'")
    if len(t) > 1 and t.endswith("/*"):
        t = t[:-2] or "/"
    if len(t) > 1 and t.endswith("/"):
        t = t.rstrip("/") or "/"
    return t.lower()


def _is_catastrophic_rm(sub_command: str) -> "str | None":
    """Return a description if sub_command is `rm -rf` (in any flag order or
    combination) targeting a wipe-everything path, else None.
    """
    try:
        tokens = shlex.split(sub_command, posix=True)
    except ValueError:
        # Unbalanced quotes etc.: don't guess at a malformed command here,
        # the regex layer (_DISK_DESTROY_RE / _FORK_BOMB_RE) still applies.
        return None

    # Skip a leading `sudo`/`command`/env-var-prefix so `sudo rm -rf /` and
    # plain `rm -rf /` are treated the same.
    i = 0
    while i < len(tokens) and (
        tokens[i] in ("sudo", "command") or re.fullmatch(r"[A-Za-z_]\w*=.*", tokens[i])
    ):
        i += 1
    tokens = tokens[i:]

    if not tokens or tokens[0].rsplit("/", 1)[-1] != "rm":
        return None

    has_recursive = False
    has_force = False
    targets: list[str] = []
    for tok in tokens[1:]:
        if tok in _RM_RECURSIVE_FLAGS:
            has_recursive = True
            continue
        if tok in _RM_FORCE_FLAGS:
            has_force = True
            continue
        if tok.startswith("-") and not tok.startswith("--"):
            # combined short flags, e.g. -rf, -fr, -rfv
            if "r" in tok or "R" in tok:
                has_recursive = True
            if "f" in tok:
                has_force = True
            continue
        if tok.startswith("-"):
            continue
        targets.append(tok)

    if not (has_recursive and has_force):
        return None

    for tok in targets:
        if _normalize_rm_target(tok) in _CATASTROPHIC_RM_TARGETS:
            return f"rm -rf targeting {tok!r}"
    return None


def catastrophic_reason(command: str) -> "str | None":
    """Return a short reason if `command` should be refused outright, else None.

    Checked against the whole string (fork bombs, disk destruction) and
    against each `;` / `&&` / `||` / `|`-separated segment (rm -rf), so a
    dangerous call chained after something harmless is still caught.
    """
    if _FORK_BOMB_RE.search(command):
        return "fork bomb"

    disk_hit = _DISK_DESTROY_RE.search(command)
    if disk_hit:
        return f"raw-disk overwrite ({disk_hit.group(0).strip()})"

    for segment in _SEPARATORS.split(command):
        segment = segment.strip()
        if not segment:
            continue
        reason = _is_catastrophic_rm(segment)
        if reason:
            return reason
    return None


# ---------------------------------------------------------------------------
# Track F — token-aware confirmation gate (ported from neo/core/safety.py)
# ---------------------------------------------------------------------------
# langbot's ``catastrophic_reason`` above is the hard-block layer (whole-system,
# irreversible actions — refused outright, no confirmation, ever).  Neo's
# ``SafetyGate`` adds the *gray zone* on top: a trusted-readonly-binary
# allowlist fast-paths ``ls``/``cat``/``grep``/… (no confirmation), and everything
# else that mutates is routed through a confirmation verdict when the gate is
# enabled.  The gate is token-aware, not prefix-aware: every chained piece
# is checked, so ``cat foo; rm -rf x`` is never treated as trusted read-only
# (and is in fact already hard-blocked by ``catastrophic_reason`` above).
#
# langbot's default policy is ``SAFE_WRITE`` with ``confirm_mutating=False`` —
# i.e. exactly the pre-Track-F behaviour: everything that is not catastrophic
# runs immediately.  ``confirm_mutating=True`` (config ``tools.confirm_mutating``)
# turns the gray zone into a confirmation path instead.

# Policy levels (mirroring neo).
READ_ONLY = "read_only"
SAFE_WRITE = "safe_write"
FULL_EXEC = "full_exec"

# Trusted read-only command *binaries* (matched after shell parsing, not by
# prefix): every pipelined/chained piece must start with one of these for the
# command to count as read-only.  ``sudo`` is unwrapped so ``sudo ls`` is
# still trusted read-only.
_TRUSTED_BINARIES = {
    "ls", "cat", "head", "tail", "pwd", "whoami", "id", "uname", "uptime",
    "date", "df", "du", "free", "ps", "which", "echo", "env", "hostname",
    "nproc", "lscpu", "lsblk", "find", "grep", "wc", "file", "stat",
}


@dataclass
class Verdict:
    """Outcome of a safety check on one shell command."""
    allowed: bool
    needs_confirm: bool = False
    reason: str = ""


def _split_commands(command: str) -> list[str]:
    """Split a shell string on separators into individual command pieces.
    Best-effort — not a real shell parser, but defeats ``safe; malicious``."""
    return [c.strip() for c in _SEPARATORS.split(command) if c.strip()]


def _first_token(cmd: str) -> str:
    try:
        tokens = shlex.split(cmd)
    except ValueError:
        tokens = cmd.split()
    if not tokens:
        return ""
    tok = tokens[0]
    if tok == "sudo" and len(tokens) > 1:
        return tokens[1]
    return tok


def is_trusted_readonly(command: str) -> bool:
    """True only if EVERY pipelined/chained piece is a trusted read-only binary
    with no output redirection.  ``cat foo; rm -rf x`` → False (the rm piece
    is not trusted); ``ls -la | grep foo`` → True (both pieces trusted);
    ``cat foo > out.txt`` → False (redirection is a write)."""
    pieces = _split_commands(command)
    if not pieces:
        return False
    for piece in pieces:
        if _first_token(piece) not in _TRUSTED_BINARIES:
            return False
        if re.search(r"[>]|>>", piece):
            return False
    return True


class SafetyGate:
    """Token-aware confirmation gate for shell commands.

    ``policy`` mirrors neo: ``read_only`` refuses anything but trusted read-only
    commands; ``safe_write``/``full_exec`` fast-path trusted read-only and route
    the rest through ``needs_confirm`` when ``confirm_mutating`` is set.  The
    catastrophic denylist (``catastrophic_reason``) always wins — hard block,
    no confirmation, ever.
    """

    def __init__(self, policy: str = SAFE_WRITE, confirm_mutating: bool = False):
        self.policy = policy
        self.confirm_mutating = confirm_mutating

    def check_shell(self, command: str) -> Verdict:
        reason = catastrophic_reason(command)
        if reason:
            return Verdict(allowed=False, reason=f"BLOCKED: {reason}")
        if self.policy == READ_ONLY:
            if is_trusted_readonly(command):
                return Verdict(allowed=True)
            return Verdict(allowed=False,
                           reason="read-only policy: only trusted read commands allowed")
        # safe_write / full_exec
        if is_trusted_readonly(command):
            return Verdict(allowed=True)
        if self.confirm_mutating:
            return Verdict(allowed=True, needs_confirm=True,
                           reason="mutating command requires confirmation")
        return Verdict(allowed=True)
