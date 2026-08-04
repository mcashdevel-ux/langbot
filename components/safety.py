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
