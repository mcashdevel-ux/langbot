"""Track F — token-aware confirmation gate (components/safety.py SafetyGate).

The gate has three outcomes for a shell command:
  - hard block (catastrophic — refused outright, no confirmation, ever)
  - fast allow  (trusted read-only binary — no confirmation)
  - confirmation path (gray zone — needs_confirm when the gate is enabled)

The headline regression: ``cat foo; rm -rf /`` is NEVER treated as trusted
read-only (and is in fact hard-blocked by the catastrophic denylist), because the
gate checks every chained piece, not just the prefix.
"""

from components.safety import (
    SafetyGate,
    Verdict,
    catastrophic_reason,
    is_trusted_readonly,
)


def test_catastrophic_chained_after_trusted_is_blocked():
    """``cat foo; rm -rf /`` — the old prefix-style trusted check would have
    fast-allowed this; the token-aware gate must not."""
    gate = SafetyGate(confirm_mutating=True)
    verdict = gate.check_shell("cat foo; rm -rf /")
    assert not verdict.allowed
    assert "BLOCKED" in verdict.reason


def test_trusted_readonly_is_fast_allowed():
    gate = SafetyGate(confirm_mutating=True)
    for cmd in ("ls -la", "cat foo.txt", "grep -r foo .", "pwd",
                 "df -h", "ps aux | grep python", "sudo ls /root"):
        verdict = gate.check_shell(cmd)
        assert verdict.allowed, cmd
        assert not verdict.needs_confirm, cmd


def test_gray_zone_requires_confirmation_when_enabled():
    gate = SafetyGate(confirm_mutating=True)
    for cmd in ("rm -rf ./build", "git push --force origin main",
                 "mkdir -p /tmp/x && touch /tmp/x/y", "python setup.py install"):
        verdict = gate.check_shell(cmd)
        assert verdict.allowed, cmd            # allowed in principle…
        assert verdict.needs_confirm, cmd      # …but needs user confirmation


def test_gray_zone_runs_immediately_when_gate_off():
    """Default langbot policy: autonomous — everything non-catastrophic runs."""
    gate = SafetyGate(confirm_mutating=False)

    for cmd in ("rm -rf ./build", "git push --force origin main", "ls -la"):
        verdict = gate.check_shell(cmd)
        assert verdict.allowed, cmd
        assert not verdict.needs_confirm, cmd


def test_read_only_policy_refuses_mutating():
    gate = SafetyGate(policy="read_only", confirm_mutating=True)
    assert gate.check_shell("ls -la").allowed
    assert not gate.check_shell("rm -rf ./build").allowed


def test_redirection_makes_trusted_binary_non_readonly():
    assert not is_trusted_readonly("cat foo > out.txt")
    assert not is_trusted_readonly("echo hi >> log.txt")
    assert is_trusted_readonly("cat foo")


def test_catastrophic_never_confirms():
    """Even with the gate on, catastrophic commands are hard-blocked, never
    routed to the confirmation path."""
    gate = SafetyGate(confirm_mutating=True)
    for cmd in ("rm -rf /", ":(){ :|:& };:", "dd if=/dev/zero of=/dev/sda"):
        verdict = gate.check_shell(cmd)
        assert not verdict.allowed, cmd
        assert not verdict.needs_confirm, cmd


def test_verdict_dataclass_defaults():
    v = Verdict(allowed=True)
    assert v.needs_confirm is False

    assert v.reason == ""


def test_catastrophic_reason_still_works():
    assert catastrophic_reason("rm -rf /home") is not None
    assert catastrophic_reason("ls -la") is None
