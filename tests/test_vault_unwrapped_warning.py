"""Tests for the unwrapped-master-key startup banner (TODO item 1, Option B).

``bootstrap()`` prints a loud warning when the vault master key is stored in
recoverable (``raw``) form, unless ``vault.warn_unwrapped`` is false. A
password-wrapped key never triggers it.
"""

import pytest

import components.vault as vault


@pytest.fixture
def vault_dir(tmp_path, monkeypatch):
    vdir = tmp_path / "vault"
    monkeypatch.setattr(vault, "VAULT_DIR", vdir)
    monkeypatch.setattr(vault, "MASTERKEY_FILE", vdir / ".masterkey")
    monkeypatch.setattr(vault, "CREDENTIALS_FILE", vdir / "credentials.json")
    monkeypatch.setattr(vault, "METADATA_FILE", vdir / "metadata.json")
    monkeypatch.delenv(vault.VAULT_PASSWORD_ENV, raising=False)
    monkeypatch.setattr(vault, "_store", None)
    monkeypatch.setattr(vault, "_redactor", None)
    monkeypatch.setattr(vault, "_ENV_LOADED", [])
    monkeypatch.setattr(vault, "_WARNED_UNWRAPPED", False)
    return vdir


def _capture_warnings(monkeypatch):
    seen = []
    monkeypatch.setattr(vault, "_warn_unwrapped_key",
                        lambda: seen.append("warned"))
    return seen


def test_raw_key_is_detected(vault_dir):
    vault.VaultStore().init_vault()
    assert vault.masterkey_is_unwrapped() is True


def test_password_wrapped_key_is_not_flagged(vault_dir):
    vault.VaultStore().init_vault(password="hunter2")
    assert vault.masterkey_is_unwrapped() is False


def test_missing_masterkey_is_not_flagged(vault_dir):
    assert vault.masterkey_is_unwrapped() is False


def test_bootstrap_warns_on_unwrapped_key(vault_dir, monkeypatch):
    seen = _capture_warnings(monkeypatch)
    vault.bootstrap()
    assert seen == ["warned"]


def test_bootstrap_silent_when_disabled(vault_dir, monkeypatch):
    seen = _capture_warnings(monkeypatch)
    monkeypatch.setattr(vault.config, "get",
                        lambda key, default, env=None: False
                        if key == "vault.warn_unwrapped" else default)
    vault.bootstrap()
    assert seen == []


def test_bootstrap_silent_for_wrapped_key(vault_dir, monkeypatch):
    seen = _capture_warnings(monkeypatch)
    vault.VaultStore().init_vault(password="hunter2")
    monkeypatch.setenv(vault.VAULT_PASSWORD_ENV, "hunter2")
    vault.bootstrap()
    assert seen == []


def test_warning_printed_at_most_once(vault_dir, monkeypatch):
    monkeypatch.setattr(vault, "_WARNED_UNWRAPPED", False)
    calls = []
    monkeypatch.setattr(vault.logger, "warning",
                        lambda msg, *a, **k: calls.append(msg))
    # Force the console path to fail so the logger fallback is exercised.
    from components import console
    monkeypatch.setattr(console, "warning",
                        lambda msg: (_ for _ in ()).throw(RuntimeError("no ui")))
    vault._warn_unwrapped_key()
    vault._warn_unwrapped_key()
    assert len(calls) == 1
    assert "UNWRAPPED" in calls[0]
