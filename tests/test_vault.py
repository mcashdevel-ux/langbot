"""Unit tests for vault.py — encryption utilities, VaultStore, RedactionFilter
and the tool dispatcher.

The vault module keeps its on-disk locations in module-level globals
(VAULT_DIR / MASTERKEY_FILE / CREDENTIALS_FILE / METADATA_FILE). The
``vault_dir`` fixture redirects all of them into a temp directory so tests
never touch the real ``memory/vault`` folder.
"""

import base64
import os

import pytest

import components.vault as vault


class _FakeAgent:
    """Minimal stand-in for the agent object passed to register/start/stop."""

    def __init__(self, config=None):
        self.registered = {}
        self._tool_observers = []
        self.config = config

    def register_tool(self, name, definition, handler):
        self.registered[name] = (definition, handler)


@pytest.fixture
def vault_dir(tmp_path, monkeypatch):
    """Redirect vault storage into an isolated temp directory."""
    vdir = tmp_path / "vault"
    monkeypatch.setattr(vault, "VAULT_DIR", vdir)
    monkeypatch.setattr(vault, "MASTERKEY_FILE", vdir / ".masterkey")
    monkeypatch.setattr(vault, "CREDENTIALS_FILE", vdir / "credentials.json")
    monkeypatch.setattr(vault, "METADATA_FILE", vdir / "metadata.json")
    return vdir


@pytest.fixture
def unlocked_store(vault_dir):
    store = vault.VaultStore()
    msg = store.init_vault()
    assert "initialized" in msg.lower()
    assert not store.is_locked()
    return store


# ---------------------------------------------------------------------------
# Encryption utilities
# ---------------------------------------------------------------------------

class TestEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        key = vault.generate_master_key()
        blob = vault.encrypt_value(key, "hunter2")
        assert isinstance(blob, str)
        assert vault.decrypt_value(key, blob) == "hunter2"

    def test_encrypt_produces_different_blobs_each_time(self):
        key = vault.generate_master_key()
        # Random salt + nonce → ciphertext differs even for identical plaintext.
        assert vault.encrypt_value(key, "same") != vault.encrypt_value(key, "same")

    def test_decrypt_with_wrong_key_returns_none(self):
        blob = vault.encrypt_value(vault.generate_master_key(), "secret")
        assert vault.decrypt_value(vault.generate_master_key(), blob) is None

    def test_decrypt_tampered_blob_returns_none(self):
        key = vault.generate_master_key()
        blob = vault.encrypt_value(key, "secret")
        # Blobs are "v2:"-prefixed AES-GCM; manipulate the base64 payload only.
        prefix, b64 = blob[:len(vault.V2_PREFIX)], blob[len(vault.V2_PREFIX):]
        raw = bytearray(base64.urlsafe_b64decode(b64.encode()))
        raw[-1] ^= 0xFF  # flip a byte in the GCM tag
        tampered = prefix + base64.urlsafe_b64encode(bytes(raw)).decode()
        assert vault.decrypt_value(key, tampered) is None

    def test_decrypt_too_short_blob_returns_none(self):
        key = vault.generate_master_key()
        short = base64.urlsafe_b64encode(b"abc").decode()
        assert vault.decrypt_value(key, short) is None

    def test_decrypt_garbage_returns_none(self):
        assert vault.decrypt_value(vault.generate_master_key(), "!!!not base64!!!") is None

    def test_generate_master_key_size(self):
        assert len(vault.generate_master_key()) == vault.KEY_SIZE

    def test_ctr_cipher_is_symmetric(self):
        # Legacy SHA256-CTR keystream XOR is symmetric (encrypt == decrypt).
        key = b"k" * 32
        nonce = b"n" * 16
        ct = vault._sha256_ctr_crypt(key, nonce, b"plaintext data")
        assert vault._sha256_ctr_crypt(key, nonce, ct) == b"plaintext data"

    def test_empty_string_roundtrip(self):
        key = vault.generate_master_key()
        assert vault.decrypt_value(key, vault.encrypt_value(key, "")) == ""

    def test_unicode_roundtrip(self):
        key = vault.generate_master_key()
        secret = "pä$$wörd-🔑-你好"
        assert vault.decrypt_value(key, vault.encrypt_value(key, secret)) == secret


# ---------------------------------------------------------------------------
# VaultStore lifecycle + CRUD
# ---------------------------------------------------------------------------

class TestVaultStore:
    def test_new_store_is_locked_before_init(self, vault_dir):
        store = vault.VaultStore()
        assert store.is_locked()

    def test_init_creates_masterkey_file(self, vault_dir):
        store = vault.VaultStore()
        store.init_vault()
        assert vault.MASTERKEY_FILE.exists()

    def test_double_init_is_noop(self, unlocked_store):
        again = unlocked_store.init_vault()
        assert "already initialized" in again.lower()

    def test_put_and_get(self, unlocked_store):
        assert unlocked_store.put("API_KEY", "abc123")
        assert unlocked_store.get("API_KEY") == "abc123"

    def test_get_missing_returns_none(self, unlocked_store):
        assert unlocked_store.get("NOPE") is None

    def test_put_when_locked_fails(self, vault_dir):
        store = vault.VaultStore()  # never initialized → locked
        assert store.put("X", "y") is False

    def test_get_when_locked_returns_none(self, vault_dir):
        store = vault.VaultStore()
        assert store.get("X") is None

    def test_put_rejects_oversized_value(self, unlocked_store):
        big = "x" * (vault.CREDENTIAL_MAX_VALUE_LEN + 1)
        assert unlocked_store.put("BIG", big) is False

    def test_put_rejects_oversized_name(self, unlocked_store):
        assert unlocked_store.put("N" * 257, "v") is False

    def test_put_rejects_when_at_capacity(self, unlocked_store, monkeypatch):
        monkeypatch.setattr(vault, "MAX_CREDENTIALS", 1)
        assert unlocked_store.put("ONE", "v")
        assert unlocked_store.put("TWO", "v") is False

    def test_remove(self, unlocked_store):
        unlocked_store.put("TEMP", "v")
        assert unlocked_store.remove("TEMP") is True
        assert unlocked_store.get("TEMP") is None

    def test_remove_missing_returns_false(self, unlocked_store):
        assert unlocked_store.remove("GHOST") is False

    def test_list_names(self, unlocked_store):
        unlocked_store.put("A", "1")
        unlocked_store.put("B", "2")
        names = [e["name"] for e in unlocked_store.list_names()]
        assert names == ["A", "B"]  # sorted
        assert all(e["encrypted"] for e in unlocked_store.list_names())

    def test_metadata_tracks_length_and_timestamps(self, unlocked_store):
        unlocked_store.put("K", "value12")
        entry = unlocked_store.list_names()[0]
        assert entry["length"] == len("value12")
        assert entry["created_at"] > 0
        assert entry["updated_at"] >= entry["created_at"]

    def test_get_updates_last_used(self, unlocked_store):
        unlocked_store.put("K", "v")
        assert unlocked_store.list_names()[0]["last_used"] == 0
        unlocked_store.get("K")
        assert unlocked_store.list_names()[0]["last_used"] > 0

    def test_get_all_plaintext(self, unlocked_store):
        unlocked_store.put("A", "1")
        unlocked_store.put("B", "2")
        assert unlocked_store.get_all_plaintext() == {"A": "1", "B": "2"}

    def test_get_all_plaintext_locked_is_empty(self, vault_dir):
        assert vault.VaultStore().get_all_plaintext() == {}

    def test_stats(self, unlocked_store):
        unlocked_store.put("A", "1")
        stats = unlocked_store.stats()
        assert stats["credential_count"] == 1
        assert stats["initialized"] is True
        assert stats["locked"] is False
        assert stats["master_key_file"] is True
        assert stats["key_type"] == "raw"

    def test_persistence_across_instances(self, vault_dir):
        store = vault.VaultStore()
        store.init_vault()
        store.put("PERSIST", "kept")
        # A fresh store auto-loads + auto-unlocks from disk (raw key).
        reloaded = vault.VaultStore()
        assert not reloaded.is_locked()
        assert reloaded.get("PERSIST") == "kept"


class TestPasswordProtectedVault:
    def test_password_init_and_unlock(self, vault_dir):
        store = vault.VaultStore()
        msg = store.init_vault(password="s3cret")
        assert "password" in msg.lower()
        store.put("K", "v")

        locked = vault.VaultStore()
        # password_wrapped key can't auto-unlock without a password
        assert locked.is_locked()
        assert locked.unlock() is False
        assert locked.unlock(password="wrong") is False
        assert locked.unlock(password="s3cret") is True
        assert locked.get("K") == "v"

    def test_stats_reports_password_key_type(self, vault_dir):
        store = vault.VaultStore()
        store.init_vault(password="pw")
        assert store.stats()["key_type"] == "password_wrapped"


# ---------------------------------------------------------------------------
# RedactionFilter
# ---------------------------------------------------------------------------

class TestRedactionFilter:
    def test_redact_replaces_known_value(self, unlocked_store):
        unlocked_store.put("TOKEN", "supersecretvalue")
        rf = vault.RedactionFilter(unlocked_store)
        rf.refresh_patterns()
        out = rf.redact("here is supersecretvalue in a log line")
        assert "supersecretvalue" not in out
        assert "TOKEN_REDACTED" in out

    def test_redact_noop_without_patterns(self, unlocked_store):
        rf = vault.RedactionFilter(unlocked_store)
        assert rf.redact("nothing here") == "nothing here"

    def test_short_values_not_used_as_patterns(self, unlocked_store):
        unlocked_store.put("PIN", "12")  # < 4 chars
        rf = vault.RedactionFilter(unlocked_store)
        rf.refresh_patterns()
        assert rf._patterns == {}

    def test_get_masked_value_long(self, unlocked_store):
        unlocked_store.put("K", "abcdefghij")
        rf = vault.RedactionFilter(unlocked_store)
        assert rf.get_masked_value("K") == "abcd...ghij"

    def test_get_masked_value_short(self, unlocked_store):
        unlocked_store.put("K", "abcd")
        rf = vault.RedactionFilter(unlocked_store)
        assert rf.get_masked_value("K") == "ab****"

    def test_get_masked_value_missing(self, unlocked_store):
        rf = vault.RedactionFilter(unlocked_store)
        assert rf.get_masked_value("MISSING") == ""


# ---------------------------------------------------------------------------
# _mask_value helper
# ---------------------------------------------------------------------------

class TestMaskValue:
    def test_empty(self):
        assert vault._mask_value("") == ""

    def test_short(self):
        assert vault._mask_value("abcd") == "ab**"

    def test_long(self):
        assert vault._mask_value("abcdefghij") == "abcd**ghij"


# ---------------------------------------------------------------------------
# Tool dispatcher (_vault_handler and helpers)
# ---------------------------------------------------------------------------

class TestVaultHandler:
    @pytest.fixture(autouse=True)
    def wired_store(self, unlocked_store, monkeypatch):
        """Point the module-level plugin globals at an unlocked store."""
        monkeypatch.setattr(vault, "_store", unlocked_store)
        monkeypatch.setattr(vault, "_redactor", vault.RedactionFilter(unlocked_store))
        return unlocked_store

    def test_store_get_roundtrip(self, monkeypatch):
        monkeypatch.setattr("os.environ", dict())
        out = vault._vault_handler("store", "my_key", "val")
        assert "stored securely" in out
        # stored uppercased
        assert vault._vault_handler("get", "my_key") == "val"

    def test_store_requires_name(self):
        assert "name is required" in vault._vault_handler("store", "", "v")

    def test_store_requires_value(self):
        assert "value is required" in vault._vault_handler("store", "K", "")

    def test_get_missing(self):
        assert "not found" in vault._vault_handler("get", "ghost")

    def test_list_empty(self):
        assert "No credentials" in vault._vault_handler("list")

    def test_list_populated(self):
        vault._vault_handler("store", "one", "v")
        out = vault._vault_handler("list")
        assert "ONE" in out
        assert "Total: 1" in out

    def test_remove(self):
        vault._vault_handler("store", "gone", "v")
        assert "removed" in vault._vault_handler("remove", "gone")
        assert "not found" in vault._vault_handler("remove", "gone")

    def test_status(self):
        out = vault._vault_handler("status")
        assert "VAULT STATUS" in out

    def test_unknown_action(self):
        assert "Unknown vault action" in vault._vault_handler("frobnicate")

    def test_handler_when_store_missing(self, monkeypatch):
        monkeypatch.setattr(vault, "_store", None)
        assert "not initialized" in vault._vault_handler("list")

    def test_handler_when_locked(self, monkeypatch, unlocked_store):
        # Force the wired store into a locked state.
        unlocked_store._initialized = False
        unlocked_store._master_key = None
        assert unlocked_store.is_locked()
        assert "locked" in vault._vault_handler("list")


# ---------------------------------------------------------------------------
# Plugin lifecycle: register / start / stop
# ---------------------------------------------------------------------------

class _Config:
    def __init__(self):
        self.llm_api_key = ""
        self.jina_api_key = ""


class TestPluginLifecycle:
    @pytest.fixture(autouse=True)
    def reset_globals(self, monkeypatch):
        monkeypatch.setattr(vault, "_store", None)
        monkeypatch.setattr(vault, "_redactor", None)
        monkeypatch.setattr(vault, "_agent_ref", None)
        monkeypatch.setattr(vault, "_ENV_LOADED", [])

    def test_register_wires_tools_and_observer(self, vault_dir):
        agent = _FakeAgent()
        vault.register(agent)
        assert "vault" in agent.registered
        assert vault._output_redactor in agent._tool_observers
        assert vault._store is not None

    def test_start_auto_initializes_and_loads_env(self, vault_dir, monkeypatch):
        env = {}
        monkeypatch.setattr("os.environ", env)
        cfg = _Config()
        agent = _FakeAgent(config=cfg)
        vault.register(agent)
        vault._store.init_vault()  # unlock before storing
        # Seed a credential that maps to a config field.
        vault._store.put("JINA_API_KEY", "jina-secret")

        vault.start(agent)
        assert env.get("JINA_API_KEY") == "jina-secret"
        assert "JINA_API_KEY" in vault._ENV_LOADED
        # config.jina_api_key populated from vault
        assert cfg.jina_api_key == "jina-secret"

    def test_start_without_store_is_noop(self, monkeypatch):
        monkeypatch.setattr(vault, "_store", None)
        vault.start(_FakeAgent())  # must not raise

    def test_start_does_not_override_existing_env(self, vault_dir, monkeypatch):
        env = {"JINA_API_KEY": "preexisting"}
        monkeypatch.setattr("os.environ", env)
        agent = _FakeAgent(config=_Config())
        vault.register(agent)
        vault._store.init_vault()
        vault._store.put("JINA_API_KEY", "from-vault")
        vault.start(agent)
        assert env["JINA_API_KEY"] == "preexisting"

    def test_stop_saves_state(self, vault_dir):
        agent = _FakeAgent()
        vault.register(agent)
        vault._store.init_vault()
        vault._store.put("K", "v")
        vault.stop(agent)  # should persist without error
        assert vault.CREDENTIALS_FILE.exists()

    def test_output_redactor_refreshes_patterns(self, vault_dir, monkeypatch):
        agent = _FakeAgent()
        vault.register(agent)
        vault._store.init_vault()
        vault._store.put("SECRET", "longsecretvalue")
        called = {"n": 0}
        orig = vault._redactor.refresh_patterns

        def spy():
            called["n"] += 1
            orig()

        monkeypatch.setattr(vault._redactor, "refresh_patterns", spy)
        vault._output_redactor("some_tool", {}, "result")
        assert called["n"] == 1

    def test_output_redactor_skips_credential_tools(self, vault_dir, monkeypatch):
        agent = _FakeAgent()
        vault.register(agent)
        monkeypatch.setattr(
            vault._redactor, "refresh_patterns",
            lambda: (_ for _ in ()).throw(AssertionError("should not refresh")),
        )
        # Credential-management tool names are exempt from redaction refresh.
        vault._output_redactor("get_credential", {}, "result")

    def test_output_redactor_returns_redacted_result(self, vault_dir):
        agent = _FakeAgent()
        vault.register(agent)
        vault._store.init_vault()
        vault._store.put("SECRET", "longsecretvalue")
        out = vault._output_redactor("some_tool", {}, "leaked longsecretvalue here")
        assert "longsecretvalue" not in out
        assert "SECRET_REDACTED" in out


# ---------------------------------------------------------------------------
# LangGraph adapter API (bootstrap / run_action / redact)
# ---------------------------------------------------------------------------

class TestLangGraphAdapter:
    @pytest.fixture(autouse=True)
    def _reset_globals(self, vault_dir, monkeypatch):
        # bootstrap() uses module-level singletons; reset them per test.
        monkeypatch.setattr(vault, "_store", None)
        monkeypatch.setattr(vault, "_redactor", None)
        monkeypatch.setattr(vault, "_ENV_LOADED", [])

    def test_bootstrap_autoinits_and_loads_env(self, monkeypatch):
        monkeypatch.delenv("MY_KEY", raising=False)
        assert vault.bootstrap() == []          # fresh vault, nothing stored
        vault.run_action("store", "MY_KEY", "abc123value")
        # A subsequent bootstrap (e.g. new process) exports it to the env.
        monkeypatch.delenv("MY_KEY", raising=False)
        loaded = vault.bootstrap()
        assert "MY_KEY" in loaded
        assert os.environ["MY_KEY"] == "abc123value"

    def test_redact_before_bootstrap_is_noop(self):
        assert vault.redact("nothing to redact") == "nothing to redact"

    def test_redact_scrubs_stored_values(self):
        vault.bootstrap()
        vault.run_action("store", "TOKEN", "secretTokenXYZ")
        assert "secretTokenXYZ" not in vault.redact("here: secretTokenXYZ end")

    def test_redact_prefers_longer_values(self):
        vault.bootstrap()
        vault.run_action("store", "SHORT", "abcd")
        vault.run_action("store", "LONG", "abcdefgh")
        out = vault.redact("value abcdefgh here")
        # The longer secret must be fully masked, not partially by the shorter.
        assert "LONG_REDACTED" in out
        assert "efgh" not in out

    def test_run_action_unknown(self):
        vault.bootstrap()
        assert "Unknown vault action" in vault.run_action("frobnicate")
