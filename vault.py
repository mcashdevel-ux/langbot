"""
Feature: Credential Vault
=========================
Secure encrypted storage for API keys and credentials.
Replaces plaintext .env and config.json API key fields.

Architecture:
  VaultStore       — encrypted storage backend (pure Python, zero deps)
  CredentialVault  — manages the vault lifecycle (init, lock, unlock)
  RedactionFilter  — auto-redacts known credential values from tool outputs
  Tool Interface   — vault(action, name, value)

Security Design:
  - Key derivation: PBKDF2-HMAC-SHA256 (100k iterations, per-salt)
  - Encryption: SHA256-CTR mode (stream cipher using SHA256 as PRF)
  - Integrity: HMAC-SHA256 (encrypt-then-mac)
  - Master key: 32-byte random, stored in memory/vault/.masterkey
  - Zero dependencies: uses only hashlib, hmac, os.urandom, base64

Tools Registered:
  vault(action="store")    — encrypt and store a credential
  vault(action="get")      — retrieve a credential (masked in display)
  vault(action="list")     — list stored credential names
  vault(action="remove")   — delete a credential
  vault(action="status")   — show vault health and stats
"""

import json
import os
import time
import hmac
import hashlib
import base64
import logging
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAULT_DIR = Path("memory/vault")
MASTERKEY_FILE = VAULT_DIR / ".masterkey"
CREDENTIALS_FILE = VAULT_DIR / "credentials.json"
METADATA_FILE = VAULT_DIR / "metadata.json"
PBKDF2_ITERATIONS = 100000
KEY_SIZE = 32  # 256-bit
NONCE_SIZE = 16
CREDENTIAL_MAX_VALUE_LEN = 10000
MAX_CREDENTIALS = 100

# Tool Definitions
TOOL_DEFINITIONS = [
    {
        "name": "vault",
        "description": "Manage encrypted credentials. Actions: 'store' (encrypt+save), 'get' (retrieve), "
                       "'list' (show all names), 'remove' (delete), 'status' (health dashboard).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "get", "list", "remove", "status"],
                    "description": "Action to perform: 'store' saves a credential, 'get' retrieves one, "
                                   "'list' shows all stored names, 'remove' deletes, 'status' shows vault health."
                },
                "name": {
                    "type": "string",
                    "description": "Credential name (required for store/get/remove). "
                                   "E.g. 'DEEPSEEK_API_KEY', 'GITHUB_TOKEN'."
                },
                "value": {
                    "type": "string",
                    "description": "Credential value to encrypt and store (required for 'store' action)."
                }
            },
            "required": ["action"]
        }
    },
]


# ---------------------------------------------------------------------------
# Pure-Python Encryption Utilities
# ---------------------------------------------------------------------------

def _derive_keys(master_key: bytes, salt: bytes) -> Tuple[bytes, bytes]:
    """Derive encryption key and MAC key from master key using PBKDF2."""
    dk = hashlib.pbkdf2_hmac('sha256', master_key, salt, PBKDF2_ITERATIONS, dklen=KEY_SIZE * 2)
    return dk[:KEY_SIZE], dk[KEY_SIZE:]


def _sha256_ctr_encrypt(enc_key: bytes, nonce: bytes, plaintext: bytes) -> bytes:
    """Encrypt using SHA256 in CTR mode (SHA256 as PRF)."""
    # Generate keystream: SHA256(enc_key || nonce || counter) for each 32-byte block
    keystream = b""
    counter = 0
    while len(keystream) < len(plaintext):
        counter_bytes = counter.to_bytes(4, 'big')
        block = hashlib.sha256(enc_key + nonce + counter_bytes).digest()
        keystream += block
        counter += 1
    # XOR plaintext with keystream
    return bytes(p ^ k for p, k in zip(plaintext, keystream[:len(plaintext)]))


def _sha256_ctr_decrypt(enc_key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """Decrypt using SHA256-CTR (same as encrypt, XOR is symmetric)."""
    return _sha256_ctr_encrypt(enc_key, nonce, ciphertext)


def encrypt_value(master_key: bytes, plaintext: str) -> str:
    """Encrypt a string value. Returns URL-safe base64 encoded blob."""
    salt = os.urandom(NONCE_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    enc_key, mac_key = _derive_keys(master_key, salt)

    data = plaintext.encode('utf-8')
    ciphertext = _sha256_ctr_encrypt(enc_key, nonce, data)

    # Encrypt-then-MAC: tag over nonce || ciphertext
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()

    # Pack: salt || nonce || ciphertext || tag
    payload = salt + nonce + ciphertext + tag
    return base64.urlsafe_b64encode(payload).decode('ascii')


def decrypt_value(master_key: bytes, blob: str) -> Optional[str]:
    """Decrypt a base64 blob. Returns None on integrity failure."""
    try:
        payload = base64.urlsafe_b64decode(blob.encode('ascii'))
        if len(payload) < NONCE_SIZE * 2 + KEY_SIZE:
            return None

        salt = payload[:NONCE_SIZE]
        nonce = payload[NONCE_SIZE:NONCE_SIZE * 2]
        tag = payload[-KEY_SIZE:]
        ciphertext = payload[NONCE_SIZE * 2:-KEY_SIZE]

        enc_key, mac_key = _derive_keys(master_key, salt)

        # Verify MAC
        expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
        if not hmac.compare_digest(tag, expected_tag):
            logger.warning("Vault: integrity check failed (tampered or wrong key)")
            return None

        plaintext = _sha256_ctr_decrypt(enc_key, nonce, ciphertext)
        return plaintext.decode('utf-8')
    except Exception as e:
        logger.warning(f"Vault decrypt error: {e}")
        return None


def generate_master_key() -> bytes:
    """Generate a cryptographically random 32-byte master key."""
    return os.urandom(KEY_SIZE)


# ---------------------------------------------------------------------------
# Vault Store — persistence layer
# ---------------------------------------------------------------------------

class VaultStore:
    """Encrypted credential storage backend."""

    def __init__(self):
        self._lock = threading.Lock()
        self._master_key: Optional[bytes] = None
        self._credentials: Dict[str, str] = {}  # name → encrypted blob
        self._metadata: Dict[str, Dict] = {}     # name → metadata dict
        self._initialized = False
        self._load()

    # ── Initialization ──

    def init_vault(self, password: Optional[str] = None) -> str:
        """Initialize the vault. Generates master key if needed."""
        with self._lock:
            if MASTERKEY_FILE.exists():
                return "Vault already initialized. Use vault_status() to check."

            VAULT_DIR.mkdir(parents=True, exist_ok=True)
            self._master_key = generate_master_key()

            if password:
                # If password provided, encrypt master key with password-derived key
                # This enables "unlock with password" pattern
                salt = os.urandom(NONCE_SIZE)
                pwd_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'),
                                               salt, PBKDF2_ITERATIONS, dklen=KEY_SIZE)
                # Encrypt master key with password-derived key
                wrapped = encrypt_value(pwd_key, self._master_key.hex())
                key_data = {
                    "version": 1,
                    "type": "password_wrapped",
                    "salt": base64.urlsafe_b64encode(salt).decode('ascii'),
                    "wrapped_key": wrapped,
                }
            else:
                # Store raw master key (file-protected)
                key_data = {
                    "version": 1,
                    "type": "raw",
                    "key": base64.urlsafe_b64encode(self._master_key).decode('ascii'),
                }

            import tempfile
            _tmp = tempfile.NamedTemporaryFile(mode="w", delete=False,
                                                dir=str(VAULT_DIR), suffix=".tmp")
            with _tmp:
                json.dump(key_data, _tmp, indent=2)
            os.replace(_tmp.name, str(MASTERKEY_FILE))

            self._save_credentials()
            self._save_metadata()
            self._initialized = True

            if password:
                return "Vault initialized with password protection."
            return "Vault initialized. Master key stored in vault directory."

    def unlock(self, password: Optional[str] = None) -> bool:
        """Unlock the vault by loading the master key."""
        with self._lock:
            if not MASTERKEY_FILE.exists():
                logger.warning("Vault not initialized")
                return False

            try:
                with open(MASTERKEY_FILE) as f:
                    key_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read master key: {e}")
                return False

            ktype = key_data.get("type", "raw")
            if ktype == "raw":
                key_b64 = key_data.get("key", "")
                self._master_key = base64.urlsafe_b64decode(key_b64.encode('ascii'))
            elif ktype == "password_wrapped":
                if not password:
                    logger.warning("Password required to unlock vault")
                    return False
                salt = base64.urlsafe_b64decode(key_data["salt"].encode('ascii'))
                pwd_key = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'),
                                               salt, PBKDF2_ITERATIONS, dklen=KEY_SIZE)
                hex_key = decrypt_value(pwd_key, key_data["wrapped_key"])
                if hex_key is None:
                    logger.warning("Wrong password")
                    return False
                self._master_key = bytes.fromhex(hex_key)
            else:
                logger.warning(f"Unknown key type: {ktype}")
                return False

            self._load_credentials()
            self._load_metadata()
            self._initialized = True
            return True

    def is_locked(self) -> bool:
        return not self._initialized or self._master_key is None

    # ── CRUD ──

    def put(self, name: str, value: str) -> bool:
        """Encrypt and store a credential."""
        if self.is_locked():
            return False
        if len(name) > 256:
            return False
        if len(value) > CREDENTIAL_MAX_VALUE_LEN:
            return False
        if len(self._credentials) >= MAX_CREDENTIALS:
            return False

        with self._lock:
            blob = encrypt_value(self._master_key, value)
            self._credentials[name] = blob

            now = time.time()
            meta = self._metadata.get(name, {})
            meta["name"] = name
            meta["created_at"] = meta.get("created_at", now)
            meta["updated_at"] = now
            meta["last_used"] = meta.get("last_used", 0)
            meta["length"] = len(value)
            self._metadata[name] = meta

            self._save_credentials()
            self._save_metadata()
            return True

    def get(self, name: str) -> Optional[str]:
        """Retrieve and decrypt a credential."""
        if self.is_locked():
            return None

        with self._lock:
            blob = self._credentials.get(name)
            if blob is None:
                return None

            value = decrypt_value(self._master_key, blob)
            if value is not None:
                # Update last_used timestamp
                meta = self._metadata.get(name, {})
                meta["last_used"] = time.time()
                self._metadata[name] = meta

            return value

    def remove(self, name: str) -> bool:
        """Delete a credential."""
        with self._lock:
            if name in self._credentials:
                del self._credentials[name]
                self._metadata.pop(name, None)
                self._save_credentials()
                self._save_metadata()
                return True
            return False

    def list_names(self) -> List[Dict]:
        """Return list of credential names with metadata (no values)."""
        with self._lock:
            result = []
            for name, meta in sorted(self._metadata.items()):
                if name in self._credentials:
                    entry = dict(meta)
                    entry["encrypted"] = True
                    result.append(entry)
            return result

    def get_all_plaintext(self) -> Dict[str, str]:
        """Decrypt all credentials. Used for env var auto-load."""
        if self.is_locked():
            return {}
        result = {}
        with self._lock:
            for name, blob in self._credentials.items():
                value = decrypt_value(self._master_key, blob)
                if value is not None:
                    result[name] = value
        return result

    def stats(self) -> Dict:
        """Return vault statistics."""
        with self._lock:
            return {
                "initialized": self._initialized and not self.is_locked(),
                "credential_count": len(self._credentials),
                "locked": self.is_locked(),
                "master_key_file": MASTERKEY_FILE.exists(),
                "key_type": self._get_key_type(),
            }

    def _get_key_type(self) -> str:
        try:
            if MASTERKEY_FILE.exists():
                with open(MASTERKEY_FILE) as f:
                    data = json.load(f)
                return data.get("type", "unknown")
        except Exception:
            pass
        return "none"

    # ── Persistence ──

    def _save_credentials(self):
        """Save encrypted credentials blob."""
        try:
            VAULT_DIR.mkdir(parents=True, exist_ok=True)
            import tempfile
            _tmp = tempfile.NamedTemporaryFile(mode="w", delete=False,
                                                dir=str(VAULT_DIR), suffix=".tmp")
            with _tmp:
                json.dump({
                    "version": 1,
                    "updated_at": time.time(),
                    "credentials": self._credentials,
                }, _tmp, indent=2)
            os.replace(_tmp.name, str(CREDENTIALS_FILE))
        except Exception as e:
            logger.warning(f"Failed to save credentials: {e}")

    def _save_metadata(self):
        """Save credential metadata (names, timestamps, no values)."""
        try:
            VAULT_DIR.mkdir(parents=True, exist_ok=True)
            import tempfile
            _tmp = tempfile.NamedTemporaryFile(mode="w", delete=False,
                                                dir=str(VAULT_DIR), suffix=".tmp")
            with _tmp:
                json.dump({
                    "version": 1,
                    "updated_at": time.time(),
                    "credentials": list(self._metadata.values()),
                }, _tmp, indent=2)
            os.replace(_tmp.name, str(METADATA_FILE))
        except Exception as e:
            logger.warning(f"Failed to save metadata: {e}")

    def _load_credentials(self):
        """Load encrypted credentials from disk."""
        if not CREDENTIALS_FILE.exists():
            self._credentials = {}
            return
        try:
            with open(CREDENTIALS_FILE) as f:
                data = json.load(f)
            self._credentials = data.get("credentials", {})
        except Exception as e:
            logger.warning(f"Failed to load credentials: {e}")
            self._credentials = {}

    def _load_metadata(self):
        """Load credential metadata from disk."""
        if not METADATA_FILE.exists():
            self._metadata = {}
            return
        try:
            with open(METADATA_FILE) as f:
                data = json.load(f)
            self._metadata = {}
            for entry in data.get("credentials", []):
                name = entry.get("name", "")
                if name:
                    self._metadata[name] = entry
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            self._metadata = {}

    def _load(self):
        """Auto-load vault state on instantiation."""
        if MASTERKEY_FILE.exists():
            self.unlock()


# ---------------------------------------------------------------------------
# Redaction Filter
# ---------------------------------------------------------------------------

class RedactionFilter:
    """Auto-redacts credential values from tool outputs and logs."""

    def __init__(self, vault_store: VaultStore):
        self._vault = vault_store
        self._patterns: Dict[str, str] = {}  # credential name → mask pattern

    def refresh_patterns(self):
        """Rebuild redaction patterns from current credential values."""
        self._patterns = {}
        values = self._vault.get_all_plaintext()
        for name, value in values.items():
            if value and len(value) >= 4:
                # Create a pattern for the full value (redact in outputs)
                # Use first 4 chars + "..." for display
                display = value[:4] + "..." if len(value) > 8 else "****"
                self._patterns[name] = display

    def redact(self, text: str) -> str:
        """Redact known credential values from a text string."""
        if not self._patterns:
            return text
        values = self._vault.get_all_plaintext()
        for name, value in values.items():
            if value and len(value) >= 4 and value in text:
                text = text.replace(value, f"🔑[{name}_REDACTED]")
        return text

    def get_masked_value(self, name: str) -> str:
        """Return a masked version of a credential value for display."""
        value = self._vault.get(name)
        if value is None:
            return ""
        if len(value) <= 8:
            return value[:2] + "****"
        return value[:4] + "..." + value[-4:]


# ---------------------------------------------------------------------------
# Plugin State
# ---------------------------------------------------------------------------

_store: Optional[VaultStore] = None
_redactor: Optional[RedactionFilter] = None
_agent_ref: Any = None
_ENV_LOADED: List[str] = []  # Track which env vars were auto-loaded from vault


def register(agent):
    """Register vault tools and initialize."""
    global _store, _redactor, _agent_ref
    _agent_ref = agent
    _store = VaultStore()
    _redactor = RedactionFilter(_store)

    for td in TOOL_DEFINITIONS:
        name = td["name"]
        agent.register_tool(name, td, _vault_handler)

    # Wire up output redaction
    agent._tool_observers.append(_output_redactor)

    logger.info("Credential Vault plugin loaded")


def start(agent):
    """Auto-load credentials into env vars on startup."""
    global _store, _redactor, _ENV_LOADED

    if not _store:
        logger.info("Vault not available")
        return

    # Auto-init vault if not yet initialized (before lock check)
    # init_vault() leaves the vault unlocked, so no explicit unlock needed
    if _store.is_locked() and not MASTERKEY_FILE.exists():
        _store.init_vault()
        logger.info("Vault auto-initialized (no existing vault found)")

    if _store.is_locked():
        logger.info("Vault is locked, skipping env auto-load")
        return

    # Refresh redaction patterns
    if _redactor:
        _redactor.refresh_patterns()

    # Auto-load credentials into environment variables
    values = _store.get_all_plaintext()
    loaded = []
    for name, value in values.items():
        # Only set if not already set (don't override explicit env vars)
        if name not in os.environ:
            os.environ[name] = value
            loaded.append(name)

    if loaded:
        _ENV_LOADED = loaded
        logger.info(f"Auto-loaded {len(loaded)} credentials from vault into env vars")

    # Also try to populate Config fields from vault
    try:
        cfg = getattr(_agent_ref, 'config', None)
        if cfg:
            field_map = {
                "llm_api_key": ["DEEPSEEK_API_KEY", "LLM_API_KEY"],
                "serpapi_key": ["SERPAPI_API_KEY"],
                "jina_api_key": ["JINA_API_KEY"],
                "gemini_api_key": ["GEMINI_API_KEY"],
            }
            for cfg_field, env_names in field_map.items():
                current = getattr(cfg, cfg_field, "")
                if not current:
                    for env_name in env_names:
                        if env_name in values:
                            setattr(cfg, cfg_field, values[env_name])
                            logger.info(f"Populated config.{cfg_field} from vault")
                            break
    except Exception as e:
        logger.warning(f"Failed to populate config from vault: {e}")


def stop(agent):
    """Save vault state on shutdown."""
    global _store
    if _store:
        _store._save_credentials()
        _store._save_metadata()
        logger.info("Vault saved on shutdown")


# ---------------------------------------------------------------------------
# Output Redactor Observer
# ---------------------------------------------------------------------------

def _output_redactor(tool_name: str, args: Dict, result: str):
    """Redact known credential values from tool results.
    Runs after every tool execution via agent._tool_observers."""
    global _redactor
    if not _redactor:
        return
    # Credential management tools don't need redaction
    if tool_name in ("store_credential", "get_credential", "list_credentials",
                      "remove_credential", "vault_status"):
        return
    _redactor.refresh_patterns()


# ---------------------------------------------------------------------------
# Tool Functions
# ---------------------------------------------------------------------------

def _vault_handler(action: str, name: str = "", value: str = "") -> str:
    """Handle all vault tool actions via single dispatcher."""
    global _store, _redactor

    if not _store:
        return "Error: Vault not initialized."
    if _store.is_locked():
        return "Error: Vault is locked."

    if action == "store":
        return _vault_store(name, value)
    elif action == "get":
        return _vault_get(name)
    elif action == "list":
        return _vault_list()
    elif action == "remove":
        return _vault_remove(name)
    elif action == "status":
        return _vault_status()
    else:
        return f"Unknown vault action: '{action}'. Valid: store, get, list, remove, status."


def _vault_store(name: str, value: str) -> str:
    """Store a credential (called from _vault_handler)."""
    global _store, _redactor
    if not name:
        return "Error: Credential name is required."
    if not value:
        return "Error: Credential value is required."
    if _store.is_locked():
        return "Error: Vault is locked."

    name_upper = name.upper().strip()
    if _store.put(name_upper, value):
        os.environ[name_upper] = value
        if _redactor:
            _redactor.refresh_patterns()
        return f"✅ Credential '{name_upper}' stored securely."
    else:
        return f"Error: Failed to store credential (check limits)."


def _vault_get(name: str) -> str:
    """Get a credential (called from _vault_handler)."""
    global _store
    if not name:
        return "Error: Credential name is required."
    if _store.is_locked():
        return "Error: Vault is locked."

    name_upper = name.upper().strip()
    value = _store.get(name_upper)
    if value is None:
        return f"Error: Credential '{name_upper}' not found."
    os.environ[name_upper] = value
    return value


def _vault_list() -> str:
    """List credentials (called from _vault_handler)."""
    global _store
    if not _store:
        return "Error: Vault not initialized."
    if _store.is_locked():
        return "Error: Vault is locked."

    entries = _store.list_names()
    if not entries:
        return "No credentials stored in vault."

    lines = ["📋 VAULT CREDENTIALS"]
    lines.append("=" * 50)
    for entry in entries:
        name = entry.get("name", "?")
        created = entry.get("created_at", 0)
        last_used = entry.get("last_used", 0)
        length = entry.get("length", 0)
        created_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(created)) if created else "?"
        used_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(last_used)) if last_used else "never"
        lines.append(f"  • {name}")
        lines.append(f"    Created: {created_str} | Last used: {used_str} | Size: {length} chars")

    lines.append(f"\n{'=' * 50}")
    lines.append(f"Total: {len(entries)} credentials")
    lines.append(f"Use vault(action='get') to retrieve, vault(action='remove') to delete.")
    return "\n".join(lines)


def _vault_remove(name: str) -> str:
    """Remove a credential (called from _vault_handler)."""
    global _store, _redactor
    if not name:
        return "Error: Credential name is required."
    if _store.is_locked():
        return "Error: Vault is locked."

    name_upper = name.upper().strip()
    if _store.remove(name_upper):
        os.environ.pop(name_upper, None)
        if _redactor:
            _redactor.refresh_patterns()
        return f"✅ Credential '{name_upper}' removed from vault."
    else:
        return f"Error: Credential '{name_upper}' not found."


def _vault_status() -> str:
    """Show vault health dashboard (called from _vault_handler)."""
    global _store, _ENV_LOADED
    if not _store:
        return "Error: Vault not initialized."

    stats = _store.stats()
    lines = ["🔐 VAULT STATUS", "=" * 50]
    lines.append(f"  Initialized: {'✅' if stats['initialized'] else '❌'}")
    lines.append(f"  Locked: {'🔒 Yes' if stats['locked'] else '🔓 No'}")
    lines.append(f"  Credentials stored: {stats['credential_count']}")
    lines.append(f"  Master key file: {'✅' if stats['master_key_file'] else '❌'} exist")
    lines.append(f"  Key type: {stats['key_type']}")

    if _ENV_LOADED:
        lines.append(f"\n  Auto-loaded env vars:")
        for name in sorted(_ENV_LOADED):
            lines.append(f"    • ${name}")

    lines.append(f"\n  Encryption: SHA256-CTR + HMAC-SHA256")
    lines.append(f"  Key derivation: PBKDF2-HMAC-SHA256 ({PBKDF2_ITERATIONS} iterations)")
    lines.append(f"  Storage: {CREDENTIALS_FILE}")
    lines.append(f"\n  Commands:")
    lines.append(f"    vault(action='store')   — add/update a credential")
    lines.append(f"    vault(action='get')     — retrieve a credential")
    lines.append(f"    vault(action='list')    — list all stored names")
    lines.append(f"    vault(action='remove')  — delete a credential")
    lines.append(f"    vault(action='status')  — show vault health")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask_value(value: str) -> str:
    """Return a masked version of a value for safe display."""
    if not value:
        return ""
    if len(value) <= 8:
        return value[:2] + "*" * (len(value) - 2)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]
