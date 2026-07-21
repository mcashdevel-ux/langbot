"""
Feature: Supabase Sync
======================
Sync local ChromaDB long-term memory ↔ Supabase cloud knowledge table.

Uses vault credentials (SUPABASE_URL + SUPABASE_SERVICE_KEY).
Falls back to env vars if vault not available.
Two-way sync: push local facts up, pull remote facts down.
Deduplicates by fact text (exact match).

Credentials are read from (in priority order):
  1. os.environ  (including anything auto-loaded from vault at startup)
  2. langbot's _vault_store (via sys.modules lookup)

Tools exposed to agent:
  supabase_sync(action="push")              — push local memory → Supabase
  supabase_sync(action="pull")              — pull Supabase → local memory
  supabase_sync(action="status")            — show config & local fact count
  supabase_sync(action="push_secrets")      — encrypt vault creds → Supabase
  supabase_sync(action="pull_secrets")      — decrypt & restore vault creds
  supabase_sync(action="list_remote_secrets") — list secret names (no decrypt)
"""

import json
import os
import sys
import time
import hashlib
import base64
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests
try:
    from cryptography.fernet import Fernet
    _HAS_FERNET = True
except ImportError:
    _HAS_FERNET = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VAULT_SECRET_MARKER = "🎯"   # prefix for secret entries in the knowledge table

# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------

def _get_vault_store():
    """Return langbot's live VaultStore instance, or None."""
    try:
        main = sys.modules.get("__main__") or sys.modules.get("langbot")
        if main:
            store = getattr(main, "_vault_store", None)
            if store and not store.is_locked():
                return store
    except Exception:
        pass
    return None


def _get_memory_dir() -> Path:
    """Return langbot's MEMORY_DIR at call time."""
    try:
        main = sys.modules.get("__main__") or sys.modules.get("langbot")
        if main:
            d = getattr(main, "MEMORY_DIR", None)
            if d:
                return Path(d)
    except Exception:
        pass
    return Path("memory")


# ---------------------------------------------------------------------------
# Supabase Sync Engine
# ---------------------------------------------------------------------------

class SupabaseSync:
    """Sync local knowledge ↔ Supabase cloud database."""

    def __init__(self):
        self.url  = os.environ.get("SUPABASE_URL", "").strip()
        self.key  = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
        # Fall back to vault if env vars aren't set
        if not (self.url and self.key):
            store = _get_vault_store()
            if store:
                self.url = store.get("SUPABASE_URL") or self.url
                self.key = store.get("SUPABASE_SERVICE_KEY") or self.key
        self.enabled = bool(self.url and self.key)

    def _headers(self) -> dict:
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates",
        }

    # ── Helpers ──

    def _knowledge_file(self) -> Path:
        """Path to the flat-file knowledge store (used for push/pull)."""
        return _get_memory_dir() / "knowledge.md"

    def _local_facts(self) -> list[str]:
        """Return active fact strings from the local ChromaDB memory store (and knowledge.md if present)."""
        facts = []
        
        # 1. Try reading from ChromaDB (primary source)
        try:
            main = sys.modules.get("__main__") or sys.modules.get("langbot")
            collection = None
            if main:
                collection = getattr(main, "memory_collection", None)
            
            if collection is None:
                import chromadb
                from chromadb.config import Settings
                persist_dir = str(_get_memory_dir() / "agent_memory_chroma")
                if os.path.exists(persist_dir):
                    client = chromadb.PersistentClient(
                        path=persist_dir,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    collection = client.get_or_create_collection("agent_longterm_memory")
            
            if collection and collection.count() > 0:
                data = collection.get(include=["metadatas"])
                if data and data.get("metadatas"):
                    for meta in data["metadatas"]:
                        if meta and "text" in meta:
                            fact_text = meta["text"].strip()
                            # Exclude secrets and duplicates
                            if fact_text and not fact_text.startswith(VAULT_SECRET_MARKER):
                                facts.append(fact_text)
        except Exception as e:
            logger.warning("supabase_sync: error reading local ChromaDB facts: %s", e)

        # 2. Backward compatibility: also read from knowledge.md if it exists, to capture any pre-existing facts
        kf = self._knowledge_file()
        if kf.exists():
            try:
                for line in kf.read_text(encoding="utf-8").splitlines():
                    stripped = line.strip()
                    if not stripped or "[pruned]" in stripped:
                        continue
                    if "]: " in stripped:
                        content = stripped.split("]: ", 1)[1].strip()
                        if content and content not in facts and not content.startswith(VAULT_SECRET_MARKER):
                            facts.append(content)
            except Exception as e:
                logger.warning("supabase_sync: error reading legacy knowledge file: %s", e)
                
        return facts

    def _store_facts_locally_batch(self, facts: list[str], timestamps: list[str]) -> bool:
        """Store multiple facts in the local ChromaDB memory store in batch."""
        if not facts:
            return True
        try:
            import uuid
            from datetime import datetime, timezone
            main = sys.modules.get("__main__") or sys.modules.get("langbot")
            collection = None
            embeddings = None
            if main:
                collection = getattr(main, "memory_collection", None)
                embeddings = getattr(main, "embeddings", None)
            
            if collection is None:
                import chromadb
                from chromadb.config import Settings
                persist_dir = str(_get_memory_dir() / "agent_memory_chroma")
                client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
                collection = client.get_or_create_collection("agent_longterm_memory")
            
            if embeddings is None:
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
            
            # Batch process in chunks of 100
            chunk_size = 100
            for i in range(0, len(facts), chunk_size):
                chunk_facts = facts[i:i+chunk_size]
                chunk_ts = timestamps[i:i+chunk_size]
                
                # Embed batch
                vectors = embeddings.embed_documents(chunk_facts)
                
                ids = [str(uuid.uuid4()) for _ in chunk_facts]
                metadatas = []
                for fact, ts in zip(chunk_facts, chunk_ts):
                    if not ts:
                        ts = datetime.now(timezone.utc).isoformat()
                    metadatas.append({
                        "text": fact,
                        "timestamp": ts,
                    })
                
                collection.add(
                    ids=ids,
                    embeddings=vectors,
                    metadatas=metadatas,
                )
            return True
        except Exception as e:
            logger.warning("supabase_sync: error storing pulled facts batch to ChromaDB: %s", e)
            return False

    def _append_facts(self, new_entries: list[str]) -> bool:
        """Append new timestamped fact lines to the local knowledge file."""
        kf = self._knowledge_file()
        try:
            kf.parent.mkdir(parents=True, exist_ok=True)
            with open(kf, "a", encoding="utf-8") as f:
                f.write("\n" + "\n".join(new_entries))
            return True
        except Exception as e:
            logger.warning("supabase_sync: error writing knowledge file: %s", e)
            return False

    # ── Push: local → Supabase ──

    def push_knowledge(self) -> str:
        if not self.enabled:
            return "Supabase not configured (set SUPABASE_URL + SUPABASE_SERVICE_KEY in vault or env)."

        entries = self._local_facts()
        if not entries:
            return "No local knowledge entries to push."

        # Fetch existing remote facts to skip duplicates
        try:
            r = requests.get(
                f"{self.url}/rest/v1/knowledge",
                headers=self._headers(),
                params={"select": "fact", "stale": "eq.false"},
                timeout=10,
            )
            existing: set[str] = set()
            if r.status_code == 200:
                for row in r.json() or []:
                    if row.get("fact"):
                        existing.add(row["fact"].strip())
        except Exception:
            existing = set()

        pushed = errors = 0
        for fact in entries:
            if fact in existing:
                continue
            payload = {"fact": fact, "tags": [], "access_count": 0, "stale": False}
            try:
                r = requests.post(
                    f"{self.url}/rest/v1/knowledge",
                    headers=self._headers(),
                    json=payload,
                    timeout=10,
                )
                if r.status_code in (200, 201):
                    pushed += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                logger.warning("supabase_sync push error: %s", e)

        return f"Pushed {pushed} new facts to Supabase ({errors} errors)."

    # ── Pull: Supabase → local ──

    def pull_knowledge(self) -> str:
        if not self.enabled:
            return "Supabase not configured (set SUPABASE_URL + SUPABASE_SERVICE_KEY in vault or env)."

        try:
            r = requests.get(
                f"{self.url}/rest/v1/knowledge",
                headers=self._headers(),
                params={
                    "select": "fact,created_at",
                    "stale": "eq.false",
                    "order": "created_at.desc",
                },
                timeout=15,
            )
            if r.status_code != 200:
                return f"Supabase error: HTTP {r.status_code}"

            remote_facts = r.json() or []
            if not remote_facts:
                return "No remote facts found in Supabase."

            local_set = set(self._local_facts())
            pulled = skipped = 0
            new_lines: list[str] = []
            facts_to_store = []
            timestamps_to_store = []

            for entry in remote_facts:
                fact = entry.get("fact", "").strip()
                if not fact or fact in local_set:
                    skipped += 1
                    continue
                
                facts_to_store.append(fact)
                timestamps_to_store.append(entry.get("created_at"))
                
                ts = entry.get("created_at", "")
                try:
                    ts = ts[:19].replace("T", " ") if ts else time.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                new_lines.append(f"- [{ts}]: {fact}")
                local_set.add(fact)
                pulled += 1

            if facts_to_store:
                self._store_facts_locally_batch(facts_to_store, timestamps_to_store)

            if new_lines:
                self._append_facts(new_lines)

            return f"Pulled {pulled} new facts from Supabase ({skipped} already local)."

        except requests.exceptions.RequestException as e:
            return f"Supabase pull failed: {e}"
        except Exception as e:
            return f"Sync pull failed: {e}"

    # ── Vault Secrets Sync ──

    @staticmethod
    def _get_fernet():
        if not _HAS_FERNET:
            return None
        master_key = os.environ.get("SECRETS_MASTER_KEY", "").strip()
        if not master_key:
            return None
        try:
            if len(master_key) != 44 or not master_key.endswith("="):
                key = base64.urlsafe_b64encode(hashlib.sha256(master_key.encode()).digest())
            else:
                key = master_key.encode()
            return Fernet(key)
        except Exception:
            return None

    def push_secrets(self) -> str:
        if not self.enabled:
            return "Supabase not configured."
        if not _HAS_FERNET:
            return "cryptography package not installed. Run: pip install cryptography"
        fernet = self._get_fernet()
        if not fernet:
            return (
                "SECRETS_MASTER_KEY not set. Generate one with:\n"
                "  python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
                "Then store it: vault(action='store', name='SECRETS_MASTER_KEY', value='...')"
            )

        store = _get_vault_store()
        creds: Dict[str, str] = store.get_all_plaintext() if store else {}
        if not creds:
            return "No vault credentials found to sync."

        # Fetch existing remote secret names to skip duplicates
        try:
            r = requests.get(
                f"{self.url}/rest/v1/knowledge",
                headers=self._headers(),
                params={"select": "fact", "stale": "eq.false",
                        "fact": f"like.{VAULT_SECRET_MARKER}%"},
                timeout=10,
            )
            existing: set[str] = set()
            if r.status_code == 200:
                for row in r.json() or []:
                    fact = row.get("fact", "")
                    if "||" in fact:
                        existing.add(fact.split("||", 1)[0].lstrip(VAULT_SECRET_MARKER))
        except Exception:
            existing = set()

        pushed = errors = 0
        for name, value in creds.items():
            if name in existing:
                continue
            try:
                encrypted = fernet.encrypt(str(value).encode()).decode()
                fact = f"{VAULT_SECRET_MARKER}{name}||{encrypted}"
                r = requests.post(
                    f"{self.url}/rest/v1/knowledge",
                    headers=self._headers(),
                    json={"fact": fact, "tags": ["secret"], "access_count": 0, "stale": False},
                    timeout=10,
                )
                if r.status_code in (200, 201):
                    pushed += 1
                else:
                    errors += 1
                    logger.warning("Failed to push secret '%s': HTTP %s", name, r.status_code)
            except Exception as e:
                errors += 1
                logger.warning("Error pushing secret '%s': %s", name, e)

        return f"Pushed {pushed} encrypted secrets to Supabase ({errors} errors)."

    def pull_secrets(self) -> str:
        if not self.enabled:
            return "Supabase not configured."
        if not _HAS_FERNET:
            return "cryptography package not installed. Run: pip install cryptography"
        fernet = self._get_fernet()
        if not fernet:
            return "SECRETS_MASTER_KEY not set."

        try:
            r = requests.get(
                f"{self.url}/rest/v1/knowledge",
                headers=self._headers(),
                params={"select": "fact", "stale": "eq.false",
                        "fact": f"like.{VAULT_SECRET_MARKER}%"},
                timeout=15,
            )
            if r.status_code != 200:
                return f"Supabase error: HTTP {r.status_code}"

            secret_entries = [e for e in (r.json() or []) if "||" in e.get("fact", "")]
            if not secret_entries:
                return "No encrypted secrets found in Supabase."

            store = _get_vault_store()
            local_creds: set[str] = set(store.get_all_plaintext().keys()) if store else set()

            pulled = errors = 0
            for entry in secret_entries:
                fact = entry.get("fact", "")
                marker_name, encrypted_value = fact.split("||", 1)
                secret_name = marker_name.lstrip(VAULT_SECRET_MARKER)
                if secret_name in local_creds:
                    continue
                try:
                    decrypted = fernet.decrypt(encrypted_value.encode()).decode()
                    if store:
                        store.put(secret_name, decrypted)
                        os.environ[secret_name] = decrypted
                    pulled += 1
                except Exception as e:
                    errors += 1
                    logger.warning("Error decrypting secret '%s': %s", secret_name, e)

            return f"Pulled {pulled} secrets from Supabase ({errors} decryption errors)."

        except requests.exceptions.RequestException as e:
            return f"Supabase pull_secrets failed: {e}"
        except Exception as e:
            return f"Sync pull_secrets failed: {e}"

    def list_remote_secrets(self) -> str:
        if not self.enabled:
            return "Supabase not configured."
        try:
            r = requests.get(
                f"{self.url}/rest/v1/knowledge",
                headers=self._headers(),
                params={"select": "fact", "stale": "eq.false",
                        "fact": f"like.{VAULT_SECRET_MARKER}%"},
                timeout=10,
            )
            if r.status_code != 200:
                return f"Supabase error: HTTP {r.status_code}"
            names = [
                e["fact"].split("||", 1)[0].lstrip(VAULT_SECRET_MARKER)
                for e in (r.json() or [])
                if "||" in e.get("fact", "")
            ]
            if not names:
                return "No encrypted secrets in Supabase."
            lines = ["☁️ Remote Secrets (Supabase)", "=" * 50]
            for name in sorted(names):
                lines.append(f"  🔒 {name}")
            return "\n".join(lines)
        except requests.exceptions.RequestException as e:
            return f"Supabase list_remote_secrets failed: {e}"

    # ── Status ──

    def status(self) -> str:
        lines = ["☁️  SUPABASE SYNC STATUS", "=" * 50]
        if self.enabled:
            lines.append("  Status    : ✅ Configured")
            lines.append(f"  URL       : {self.url}")
            lines.append(f"  Key       : {self.key[:8]}...{self.key[-4:]}")
        else:
            lines.append("  Status    : ❌ Not configured")
            lines.append("  Set SUPABASE_URL and SUPABASE_SERVICE_KEY in vault or env vars.")

        # ChromaDB memory count
        try:
            main = sys.modules.get("__main__") or sys.modules.get("langbot")
            collection = None
            if main:
                collection = getattr(main, "memory_collection", None)
            if collection is None:
                import chromadb
                from chromadb.config import Settings
                persist_dir = str(_get_memory_dir() / "agent_memory_chroma")
                if os.path.exists(persist_dir):
                    client = chromadb.PersistentClient(
                        path=persist_dir,
                        settings=Settings(anonymized_telemetry=False),
                    )
                    collection = client.get_or_create_collection("agent_longterm_memory")
            if collection:
                lines.append(f"  Local ChromaDB facts: {collection.count()}")
        except Exception:
            pass

        kf = self._knowledge_file()
        if kf.exists():
            try:
                count = sum(
                    1 for line in kf.read_text(encoding="utf-8").splitlines()
                    if "]: " in line and "[pruned]" not in line
                )
                lines.append(f"  Local knowledge.md facts: {count}")
            except Exception:
                pass

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool dispatcher (called from langbot's @tool wrapper)
# ---------------------------------------------------------------------------

def _sync_handler(action: str) -> str:
    syncer = SupabaseSync()
    dispatch = {
        "push":                syncer.push_knowledge,
        "pull":                syncer.pull_knowledge,
        "push_secrets":        syncer.push_secrets,
        "pull_secrets":        syncer.pull_secrets,
        "list_remote_secrets": syncer.list_remote_secrets,
        "status":              syncer.status,
    }
    fn = dispatch.get(action)
    if fn is None:
        return (
            f"Unknown action: '{action}'. "
            "Valid: push, pull, status, push_secrets, pull_secrets, list_remote_secrets."
        )
    return fn()
