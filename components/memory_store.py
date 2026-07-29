"""Long-term semantic memory store (embeddings + ChromaDB).

Canonical, importable home for the memory state that used to live as globals in
``langbot.py``. Both the main thread (``remember``/``recall``, ``/save``) and the
background memory worker read and write the same collection through here, so
writes are serialized behind one lock rather than relying on assumptions about
ChromaDB's internal concurrency guarantees.

Reads deliberately stay outside the lock: the worst case is a ``recall`` issued
mid-batch-write missing the newest facts, which is the accepted eventual
consistency of asynchronous distillation.

The embedding model and the Chroma collection are built on first use (the agent
warms them at startup via ``get_embeddings()``), so importing this module is
cheap and its state is injectable in tests.
"""

import threading
import uuid
from datetime import datetime, timezone

import chromadb
from chromadb.config import Settings

from . import console as ui
from .config import config
from .utils import suppress_native_output

CHROMA_PERSIST_DIR = config.get("paths.chroma_dir", "./memory/agent_memory_chroma",
                                env="AGENT_CHROMA_DIR")
COLLECTION_NAME = config.get("memory.collection_name", "agent_longterm_memory")
EMBEDDING_MODEL = config.get("memory.embedding_model",
                             "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DEVICE = config.get("memory.embedding_device", "cpu")

_write_lock = threading.Lock()
_init_lock = threading.Lock()
_embeddings = None
_client = None
_collection = None


def get_embeddings():
    """Return the embedding model, loading it (quietly) on first use.

    The heavy transformers/tqdm progress output goes to the raw stderr fd, so it
    is muted at the fd level while the weights load.
    """
    global _embeddings
    with _init_lock:
        if _embeddings is not None:
            return _embeddings
        ui.info("Loading embedding model...")
        try:
            import transformers  # noqa: WPS433 (optional, only to quiet it)

            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
        except Exception:
            pass
        from langchain_huggingface import HuggingFaceEmbeddings

        with suppress_native_output():
            _embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': EMBEDDING_DEVICE},
                encode_kwargs={'normalize_embeddings': True},
            )
        ui.success("Embedding model ready.")
        return _embeddings


def get_collection():
    """Return the persistent Chroma collection, creating it on first use."""
    global _client, _collection
    with _init_lock:
        if _collection is None:
            _client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return _collection


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def store_memory(text: str) -> str:
    mem_id = str(uuid.uuid4())
    vector = get_embeddings().embed_query(text)
    with _write_lock:
        get_collection().add(
            ids=[mem_id],
            embeddings=[vector],
            metadatas=[{"text": text, "timestamp": _now_iso()}],
        )
    return mem_id


def store_memories_batch(texts: list[str], timestamps: "list[str] | None" = None) -> list[str]:
    """Store many facts with a single ``embed_documents`` call.

    ``timestamps`` preserves original times for facts pulled from elsewhere
    (e.g. Supabase); missing or blank entries fall back to now.
    """
    if not texts:
        return []
    ids = [str(uuid.uuid4()) for _ in texts]
    vectors = get_embeddings().embed_documents(texts)
    now = _now_iso()
    stamps = list(timestamps or [])
    stamps += [now] * (len(texts) - len(stamps))
    with _write_lock:
        get_collection().add(
            ids=ids,
            embeddings=vectors,
            metadatas=[
                {"text": t, "timestamp": ts or now} for t, ts in zip(texts, stamps)
            ],
        )
    return ids


def recall_memories(query: str, n: int = 3) -> list[str]:
    collection = get_collection()
    total = collection.count()
    if total == 0:
        # Chroma rejects n_results < 1, so guard the empty-store case explicitly.
        return []
    query_vec = get_embeddings().embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(n, total),
    )
    if not results or not results["metadatas"] or not results["metadatas"][0]:
        return []
    return [meta.get("text", "") for meta in results["metadatas"][0] if meta.get("text")]


def count() -> int:
    return get_collection().count()
