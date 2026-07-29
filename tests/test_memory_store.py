"""Tests for memory_store.py against a real (temp) ChromaDB collection.

The embedding model is replaced with a cheap deterministic stand-in so these run
without downloading weights; everything below the embedding call — batching,
metadata, the write lock, concurrent read-during-write — is the real thing.
"""

import hashlib
import threading
import time

import pytest

import components.memory_store as memory_store

DIM = 16


def _vector(text: str) -> list:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [b / 255.0 for b in digest[:DIM]]


class _FakeEmbeddings:
    def __init__(self):
        self.query_calls = 0
        self.document_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return _vector(text)

    def embed_documents(self, texts):
        self.document_calls += 1
        return [_vector(t) for t in texts]


@pytest.fixture
def store(tmp_path, monkeypatch):
    fake = _FakeEmbeddings()
    monkeypatch.setattr(memory_store, "_embeddings", fake)
    monkeypatch.setattr(memory_store, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(memory_store, "_client", None)
    monkeypatch.setattr(memory_store, "_collection", None)
    yield memory_store, fake
    memory_store._client = None
    memory_store._collection = None


class TestStoreAndRecall:
    def test_store_then_recall(self, store):
        ms, _ = store
        ms.store_memory("the project lives at ~/repos/langbot")
        assert ms.count() == 1
        assert ms.recall_memories("the project lives at ~/repos/langbot") == [
            "the project lives at ~/repos/langbot"
        ]

    def test_recall_on_empty_store(self, store):
        ms, _ = store
        assert ms.recall_memories("anything") == []

    def test_batch_uses_one_embedding_call(self, store):
        ms, fake = store
        facts = [f"fact number {i}" for i in range(10)]
        ids = ms.store_memories_batch(facts)
        assert len(ids) == 10
        assert fake.document_calls == 1
        assert ms.count() == 10

    def test_batch_empty_is_a_noop(self, store):
        ms, fake = store
        assert ms.store_memories_batch([]) == []
        assert fake.document_calls == 0

    def test_batch_preserves_supplied_timestamps(self, store):
        ms, _ = store
        ms.store_memories_batch(["old fact", "newer fact"],
                                timestamps=["2020-01-01T00:00:00Z", ""])
        metas = ms.get_collection().get(include=["metadatas"])["metadatas"]
        by_text = {m["text"]: m["timestamp"] for m in metas}
        assert by_text["old fact"] == "2020-01-01T00:00:00Z"
        assert by_text["newer fact"] != ""

    def test_persists_across_client_rebuild(self, store):
        ms, _ = store
        ms.store_memory("durable fact")
        ms._client = None
        ms._collection = None
        assert ms.count() == 1


class TestWorkerIntegration:
    def test_enqueued_job_becomes_recallable(self, store):
        ms, _ = store
        from components.memory_worker import DistillJob, MemoryWorker

        class _LLM:
            def invoke(self, prompt):
                return type("R", (), {"content": '["langbot uses ChromaDB for memory"]'})()

        worker = MemoryWorker(llm=_LLM())
        try:
            worker.enqueue(DistillJob(
                user_text="where are memories stored?",
                ai_text="in ChromaDB",
                tool_context="[read_any_file]: chromadb.PersistentClient(...)",
            ))
            deadline = time.time() + 10
            while time.time() < deadline and ms.count() == 0:
                time.sleep(0.05)
            assert ms.recall_memories("ChromaDB") == ["langbot uses ChromaDB for memory"]
        finally:
            worker.shutdown(timeout=2.0)


class TestConcurrency:
    def test_concurrent_writes_and_reads(self, store):
        ms, _ = store
        errors = []
        stop = threading.Event()

        def writer(n):
            try:
                for i in range(10):
                    ms.store_memories_batch([f"writer {n} fact {i}-{j}" for j in range(5)])
            except Exception as e:  # noqa: BLE001 - surfaced via `errors`
                errors.append(e)

        def reader():
            try:
                while not stop.is_set():
                    ms.recall_memories("writer", n=3)
                    time.sleep(0.005)
            except Exception as e:  # noqa: BLE001 - surfaced via `errors`
                errors.append(e)

        writers = [threading.Thread(target=writer, args=(n,)) for n in range(3)]
        readers = [threading.Thread(target=reader) for _ in range(3)]
        for t in writers + readers:
            t.start()
        for t in writers:
            t.join(timeout=60)
        stop.set()
        for t in readers:
            t.join(timeout=10)

        assert errors == []
        assert ms.count() == 150
