import time
import threading

import pytest

import langbot


class DummyEmbeddings:
    def embed_query(self, text):
        return [0.0]


class DummyCollection:
    def __init__(self):
        self._data = []

    def add(self, ids, embeddings, metadatas):
        self._data.append((ids, embeddings, metadatas))

    def count(self):
        return len(self._data)

    def query(self, query_embeddings, n_results=1):
        return {"metadatas": [[m[2][0]["text"] for m in self._data]]}


def test_store_before_warmup_returns_message():
    # Ensure warmup not started
    langbot._warmup_started = False
    langbot.embeddings = None
    langbot.memory_collection = None

    res = langbot._store_memory("a fact")
    assert isinstance(res, str)
    assert "initializing" in res.lower()


def test_recall_before_warmup_empty_list():
    langbot._warmup_started = False
    langbot.embeddings = None
    langbot.memory_collection = None

    res = langbot._recall_memories("q")
    assert res == []


def test_store_and_recall_after_mock_ready():
    langbot.embeddings = DummyEmbeddings()
    langbot.memory_collection = DummyCollection()

    mem_id = langbot._store_memory("important fact")
    # mem_id should be a uuid-like string
    assert isinstance(mem_id, str)

    results = langbot._recall_memories("important")
    # Our dummy query returns whatever was stored
    assert isinstance(results, list)


def test_health_command_runs_without_error():
    # _handle_slash writes to UI; ensure it doesn't raise
    config = {"configurable": {"thread_id": "test"}}
    res = langbot._handle_slash("/health", config)
    assert res is False
