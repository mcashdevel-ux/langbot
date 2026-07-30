"""Unit tests for memory_worker.py — background distillation queue.

The worker is exercised without ChromaDB or an LLM: ``store_fn`` is injected and
the LLM is a stub returning canned JSON, so these tests cover the queueing,
batching, backpressure, non-blocking and shutdown behaviour only.
"""

import json
import threading
import time

import pytest

from components.memory_worker import DistillJob, MemoryWorker, parse_facts


class TestParseFacts:
    """The distiller's output shape varies wildly across small local models, and a
    parse miss silently loses a whole turn's knowledge."""

    @pytest.mark.parametrize("raw, expected", [
        ('["a", "b"]', ["a", "b"]),                                    # as prompted
        ('```json\n["fenced"]\n```', ["fenced"]),
        ('[]', []),                                                    # nothing to store
        ('{"content": "[\\"wrapped\\"]", "tool_calls": []}', ["wrapped"]),
        ('{"content": ["listed"], "tool_calls": []}', ["listed"]),
        ('{"facts": ["keyed"]}', ["keyed"]),
        ('{"memories": ["keyed"]}', ["keyed"]),
        ('{"extracted_facts": ["odd key"]}', ["odd key"]),             # sole list value
        ('[{"fact": "as object"}]', ["as object"]),
        ('[{"text": "as object"}]', ["as object"]),
        ('Here you go:\n["in prose"]\nHope that helps.', ["in prose"]),
        ("- bullet one\n- bullet two", ["bullet one", "bullet two"]),
        ("1. numbered\n2) also numbered", ["numbered", "also numbered"]),
        ('["  padded  ", "", "kept"]', ["padded", "kept"]),
    ])
    def test_accepted_shapes(self, raw, expected):
        assert parse_facts(raw) == expected

    @pytest.mark.parametrize("raw", [
        "I'm afraid I can't do that",
        "",
        None,
        123,
        '{"status": "ok"}',                     # object with no list anywhere
        '{"a": [1], "b": [2]}',                 # ambiguous: two candidate lists
        '[["nested"], 3]',                      # items are not fact-like
    ])
    def test_unreadable_returns_none(self, raw):
        assert parse_facts(raw) is None


class _StubLLM:
    """Returns a fixed JSON array; optionally sleeps to simulate a slow call."""

    def __init__(self, facts=("a fact",), delay: float = 0.0, raw: str = None):
        self._raw = raw if raw is not None else json.dumps(list(facts))
        self._delay = delay
        self.calls = 0
        self.prompts = []

    def invoke(self, prompt):
        self.calls += 1
        self.prompts.append(prompt)
        if self._delay:
            time.sleep(self._delay)
        return type("Resp", (), {"content": self._raw})()


def _job(user="what tools?", ai="here they are", tool_context="[grep]: 30 matches"):
    return DistillJob(user_text=user, ai_text=ai, tool_context=tool_context)


class _Store:
    def __init__(self):
        self.batches = []
        self._lock = threading.Lock()

    def __call__(self, facts):
        with self._lock:
            self.batches.append(list(facts))
        return [str(i) for i in range(len(facts))]

    @property
    def facts(self):
        with self._lock:
            return [f for batch in self.batches for f in batch]


@pytest.fixture
def worker_factory():
    created = []

    def make(**kwargs):
        w = MemoryWorker(**kwargs)
        created.append(w)
        return w

    yield make
    for w in created:
        w.shutdown(timeout=2.0)


def _wait_until(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


class TestProcessing:
    def test_job_is_distilled_and_stored(self, worker_factory):
        store = _Store()
        llm = _StubLLM(facts=["langbot lives in ~/repos/langbot"])
        w = worker_factory(llm=llm, store_fn=store)
        assert w.enqueue(_job())
        assert _wait_until(lambda: store.facts)
        assert store.facts == ["langbot lives in ~/repos/langbot"]
        assert "30 matches" in llm.prompts[0]

    def test_no_store_call_when_no_facts(self, worker_factory):
        store = _Store()
        w = worker_factory(llm=_StubLLM(facts=[]), store_fn=store)
        w.enqueue(_job())
        assert _wait_until(lambda: w.qsize() == 0)
        time.sleep(0.1)
        assert store.batches == []

    def test_unparsable_output_is_skipped(self, worker_factory):
        store = _Store()
        w = worker_factory(llm=_StubLLM(raw="I'm afraid I can't do that"), store_fn=store)
        w.enqueue(_job())
        assert _wait_until(lambda: w.qsize() == 0)
        time.sleep(0.1)
        assert store.batches == []

    def test_fenced_json_is_parsed(self, worker_factory):
        store = _Store()
        w = worker_factory(llm=_StubLLM(raw='```json\n["fenced fact"]\n```'), store_fn=store)
        w.enqueue(_job())
        assert _wait_until(lambda: store.facts == ["fenced fact"])

    def test_envelope_wrapped_array_is_parsed(self, worker_factory):
        """Models that wrap answers in {"content": ...} wrap the fact array too."""
        store = _Store()
        raw = '{"content": "[\\"wrapped fact\\"]", "tool_calls": []}'
        w = worker_factory(llm=_StubLLM(raw=raw), store_fn=store)
        w.enqueue(_job())
        assert _wait_until(lambda: store.facts == ["wrapped fact"])

    def test_object_with_a_fact_list_is_parsed(self, worker_factory):
        store = _Store()
        w = worker_factory(llm=_StubLLM(raw='{"facts": ["object fact"]}'), store_fn=store)
        w.enqueue(_job())
        assert _wait_until(lambda: store.facts == ["object fact"])

    def test_store_failure_does_not_kill_the_thread(self, worker_factory):
        calls = {"n": 0}

        def flaky(facts):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("chroma down")
            return []

        w = worker_factory(llm=_StubLLM(), store_fn=flaky)
        w.enqueue(_job())
        assert _wait_until(lambda: calls["n"] == 1)
        w.enqueue(_job())
        assert _wait_until(lambda: calls["n"] == 2)

    def test_queued_jobs_share_one_store_call(self, worker_factory):
        store = _Store()
        # A slow first call lets the following jobs pile up, so they are drained
        # together and stored in a single batch.
        w = worker_factory(llm=_StubLLM(delay=0.2), store_fn=store)
        for _ in range(3):
            w.enqueue(_job())
        assert _wait_until(lambda: len(store.facts) == 3, timeout=10)
        assert len(store.batches) < 3


class TestBackpressure:
    def test_enqueue_never_blocks_the_caller(self, worker_factory):
        w = worker_factory(llm=_StubLLM(delay=1.0), store_fn=_Store(), max_queue_size=50)
        start = time.time()
        for _ in range(20):
            w.enqueue(_job())
        assert time.time() - start < 0.5

    def test_full_queue_drops_and_counts(self, worker_factory):
        w = worker_factory(llm=_StubLLM(delay=5.0), store_fn=_Store(), max_queue_size=2)
        accepted = [w.enqueue(_job()) for _ in range(10)]
        assert accepted[0] is True
        assert False in accepted
        assert w.dropped_count() == accepted.count(False)


class TestShutdown:
    def test_drains_queue(self, worker_factory):
        store = _Store()
        w = worker_factory(llm=_StubLLM(), store_fn=store)
        for _ in range(3):
            w.enqueue(_job())
        w.shutdown(timeout=10.0)
        assert w.qsize() == 0
        assert len(store.facts) == 3

    def test_short_timeout_warns_instead_of_raising(self, worker_factory, caplog):
        w = worker_factory(llm=_StubLLM(delay=0.5), store_fn=_Store(), max_queue_size=20)
        for _ in range(10):
            w.enqueue(_job())
        with caplog.at_level("WARNING"):
            w.shutdown(timeout=0.1)
        assert any("still queued" in r.message for r in caplog.records)
