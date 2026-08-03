"""Unit tests for warmup.py — background initialisation of the slow components."""

import threading
import time

from components.warmup import FAILED, PENDING, READY, RUNNING, Warmup


def _flag_job(calls, name="job", delay=0.0):
    def job():
        calls.append(name)
        if delay:
            time.sleep(delay)
    return job


class TestStart:
    def test_jobs_run_and_report_ready(self):
        calls = []
        w = Warmup({"a": _flag_job(calls, "a"), "b": _flag_job(calls, "b")})
        assert w.start() is True
        assert w.wait(5) is True
        assert calls == ["a", "b"]
        assert w.status() == {"a": READY, "b": READY}
        assert w.is_ready()

    def test_nothing_runs_until_start(self):
        calls = []
        w = Warmup({"a": _flag_job(calls)})
        assert w.status() == {"a": PENDING}
        assert calls == []

    def test_concurrent_starts_run_the_jobs_once(self):
        # The whole point of the guard: a second caller must not kick off its own
        # copy of a multi-second model load.
        calls = []
        w = Warmup({"slow": _flag_job(calls, "slow", delay=0.3)})
        ready = threading.Barrier(8)

        def racer():
            ready.wait()
            w.start()

        threads = [threading.Thread(target=racer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert w.wait(5) is True
        assert calls == ["slow"]

    def test_start_is_a_no_op_once_everything_is_ready(self):
        calls = []
        w = Warmup({"a": _flag_job(calls, "a")})
        w.start()
        w.wait(5)
        assert w.start() is False
        assert calls == ["a"]

    def test_start_is_refused_while_a_run_is_in_flight(self):
        w = Warmup({"slow": _flag_job([], delay=0.3)})
        assert w.start() is True
        assert w.start() is False
        w.wait(5)


class TestFailure:
    def test_a_failing_job_does_not_strand_the_others(self):
        calls = []

        def boom():
            raise RuntimeError("no model")

        w = Warmup({"broken": boom, "fine": _flag_job(calls, "fine")})
        w.start()
        assert w.wait(5) is True
        assert w.status() == {"broken": FAILED, "fine": READY}
        assert calls == ["fine"]
        assert not w.is_ready()
        assert "no model" in str(w.errors()["broken"])

    def test_a_failure_is_retried_and_cleared_on_the_next_start(self):
        # A transient failure (a locked Chroma dir, say) must not disable the
        # component for the rest of the session.
        attempts = []

        def flaky():
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("locked")

        w = Warmup({"flaky": flaky})
        w.start()
        w.wait(5)
        assert w.status() == {"flaky": FAILED}

        assert w.start() is True
        w.wait(5)
        assert w.status() == {"flaky": READY}
        assert w.errors() == {}
        assert len(attempts) == 2

    def test_a_retry_does_not_re_run_what_already_succeeded(self):
        calls = []
        state = {"fail": True}

        def flaky():
            if state["fail"]:
                state["fail"] = False
                raise RuntimeError("once")

        w = Warmup({"done": _flag_job(calls, "done"), "flaky": flaky})
        w.start()
        w.wait(5)
        w.start()
        w.wait(5)
        assert calls == ["done"]
        assert w.is_ready()


class TestReporting:
    def test_wait_times_out_while_a_job_is_still_running(self):
        w = Warmup({"slow": _flag_job([], delay=0.5)})
        w.start()
        assert w.wait(0.05) is False
        assert w.status()["slow"] == RUNNING
        assert w.wait(5) is True

    def test_summary_names_each_component_and_its_state(self):
        def boom():
            raise RuntimeError("disk full")

        w = Warmup({"embeddings": _flag_job([]), "memory store": boom})
        w.start()
        w.wait(5)
        summary = w.summary()
        assert "embeddings ready" in summary
        assert "memory store failed (disk full)" in summary

    def test_summary_before_start(self):
        assert Warmup({"embeddings": _flag_job([])}).summary() == "embeddings pending"

    def test_summary_with_no_jobs(self):
        assert Warmup({}).summary() == "nothing to warm"

    def test_an_empty_warmup_is_ready_and_has_nothing_to_start(self):
        w = Warmup({})
        assert w.is_ready()
        assert w.start() is False
