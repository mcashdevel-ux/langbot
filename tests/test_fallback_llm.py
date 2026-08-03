"""Unit tests for fallback_llm.py — tiered distillation LLM with rate limits.

No network and no real client: tiers are injected through ``client_factory`` and
time is passed explicitly where the code allows, so quota accounting, header
parsing, cooldowns and the local-model fallback are all covered offline.
"""

import time

import pytest

from components.fallback_llm import (
    DEFAULT_TIERS,
    FallbackLLM,
    Tier,
    parse_duration,
    response_headers,
    response_tokens,
    response_text,
)


class FakeResponse:
    """Stands in for an ``AIMessage`` with langchain's header/usage metadata."""

    def __init__(self, content="[]", headers=None, total_tokens=None):
        self.content = content
        self.response_metadata = {} if headers is None else {"headers": headers}
        self.usage_metadata = (None if total_tokens is None
                               else {"total_tokens": total_tokens})


class FakeClient:
    def __init__(self, responses=None, error=None):
        self._responses = list(responses or [])
        self._error = error
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        if self._error is not None:
            raise self._error
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse()


class Boom(Exception):
    """An error carrying a 429-style response, as the OpenAI SDK's does."""

    def __init__(self, retry_after=None):
        super().__init__("too many requests")
        headers = {} if retry_after is None else {"retry-after": retry_after}
        self.response = type("Resp", (), {"headers": headers})()


def make_llm(clients, **kwargs):
    """A FallbackLLM over one tier per client in ``clients``, keyed by model name."""
    tiers = [{"model": name, "base_url": "http://tier", **spec}
             for name, (spec, _) in clients.items()]
    by_model = {name: client for name, (_, client) in clients.items()}
    local = FakeClient([FakeResponse("local")])
    llm = FallbackLLM(local, tiers=tiers,
                      client_factory=lambda tier: by_model[tier.model], **kwargs)
    return llm, by_model, local


class TestParseDuration:
    @pytest.mark.parametrize("raw, expected", [
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),
        ("1h2m3s", 3723.0),
        ("100ms", 0.1),
        ("120", 120.0),
        (2, 2.0),
        ("", None),
        (None, None),
        ("soon", None),
    ])
    def test_values(self, raw, expected):
        assert parse_duration(raw) == expected


class TestResponseReaders:
    def test_reads_text_headers_and_usage(self):
        r = FakeResponse("hi", {"X-RateLimit-Remaining-Tokens": "5"}, total_tokens=7)
        assert response_text(r) == "hi"
        # Header names are lowercased, since casing is not guaranteed on the wire.
        assert response_headers(r) == {"x-ratelimit-remaining-tokens": "5"}
        assert response_tokens(r) == 7

    def test_missing_metadata_is_not_an_error(self):
        assert response_headers(object()) == {}
        assert response_tokens(object()) is None
        assert response_text(object()).startswith("<object")

    def test_non_string_content_is_stringified(self):
        assert response_text(FakeResponse(content=[{"text": "a"}])) == "[{'text': 'a'}]"


class TestTierBudget:
    def test_missing_api_key_disables_tier(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        tier = Tier({"model": "m", "base_url": "u", "api_key_env": "GROQ_API_KEY"},
                    0.0, 30.0)
        assert tier.skip_reason(10, 1000.0) == "no $GROQ_API_KEY"
        monkeypatch.setenv("GROQ_API_KEY", "k")
        assert tier.skip_reason(10, 1000.0) is None

    def test_local_windows_block_then_expire(self):
        tier = Tier({"model": "m", "base_url": "u", "rpm": 2, "tpm": 1000}, 0.0, 30.0)
        tier.record(100, 1000.0)
        tier.record(100, 1001.0)
        assert tier.skip_reason(10, 1002.0) == "rpm spent"
        # The 60s window has rolled past both calls.
        assert tier.skip_reason(10, 1062.0) is None

    def test_request_larger_than_tpm_is_never_attempted(self):
        tier = Tier({"model": "m", "base_url": "u", "tpm": 8000}, 0.0, 30.0)
        assert tier.skip_reason(9000, 1000.0) == "tpm spent"

    def test_daily_caps_are_separate_from_per_minute(self):
        tier = Tier({"model": "m", "base_url": "u", "rpd": 1, "tpd": 500}, 0.0, 30.0)
        tier.record(100, 1000.0)
        # Well past the minute window, still inside the day.
        assert tier.skip_reason(10, 1000.0 + 600) == "rpd spent"

    def test_headers_override_local_accounting(self):
        tier = Tier({"model": "m", "base_url": "u"}, 0.0, 30.0)
        tier.sync({"x-ratelimit-remaining-tokens": "300",
                   "x-ratelimit-reset-tokens": "30s",
                   "x-ratelimit-remaining-requests": "0",
                   "x-ratelimit-reset-requests": "2m"}, 1000.0)
        # Requests are exhausted for the day even though nothing was spent locally.
        assert tier.skip_reason(10, 1001.0) == "rpd reported spent"
        # Once the reported windows lapse, local accounting is back in charge.
        assert tier.skip_reason(10, 1000.0 + 121) is None

    def test_reported_token_headroom_is_spent_between_responses(self):
        tier = Tier({"model": "m", "base_url": "u"}, 0.0, 30.0)
        tier.sync({"x-ratelimit-remaining-tokens": "500",
                   "x-ratelimit-reset-tokens": "60s"}, 1000.0)
        tier.record(400, 1001.0)
        assert tier.skip_reason(200, 1002.0) == "tpm reported spent"


class TestFallbackChain:
    def test_first_tier_wins(self):
        first = FakeClient([FakeResponse("first")])
        second = FakeClient([FakeResponse("second")])
        llm, _, local = make_llm({"a": ({}, first), "b": ({}, second)})
        assert llm.invoke("p").content == "first"
        assert (second.calls, local.calls) == (0, 0)

    def test_failure_falls_through_to_the_next_tier(self):
        broken = FakeClient(error=Boom())
        good = FakeClient([FakeResponse("second")])
        llm, _, local = make_llm({"a": ({}, broken), "b": ({}, good)})
        assert llm.invoke("p").content == "second"
        assert local.calls == 0

    def test_all_tiers_exhausted_falls_back_to_local(self):
        llm, _, local = make_llm({"a": ({}, FakeClient(error=Boom())),
                                  "b": ({}, FakeClient(error=Boom()))})
        assert llm.invoke("p").content == "local"
        assert local.calls == 1

    def test_failed_tier_is_not_retried_while_cooling(self):
        broken = FakeClient(error=Boom())
        llm, _, local = make_llm({"a": ({}, broken)}, cooldown=600.0)
        llm.invoke("p")
        llm.invoke("p")
        assert broken.calls == 1
        assert local.calls == 2

    def test_retry_after_sets_the_cooldown(self):
        broken = FakeClient(error=Boom(retry_after="2"))
        llm, _, _ = make_llm({"a": ({}, broken)}, cooldown=600.0)
        llm.invoke("p")
        # Two seconds, as asked — not the much longer default cooldown.
        assert llm.tiers[0].blocked_until == pytest.approx(time.time() + 2.0, abs=1.0)

    def test_unusable_output_moves_to_the_next_tier(self):
        prose = FakeClient([FakeResponse("I cannot help with that")])
        good = FakeClient([FakeResponse('[{"fact": "x"}]')])
        llm, _, _ = make_llm({"a": ({}, prose), "b": ({}, good)},
                             validate=lambda text: text.strip().startswith("["))
        assert llm.invoke("p").content == '[{"fact": "x"}]'
        assert llm.tiers[0].blocked_reason == "unusable output"

    def test_tier_is_skipped_once_its_reported_quota_is_gone(self):
        first = FakeClient([FakeResponse("first", {
            "x-ratelimit-remaining-tokens": "0",
            "x-ratelimit-reset-tokens": "60s",
        })])
        second = FakeClient([FakeResponse("second")])
        llm, _, _ = make_llm({"a": ({}, first), "b": ({}, second)})
        assert llm.invoke("p").content == "first"
        assert llm.invoke("p").content == "second"
        assert first.calls == 1

    def test_reported_usage_is_charged_to_the_tier(self):
        client = FakeClient([FakeResponse("ok", total_tokens=900)])
        llm, _, local = make_llm({"a": ({"tpm": 1000}, client)},
                                 reserve_output_tokens=10)
        llm.invoke("p")
        # 900 spent of 1000: the next call cannot fit and goes local.
        assert llm.invoke("p" * 400).content == "local"

    def test_malformed_tiers_are_ignored(self):
        llm = FallbackLLM(FakeClient(), tiers=[{"model": "m"}, "nonsense",
                                               {"model": "m2", "base_url": "u"}])
        assert [t.model for t in llm.tiers] == ["m2"]

    def test_no_tiers_means_local_only(self):
        local = FakeClient([FakeResponse("local")])
        llm = FallbackLLM(local, tiers=[])
        assert llm.invoke("p").content == "local"

    def test_describe_lists_the_chain_and_ends_local(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        llm = FallbackLLM(FakeClient())
        described = llm.describe()
        assert described.endswith("local (ready)")
        assert "no $GROQ_API_KEY" in described
        for spec in DEFAULT_TIERS:
            assert spec["model"] in described


class TestDefaultTiers:
    def test_ordered_by_quality_and_end_with_the_deepest_daily_quota(self):
        models = [t["model"] for t in DEFAULT_TIERS]
        assert models[0] == "llama-3.3-70b-versatile"
        assert models[-1] == "llama-3.1-8b-instant"

    def test_every_tier_publishes_limits_and_a_key_source(self):
        for spec in DEFAULT_TIERS:
            assert spec["api_key_env"] == "GROQ_API_KEY"
            assert spec["base_url"].startswith("https://")
            for key in ("rpm", "rpd", "tpm", "tpd"):
                assert spec[key] > 0
