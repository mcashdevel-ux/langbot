"""Tiered LLM with rate-limit accounting, used for background memory distillation.

Distillation (``components/memory_worker.py``) is the one LLM call in langbot that
is both off the critical path and cheap: the prompt is a truncated turn summary and
the answer is a short JSON array. That makes it a good fit for hosted free-tier
models, which are far better than a 9B local model at returning the strict JSON the
distiller parses, while their tight quotas are affordable for a few small calls per
turn.

The catch is that a hosted tier can vanish mid-session — quota exhausted, network
down, no API key configured — and distillation must never become a hard dependency
on one. So tiers are tried in order and the last one is always the local model the
agent itself runs on:

    llama-3.3-70b-versatile   best free instruction-follower; strictest quota
    openai/gpt-oss-120b       comparable quality, separate per-model quota
    qwen/qwen3.6-27b          smaller, and native to the prompt's /no_think hint
    llama-3.1-8b-instant      weakest, but 14.4K requests/day of headroom
    (local)                   always available, no network, no quota

Quotas are per model *and* per organization, so the chain genuinely multiplies the
available budget rather than re-hitting one bucket.

Rate limits are respected before a call rather than discovered by failing one
(https://console.groq.com/docs/rate-limits):

* Each tier carries the documented RPM/RPD/TPM/TPD caps for its model, and this
  module keeps sliding 60-second and 24-hour windows of what it has spent. A tier
  whose next call would not fit is skipped, so the fallback happens silently
  instead of costing a 429.
* Groq reports the truth in every response, and it is authoritative over local
  accounting (limits are org-wide, so other clients spend the same buckets):
  ``x-ratelimit-remaining-tokens`` / ``x-ratelimit-reset-tokens`` are per-minute
  tokens, ``x-ratelimit-remaining-requests`` / ``-reset-requests`` are per-*day*
  requests — a header naming trap the parsing here accounts for.
* A 429 still cools the tier down for exactly as long as its ``retry-after`` says.

A tier is also cooled down when it returns output the caller cannot use — see
``validate`` — since a fallback chain whose first link reliably answers in prose is
worse than no chain at all.
"""

import logging
import os
import re
import threading
import time
from collections import deque

from langchain_openai import ChatOpenAI

from .config import config
from .context_budget import estimate_tokens

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Ordered best-first, with the documented Groq free-tier limits per model. Every
# entry is overridable via the "distill.tiers" config key; an empty list there
# disables remote tiers and leaves the local model doing the work, as it did
# before this module existed.
DEFAULT_TIERS = [
    {"model": "llama-3.3-70b-versatile", "base_url": GROQ_BASE_URL,
     "api_key_env": "GROQ_API_KEY",
     "rpm": 30, "rpd": 1000, "tpm": 12000, "tpd": 100000},
    {"model": "openai/gpt-oss-120b", "base_url": GROQ_BASE_URL,
     "api_key_env": "GROQ_API_KEY",
     "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    {"model": "qwen/qwen3.6-27b", "base_url": GROQ_BASE_URL,
     "api_key_env": "GROQ_API_KEY",
     "rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    {"model": "llama-3.1-8b-instant", "base_url": GROQ_BASE_URL,
     "api_key_env": "GROQ_API_KEY",
     "rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 500000},
]

MINUTE = 60.0
DAY = 86400.0

DEFAULT_COOLDOWN = 300.0
# A server may ask for a very long wait (a daily quota reset); cap it so a tier is
# retried within a long session rather than being written off for hours.
MAX_COOLDOWN = 1800.0
DEFAULT_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.0
# Room left for the completion when checking a token budget: the distiller answers
# with a short JSON array, but the reservation has to cover a chatty model too.
DEFAULT_RESERVE_OUTPUT_TOKENS = 512

_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|s|m|h|d)")


def parse_duration(raw) -> "float | None":
    """Seconds from a Groq reset value such as ``7.66s``, ``2m59.56s`` or ``120``."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    matches = _DURATION_RE.findall(text)
    if not matches:
        try:
            return max(0.0, float(text))
        except ValueError:
            return None
    scale = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    return sum(float(value) * scale[unit] for value, unit in matches)


def _header_int(headers: dict, name: str) -> "int | None":
    raw = headers.get(name)
    if raw is None:
        return None
    try:
        return int(float(str(raw).strip()))
    except ValueError:
        return None


def response_headers(response) -> dict:
    """Rate-limit headers langchain attached to a response, or ``{}``.

    Requires ``include_response_headers=True`` on the client, which this module
    sets; without it a tier still works, purely on local accounting.
    """
    try:
        headers = response.response_metadata.get("headers")
    except AttributeError:
        return {}
    if not isinstance(headers, dict):
        return {}
    return {str(k).lower(): v for k, v in headers.items()}


def response_text(response) -> str:
    """The text of a chat response, for callers validating what a tier returned."""
    try:
        content = response.content
    except AttributeError:
        return str(response)
    return content if isinstance(content, str) else str(content)


def response_tokens(response) -> "int | None":
    """Total tokens the server reported spending on a response, if it said."""
    try:
        usage = response.usage_metadata
    except AttributeError:
        return None
    if not isinstance(usage, dict):
        return None
    total = usage.get("total_tokens")
    return total if isinstance(total, int) else None


def _retry_after_seconds(exc: Exception) -> "float | None":
    """The server's ``retry-after``, in seconds, if this looks like a 429.

    Reached through the OpenAI SDK's exception (``exc.response.headers``) without
    importing it, since langchain may wrap or replace the error type.
    """
    try:
        raw = exc.response.headers.get("retry-after")
    except AttributeError:
        return None
    return parse_duration(raw)


class Window:
    """Requests and tokens spent inside a rolling time window."""

    def __init__(self, span: float):
        self.span = span
        self._events: "deque[tuple[float, int]]" = deque()
        self._tokens = 0

    def prune(self, now: float) -> None:
        cutoff = now - self.span
        while self._events and self._events[0][0] <= cutoff:
            self._tokens -= self._events.popleft()[1]

    def record(self, tokens: int, now: float) -> None:
        self._events.append((now, tokens))
        self._tokens += tokens

    @property
    def requests(self) -> int:
        return len(self._events)

    @property
    def tokens(self) -> int:
        return self._tokens


class Tier:
    """One configured model: its client, its quota accounting, its cooldown."""

    def __init__(self, spec: dict, temperature: float, timeout: float):
        self.model = str(spec.get("model", "")).strip()
        self.base_url = str(spec.get("base_url", "")).strip()
        self.api_key_env = str(spec.get("api_key_env", "")).strip()
        self.temperature = float(spec.get("temperature", temperature))
        self.timeout = float(spec.get("timeout", timeout))
        # 0 means "not published / unlimited" and disables that check.
        self.rpm = int(spec.get("rpm", 0) or 0)
        self.rpd = int(spec.get("rpd", 0) or 0)
        self.tpm = int(spec.get("tpm", 0) or 0)
        self.tpd = int(spec.get("tpd", 0) or 0)
        self._client = None
        self._minute = Window(MINUTE)
        self._day = Window(DAY)
        self.blocked_until = 0.0
        self.blocked_reason = ""
        # Server-reported state; None until a response says otherwise.
        self.remaining_tokens: "int | None" = None
        self.remaining_requests: "int | None" = None
        self.tokens_reset_at = 0.0
        self.requests_reset_at = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.model and self.base_url)

    def api_key(self) -> str:
        """The key for this tier, or "" when it is not configured.

        Read per call rather than cached: ``components/vault.py`` populates the
        environment during startup, and a key can be added to the vault mid-session.
        """
        if not self.api_key_env:
            return "not-needed"
        return os.environ.get(self.api_key_env, "").strip()

    def block(self, seconds: float, reason: str, now: float) -> None:
        self.blocked_until = max(self.blocked_until, now + min(seconds, MAX_COOLDOWN))
        self.blocked_reason = reason

    def skip_reason(self, need_tokens: int, now: float) -> "str | None":
        """Why this tier cannot take a call of ``need_tokens`` now, else None."""
        if not self.api_key():
            return f"no ${self.api_key_env}"
        if now < self.blocked_until:
            return f"{self.blocked_reason or 'cooling'} {self.blocked_until - now:.0f}s"

        self._minute.prune(now)
        self._day.prune(now)
        if self.rpm and self._minute.requests >= self.rpm:
            return "rpm spent"
        if self.rpd and self._day.requests >= self.rpd:
            return "rpd spent"
        if self.tpm and self._minute.tokens + need_tokens > self.tpm:
            return "tpm spent"
        if self.tpd and self._day.tokens + need_tokens > self.tpd:
            return "tpd spent"

        # Server-reported remainders win while they are still current.
        if (self.remaining_tokens is not None and now < self.tokens_reset_at
                and self.remaining_tokens < need_tokens):
            return "tpm reported spent"
        if (self.remaining_requests is not None and now < self.requests_reset_at
                and self.remaining_requests < 1):
            return "rpd reported spent"
        return None

    def record(self, tokens: int, now: float) -> None:
        self._minute.record(tokens, now)
        self._day.record(tokens, now)
        # Keep the reported remainders consistent between responses, so several
        # calls inside one window cannot each see the same stale headroom.
        if self.remaining_tokens is not None:
            self.remaining_tokens = max(0, self.remaining_tokens - tokens)
        if self.remaining_requests is not None:
            self.remaining_requests = max(0, self.remaining_requests - 1)

    def sync(self, headers: dict, now: float) -> None:
        """Adopt Groq's own view of the quota from a response's headers."""
        if not headers:
            return
        tokens = _header_int(headers, "x-ratelimit-remaining-tokens")
        reset_tokens = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if tokens is not None:
            self.remaining_tokens = tokens
            # Without a reset value, treat the remainder as covering one window.
            self.tokens_reset_at = now + (MINUTE if reset_tokens is None
                                          else reset_tokens)
        requests = _header_int(headers, "x-ratelimit-remaining-requests")
        reset_requests = parse_duration(headers.get("x-ratelimit-reset-requests"))
        if requests is not None:
            self.remaining_requests = requests
            self.requests_reset_at = now + (DAY if reset_requests is None
                                            else reset_requests)

    def client(self, api_key: str):
        if self._client is None:
            self._client = ChatOpenAI(
                model=self.model,
                base_url=self.base_url,
                api_key=api_key,
                temperature=self.temperature,
                timeout=self.timeout,
                # Rate-limit headers are the whole point: they let the *next* call
                # pick a tier that still has budget.
                include_response_headers=True,
                # Retrying inside a tier delays the fallback for no gain: a 429
                # here means this model's quota is gone, and the next tier has its own.
                max_retries=0,
            )
        return self._client


class FallbackLLM:
    """Invoke the first tier with budget left, falling back to the local model.

    Exposes only ``invoke``, so it is a drop-in for the ``llm`` object that
    ``MemoryWorker`` is given, and returns the tier's own response unchanged.

    ``validate`` is called with the response's text and should return False when
    the caller cannot use it; the tier is then cooled down like a failed one.
    """

    def __init__(self, local_llm, tiers=None, validate=None,
                 cooldown: float = DEFAULT_COOLDOWN,
                 temperature: float = DEFAULT_TEMPERATURE,
                 timeout: float = DEFAULT_TIMEOUT,
                 reserve_output_tokens: int = DEFAULT_RESERVE_OUTPUT_TOKENS,
                 client_factory=None):
        self._local = local_llm
        self._validate = validate
        self._cooldown = cooldown
        self._reserve = reserve_output_tokens
        self._client_factory = client_factory
        self._lock = threading.Lock()
        self._tiers = []
        for spec in (DEFAULT_TIERS if tiers is None else tiers):
            if not isinstance(spec, dict):
                logger.warning("fallback_llm: ignoring non-object tier %r", spec)
                continue
            tier = Tier(spec, temperature, timeout)
            if not tier.valid:
                logger.warning("fallback_llm: ignoring tier without model/base_url: %r",
                               spec)
                continue
            self._tiers.append(tier)

    @property
    def tiers(self) -> "list[Tier]":
        return list(self._tiers)

    def describe(self) -> str:
        """One line for ``/health``: the chain, and what is usable right now."""
        now = time.time()
        parts = []
        with self._lock:
            for tier in self._tiers:
                reason = tier.skip_reason(self._reserve, now)
                parts.append(f"{tier.model} ({reason or 'ready'})")
        parts.append("local (ready)")
        return " -> ".join(parts)

    def _need_tokens(self, prompt) -> int:
        text = prompt if isinstance(prompt, str) else str(prompt)
        return estimate_tokens(text) + self._reserve

    def invoke(self, prompt):
        need = self._need_tokens(prompt)
        for tier in self._tiers:
            now = time.time()
            with self._lock:
                reason = tier.skip_reason(need, now)
            if reason is not None:
                logger.debug("fallback_llm: skipping %s (%s)", tier.model, reason)
                continue
            response = self._try(tier, prompt, need, now)
            if response is not None:
                return response
        return self._local.invoke(prompt)

    def _try(self, tier: Tier, prompt, need: int, now: float):
        """Call one tier. Returns None (after blocking it) on any failure."""
        try:
            factory = self._client_factory
            client = (factory(tier) if factory is not None
                      else tier.client(tier.api_key()))
            response = client.invoke(prompt)
        except Exception as exc:
            wait = _retry_after_seconds(exc)
            with self._lock:
                # Spend the estimate anyway: a request that 429s still counted
                # against the org's buckets on the way in.
                tier.record(need, now)
                tier.block(self._cooldown if wait is None else wait,
                           "rate limited" if wait is not None else "failed", now)
            logger.warning("fallback_llm: %s failed (%s: %s), trying next tier",
                           tier.model, type(exc).__name__, exc)
            return None

        spent = response_tokens(response)
        with self._lock:
            tier.record(need if spent is None else spent, now)
            tier.sync(response_headers(response), now)

        if self._validate is not None and not self._validate(response_text(response)):
            with self._lock:
                tier.block(self._cooldown, "unusable output", now)
            logger.warning("fallback_llm: %s returned unusable output, trying next tier",
                           tier.model)
            return None
        logger.debug("fallback_llm: served by %s", tier.model)
        return response


def build(local_llm, validate=None) -> FallbackLLM:
    """A ``FallbackLLM`` wired from the ``distill.*`` config keys."""
    return FallbackLLM(
        local_llm,
        tiers=config.get("distill.tiers", DEFAULT_TIERS),
        validate=validate,
        cooldown=config.get("distill.cooldown_seconds", DEFAULT_COOLDOWN),
        temperature=config.get("distill.temperature", DEFAULT_TEMPERATURE),
        timeout=config.get("distill.timeout", DEFAULT_TIMEOUT),
        reserve_output_tokens=config.get("distill.reserve_output_tokens",
                                         DEFAULT_RESERVE_OUTPUT_TOKENS),
    )
