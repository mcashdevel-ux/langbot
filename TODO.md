# TODO

Working list, ordered by what to do first. `CODE_REVIEW.md` holds the findings and their
status; this file holds the *work*, including items that were never review findings.

Grouped by what a task needs, because that is what actually blocks it: a live model, a
decision, or neither.

## Needs neither — config and logic only

- [ ] **Nits: L4, L5, L7 (`engines.py`), L8, L9 (`vault.py`).** Implicit `git clone` side
      effect and stale docstrings from another project; a fresh `requests.Session()` per
      request; a `MAX_CREDENTIALS` length check read outside the lock; two masking helpers
      where one is unused. Batchable into one cleanup PR.
- [ ] **Task-based eval harness.** 10–15 scripted REPL tasks run after each merge. Unit tests
      confirm each part does what it is told in isolation; only this shows whether compaction
      and tool routing are net-positive *together*. Biggest lift here, and the thing that
      makes every claim below checkable.

## Needs a decision

- [ ] **Vault default (C3).** Pick one: default `LANGBOT_VAULT_PASSWORD` on, prompting at
      first run, or make the startup banner warn loudly while it is unset. Either closes C3;
      leaving it is what keeps the master key recoverable beside the ciphertext.

## Needs a real session against the local model

- [ ] **Confirm `--jinja` took.** Grammar-constrained tool calls are a `llama-server` flag,
      outside this repo. The signal is `/health`'s `tool-call repairs` staying near
      `0 recovered` over a real session; if it does, start deleting the parts of
      `tool_call_repair.py` that are then dead weight instead of keeping them as insurance.
- [ ] **Shrink `context.reserve_tokens` (8192).** Measured offline: tool schemas cost ~950
      tokens for a typical per-turn binding (~2,170 if all 20 were bound) and the system
      prompt 191, so overhead runs ~1.2K. Read `/health`'s `context` line over a real session
      (the rolling summary can add up to `summary_max_chars`), then hand the difference back
      to conversation.
- [ ] **Thinking-mode toggle for Qwen3.** `<think>` output is stripped for display but nothing
      controls whether the model reasons before answering, and `<thought>` already has a tag
      in the system prompt. Measure whether `/no_think` (or the llama.cpp reasoning-format
      flag) costs tool-selection accuracy; if not, it is free context back.
- [ ] **Embedding-based tool routing.** `tool_router`'s triggers are exact-match regexes, so
      "check what's stored for auth" misses the `vault` binding that "vault credential" hits.
      The MiniLM model is already loaded for memory. Do this when a missed binding is
      actually observed, not before.

## Deferred on purpose

- **Blast-radius gate** — a justification field required before the shell tool runs
  `rm -rf`, `push --force`, `DROP`/`TRUNCATE`, or a vault write. Deliberately not started:
  the agent is unrestricted by design (M6), so this is a change of stance, not a fix.

## Done recently

| # | What | Where |
|---|------|-------|
| #27 | Token-budgeted history, per-turn tool binding, background warmup | `context_budget.py`, `tool_router.py`, `warmup.py` |
| #30, #31 | Rate-limit-aware distillation tier chain, local model last | `fallback_llm.py` |
| #32 | Nudges and the summary stopped sending non-leading system messages | `routing.py`, `langbot.py` |
| #33 | Tool-call repair counters in `/health` | `tool_call_repair.py` |
| #35 | CI workflow removed — Actions is blocked by the account's billing lock | — |
| #36 | Start-up disk sweep: old scratch entries, abandoned checkpoint threads | `housekeeping.py` |
| #38 | Stagnation guard: a call repeated verbatim in a turn is not re-run | `routing.py` |
| #39 | Prompt composition and prompt-cache cost of compaction, in `/health` | `context_budget.py` |
