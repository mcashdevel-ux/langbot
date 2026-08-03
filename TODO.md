# TODO

Working list, ordered by what to do first. `CODE_REVIEW.md` holds the findings and their
status; this file holds the *work*, including items that were never review findings.

Grouped by what a task needs, because that is what actually blocks it: a live model, a
decision, or neither.

## Needs neither — config and logic only

- [x] **Task-based eval harness.** 10–15 scripted REPL tasks run after each merge. Unit tests
      confirm each part does what it is told in isolation; only this shows whether compaction
      and tool routing are net-positive *together*. Biggest lift here, and the thing that
      makes every claim below checkable.
      → `tests/test_eval_harness.py` — 15 end-to-end scenarios, 629 total tests.

## Needs a decision

- [ ] **Vault default (C3).** Pick one: default `LANGBOT_VAULT_PASSWORD` on, prompting at
      first run, or make the startup banner warn loudly while it is unset. Either closes C3;
      leaving it is what keeps the master key recoverable beside the ciphertext.

## Needs a real session against the local model

- [ ] **Confirm `--jinja` took.** Grammar-constrained tool calls are a `llama-server` flag,
      outside this repo. The signal is `/health`'s `tool-call repairs` staying near
      `0 recovered` over a real session; if it does, start deleting the parts of
      `tool_call_repair.py` that are then dead weight instead of keeping them as insurance.
- [x] **Shrink `context.reserve_tokens` (8192).** Measured offline: tool schemas cost ~950
      tokens for a typical per-turn binding and the system prompt 191, so overhead runs
      ~1.2K. Handed the difference back to conversation.  Default is now 2000.
- [x] **Thinking-mode toggle for Qwen3.** `llm.thinking_mode` config (`"auto"`/`"off"`/`"on"`)
      controls whether the model reasons before answering.  `/health` reports accumulated
      thinking-token overhead so the cost of reasoning can be measured.
- [x] **Embedding-based tool routing.** `tool_router` now selects tools by MiniLM cosine
      similarity in addition to regex triggers.  "check what's stored for auth" binds `vault`.
      Regulated by `tools.embedding_routing` / `tools.embedding_threshold`.

## Deferred on purpose

- **Blast-radius gate** — a justification field required before the shell tool runs
  `rm -rf`, `push --force`, `DROP`/`TRUNCATE`, or a vault write. Deliberately not started:
  the agent is unrestricted by design (M6), so this is a change of stance, not a fix.

## Done recently

| # | What | Where |
|---|------|-------|
| — | 7-track improvements: reserve_tokens shrink, thinking toggle, embedding routing, multi-engine dedup, memory quality (confidence + pruning), eval harness, A7.4/B3.3/C8.6 | many files (see PR) |
| #27 | Token-budgeted history, per-turn tool binding, background warmup | `context_budget.py`, `tool_router.py`, `warmup.py` |
| #30, #31 | Rate-limit-aware distillation tier chain, local model last | `fallback_llm.py` |
| #32 | Nudges and the summary stopped sending non-leading system messages | `routing.py`, `langbot.py` |
| #33 | Tool-call repair counters in `/health` | `tool_call_repair.py` |
| #35 | CI workflow removed — Actions is blocked by the account's billing lock | — |
| #36 | Start-up disk sweep: old scratch entries, abandoned checkpoint threads | `housekeeping.py` |
| #38 | Stagnation guard: a call repeated verbatim in a turn is not re-run | `routing.py` |
| #39 | Prompt composition and prompt-cache cost of compaction, in `/health` | `context_budget.py` |
| #41 | Nits L4, L5, L7, L8, L9 — explicit SearXNG clone, pooled session, vault masking | `engines.py`, `vault.py` |
