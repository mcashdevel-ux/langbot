# TODO

Working list, ordered by what to do first. `CODE_REVIEW.md` holds the findings and their
status; this file holds the *work*, including items that were never review findings.

---

## 1. Vault default — C3 (no live model, needs a decision)

The vault master key is stored recoverably on disk (`./memory/vault/.masterkey`).
Without `LANGBOT_VAULT_PASSWORD`, encryption at rest only defends against other users
on the host.  Pick one approach:

### Option A — Password-on-by-default (safer)
- [ ] Add `_prompt_for_password()` to vault.py — reads password from stdin, wraps
  the master key via PBKDF2.
- [ ] Call it during `bootstrap()` on first vault creation only (not every startup).
- [ ] For an existing unwrapped vault, print a migration prompt.
- [ ] Tests: with/without password; CI-compatible (mock stdin, or skip).

### Option B — Loud startup banner (simplest)
- [ ] Add `vault.warn_unwrapped` config key (default `true`).
- [ ] In `bootstrap()`, print an unmissable banner when the key is unwrapped.
- [ ] Tests: banner appears when on, suppressed when off.

### Either way — fix stale README
- [ ] The `/health` example still says "reserve 8192" and "a 8192 reserve is mostly
  headroom."  Update to 2000.

---

# Doc fixes done (PR #43).

---

## 3. Confirm --jinja → delete tool_call_repair.py (needs one live session)

Small models print tool calls as text. The server-side fix is `llama-server --jinja`.
The signal is `/health` showing `tool-call repairs: 0 recovered`.

- [ ] **Manual:** Start llama-server with `--jinja`, run 10+ tool turns, check `/health`.
- [ ] **If confirmed:** delete `components/tool_call_repair.py`, its import/wiring
  in `langbot.py`, the `compat` config section, `tests/test_tool_call_repair.py`,
  and update the README's "Weak / fine-tuned local models" section.

---

## 4. Eval-harness live run (needs one live session)

The harness uses canned LLM responses. Run 3 critical eval tasks against your live model
and compare real results to expectations:

- [ ] Task 4 — store + recall preference (memory persistence)
- [ ] Task 7 — search → read → patch → diff (full tool chain)
- [ ] Task 14 — long session → compaction (compaction quality)
- [ ] Record `/health` counters for each and note any divergence.

---

## 5. Future / deferred

- **Blast-radius gate** — warn on `rm -rf`, `push --force`, `DROP` (deferred: M6).
- **New agent tools** ✔ (PR #45: plugin system + py_eval + http_request; see `tools/plugins/`). More tools welcome.
- **Terminal UX** ✔ (PR #44: context health bar in REPL, streaming multi-engine search progress, tool timing in panels). More improvements welcome.
- **SearXNG engine adapters** ✔ (PR #44: StackExchange + PubMed added alongside DDG/Wikipedia/arXiv/GitHub).
- **Model compatibility** — prompt/nudge tuning for troublesome local models.
- **CI** — GitHub Actions blocked by billing lock (#35).

---

## Done recently

| # | What |
|---|------|
| PR #45 | Plugin tool system (`tools/plugins/`) — auto-discovered tools with router registration; `py_eval` (sandboxed Python) + `http_request` (direct HTTP client) |
| PR #44 | Terminal UX: context health bar in REPL, streaming multi-engine search progress, tool timing in panels; new engines: StackExchange + PubMed |
| PR #43 | Doc fixes: stale numbers, missing modules, done-item marking across CODE_REVIEW/README/CONTRIBUTING/TODO |
| PR #42 | 7-track improvements: reserve shrink (8192→2000), thinking toggle, embedding routing, multi-engine dedup + authority scoring, memory quality (confidence + pruning), eval harness, A7.4/B3.3/C8.6 |
| #27 | Token-budgeted compaction, per-turn tool binding, warmup thread |
| #30/#31 | Rate-limit-aware distillation tier chain |
| #32 | Nudges/summary no longer non-leading system messages |
| #33 | Tool-call repair counters in `/health` |
| #36 | Start-up disk sweep (scratch + checkpoints) |
| #38 | Stagnation guard for repeated tool calls |
| #39 | Prompt composition stats in `/health` |
| #41 | SearXNG clone logging, pooled HTTP session, vault masking |
