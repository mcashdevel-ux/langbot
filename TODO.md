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

## 2. Doc fixes — stale references & missing info (no live model)

| # | What | Where | Effort |
|---|------|-------|--------|
| 1 | Stale "556-test suite" in CODE_REVIEW intro → should say 629 | `CODE_REVIEW.md` L27 | 2 min |
| 2 | Stale "reserve 8192" in README context budget example + paragraph | `README.md` | 5 min |
| 3 | `components/warmup.py` missing from README project-layout tree | `README.md` | 2 min |
| 4 | `MEMORY_POLICY.md` not updated for Track 11 `confidence` metadata column | `MEMORY_POLICY.md` | 5 min |
| 5 | `components/supabase_sync.py` exists but is never documented — investigate (feature or dead code?) | `supabase_sync.py` | 15 min |
| 6 | `CONTRIBUTING.md` "Where help is wanted" lists Memory quality and Search engines as open, but both got PR #42 work — add a ✓ or rephrase | `CONTRIBUTING.md` | 10 min |

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
- **New agent tools** — DB queries, email, data extraction (see CONTRIBUTING.md).
- **Terminal UX** — paste handling, streaming improvements.
- **Model compatibility** — prompt/nudge tuning for troublesome local models.
- **CI** — GitHub Actions blocked by billing lock (#35).

---

## Done recently

| # | What |
|---|------|
| PR #42 | 7-track improvements: reserve shrink (8192→2000), thinking toggle, embedding routing, multi-engine dedup + authority scoring, memory quality (confidence + pruning), eval harness, A7.4/B3.3/C8.6 |
| #27 | Token-budgeted compaction, per-turn tool binding, warmup thread |
| #30/#31 | Rate-limit-aware distillation tier chain |
| #32 | Nudges/summary no longer non-leading system messages |
| #33 | Tool-call repair counters in `/health` |
| #36 | Start-up disk sweep (scratch + checkpoints) |
| #38 | Stagnation guard for repeated tool calls |
| #39 | Prompt composition stats in `/health` |
| #41 | SearXNG clone logging, pooled HTTP session, vault masking |
