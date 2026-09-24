# TODO

Working list, ordered by what to do first. `CODE_REVIEW.md` holds the findings and their
status; this file holds the *work*, including items that were never review findings.

---

## 0. One thread ate the disk — closed, with a caveat

A single long session reached **9 GB** (87% of a 10 GB checkpoint store; see
`docs/sitrep-2026-09-23-checkpoint-db-lock.md`). Root cause B was "fixed" by
`prune_thread_history`, but that only bounds the **active** thread, and
`prune_checkpoints` keeps the newest 20 threads whatever their size — so a fat thread
that is abandoned yet still recent was bounded by **nothing**. Verified live: a 263 MB
thread survived a full sweep untouched (`0 threads, 0 rows`).

- [x] `prune_fat_threads()` — a per-thread ceiling applied to *every* thread
  (`housekeeping.checkpoint_max_thread_mb`, default 256). Over-budget threads are
  trimmed to `checkpoint_keep_per_thread`, halving until they fit or one row remains;
  the newest checkpoint always survives, so a resume still replays from valid state.
- [x] Verified against a copy of the live DB: 263 MB → 13.5 MB, 122 MB → 17.4 MB
  (2,299 rows) in one sweep.
- [x] Verified against the real O(N²) shape: a 1,351 MB single-session thread
  → 35.8 MB in 2.1 s, newest checkpoint kept.
- [x] `tests/test_housekeeping_fat_threads.py` (16 tests). Suite: **848 passed**.

**Still open — the file does not shrink.** Deleting rows only frees pages *inside*
the file; the 9 GB thread leaves a 9 GB file, and the WAL cannot be truncated while a
session holds the DB. Reclamation is offline only:

- [ ] Run `python scripts/vacuum_checkpoints.py` with no session live to actually
  return the space (the live PID holds the DB today).

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

### Option B — Loud startup banner (simplest) ✔ chosen
- [x] Add `vault.warn_unwrapped` config key (default `true`).
- [x] In `bootstrap()`, print an unmissable banner when the key is unwrapped
  (`masterkey_is_unwrapped()` + `_warn_unwrapped_key()`, once per process).
- [x] Tests: banner appears when on, suppressed when off, silent for a wrapped key
  (`tests/test_vault_unwrapped_warning.py`).

### Either way — fix stale README ✔
- [x] The `/health` example no longer says "reserve 8192"; the reserve is 2000 and the
  README documents it as tight-but-measured. Context budget default is now 1M tokens.

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

> **Closed since this list was written:** the unbounded intra-thread checkpoint growth
> (sitrep 2026-09-23, root cause B) is fixed — the start-up sweep now trims the live
> thread's oldest snapshots (`housekeeping.checkpoint_keep_per_thread` / `checkpoint_max_mb`)
> and vacuums once at start. See `docs/sitrep-2026-09-24-checkpoint-growth-fix.md`.

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
| #46 | Bound intra-thread checkpoint growth: trim the live thread's oldest snapshots + start-up vacuum (`prune_thread_history`) |
| #47 | Show model reasoning: recover the dropped `reasoning_content` field (`ReasoningChatOpenAI`), split inline think tags properly, reasoning-token accounting, intent subtitle on tool results |
