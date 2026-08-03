# Code Review — mcashdevel-ux/langbot

**Scope note:** The original review covered the single initial commit
(`2b40943` "Initial commit", 6 Python files, ~2,451 lines) since there were no pull
requests to review. Several findings have since been addressed in follow-up PRs:

- **#7** — added this review + the README (docs only).
- **#8** — wired `vault`/`input`/`console` into the LangGraph agent and fixed
  C1, C2, M1, M5 (and confirmed the earlier AES-GCM/permission work for C3/C4).
- **#9** — moved the support modules into a `components/` package.
- **later PRs (through #25)** — split the tools into their own modules, added the
  optional config file, the on-disk scratchpad, the background memory worker and
  tagged/lexical memory search, the tool-call repair layer, and a 465-test suite.
  These closed M2, M4, L2, L3, L6, L11 and L12.
- **#27** — fit the agent to a 32K window and a 9B model: token-budgeted history
  compaction (`context_budget`), per-turn tool binding (`tool_router`), background
  warmup (`warmup`), and the nudge-budget fix. Closed M7 and M8.
- **#30, #31** — distillation on a rate-limit-aware tier chain with the local model last
  (`fallback_llm`), then a swap of the two Groq models deprecated 2026-08-16.
- **#32** — nudges and the rolling summary no longer send a mid-list system message (M9).
- **#33** — `tool_call_repair.stats()`, surfaced by `/health`, so priority 1 below has a
  completion signal.
- **#36** — the start-up disk sweep (`housekeeping`). Closed L13 and L14.

Statuses below reflect the **current** `main`, re-verified at `4c993e0` with a 629-test
suite. Legend: ✅ fixed · ⚠️ partially addressed / mitigated · ⬜ open · ⚙️ intentional
by design.

**Repo shape (current):** `langbot.py` (root) is the LangGraph/LangChain entrypoint and
holds only the tool wiring, the graph, and the REPL. Everything else lives in
`components/`:

| Area | Modules |
|------|---------|
| Tools | `file_ops`, `code_search`, `tasks`, `web_tools`, `engines`, `scratch`, `vault` |
| Memory | `memory_store`, `memory_worker`, `fallback_llm`, `supabase_sync` |
| Prompt economy | `context_budget` (history compaction), `tool_router` (per-turn tool binding) |
| Agent-loop guards | `routing` (nudges, duplicate answers), `tool_call_repair` |
| Lifecycle | `warmup` (background init), `housekeeping` (start-up disk sweep) |
| Plumbing | `config`, `logging_setup`, `input`, `console`, `utils` |

`vault`/`input`/`console` are no longer orphaned — the vault is exposed as the `vault` tool
with startup env auto-load and output redaction, and the REPL uses the readline input +
console UI.

## Summary

| ID | Severity | File | Issue | Status |
|----|----------|------|-------|--------|
| C1 | Critical | components/console.py | Backslash-in-f-string → `SyntaxError` on Python < 3.12 | ✅ Fixed (#8) |
| C2 | Critical | components/vault.py | Auto-redaction is a no-op (`redact()` never called) | ✅ Fixed (#8) |
| C3 | Critical | components/vault.py | Master key stored recoverably beside ciphertext | ⚠️ Mitigated |
| C4 | High | components/vault.py | Hand-rolled SHA256-CTR crypto | ✅ Fixed (AES-256-GCM) |
| M1 | Medium | components/vault.py | PBKDF2 (100k iters) storm on every tool call | ✅ Fixed (#8) |
| M2 | Medium | components/scratch.py | `read_scratch` mixes byte/char offsets → breaks on non-ASCII | ✅ Fixed |
| M3 | Medium | langbot.py | `_store_memory` shelled out to `date` for a timestamp | ✅ Fixed |
| M4 | Medium | components/memory_store.py | `n_results=0` passed to Chroma on empty memory | ✅ Fixed |
| M5 | Medium | components/vault.py | Vault/input/console not integrated with the agent | ✅ Fixed (#8/#9) |
| M6 | Medium | langbot.py | Unrestricted shell/file tools, no sandbox | ⚙️ By design |
| M7 | Medium | langbot.py | Thread history grows unbounded; every turn resends all of it | ✅ Fixed |
| M8 | Medium | components/routing.py | Nudge budget is spent by tool rounds, disabling nudges mid-task | ✅ Fixed |
| M9 | Medium | components/routing.py, langbot.py | Nudges/summary sent as non-leading system messages → server 500 ends the session | ✅ Fixed (#32) |
| M10 | Medium | components/fallback_llm.py | Distillation tiers pinned to models Groq shuts down 2026-08-16 | ✅ Fixed (#31) |
| L1–L14 | Low | various | See below | mixed |

---

## Critical / High

### C1. `console.py` did not import on Python < 3.12 (SyntaxError) — ✅ Fixed
`console.py` used backslash escapes inside f-string expression parts in ~29 places, e.g.
`f"{Fore.MAGENTA}{'\u2500' * pad}{Style.RESET_ALL}"`. Backslashes inside f-string
replacement fields were only allowed starting in **Python 3.12 (PEP 701)**; on 3.10/3.11
this is a hard `SyntaxError` at import time.
**Fix (merged):** the box-drawing characters were hoisted into module-level constants
(`_HLINE`, `_DLINE`, `_MIDDOT`, `_BLOCK_FULL`, `_BLOCK_LIGHT`) referenced inside the
f-strings, so the file imports on 3.10+.

### C2. Vault "auto-redaction" was a complete no-op — ✅ Fixed
`RedactionFilter.redact()` was never called; the observer only ran `refresh_patterns()`
and returned `None`, so credential values passed through tool output unredacted.
**Fix (merged):**
- `_output_redactor` now returns `_redactor.redact(result)`.
- The LangGraph `ToolNode` is wrapped (`tools_node` in `langbot.py`) so every tool
  message's content is passed through `vault.redact()` before it re-enters the model
  (the `vault` tool itself is exempt — `get` is meant to return the value).

### C3. Vault master key stored recoverably beside the ciphertext — ⚠️ Mitigated
Original code auto-initialized a raw master key base64-encoded next to `credentials.json`.
**Addressed:**
- Files are now restricted to `0600` and the vault dir to `0700`.
- Optional `LANGBOT_VAULT_PASSWORD` wraps the master key with a password-derived key.
**Remaining:** with no password set, the key is still stored in recoverable form on disk,
so encryption at rest primarily defends against *other users on the host* rather than
someone who can read the vault directory. Set `LANGBOT_VAULT_PASSWORD` for real at-rest
protection.

### C4. Hand-rolled SHA256-CTR stream cipher — ✅ Fixed
New credentials are encrypted with **AES-256-GCM** (`cryptography`), with a `v2:` blob
prefix. Legacy SHA256-CTR blobs remain decryptable for backward compatibility and are
migrated on write.

---

## Medium

### M1. Redaction triggered a PBKDF2 storm on every tool call — ✅ Fixed
`RedactionFilter` now caches decrypted values on `refresh_patterns()` (called only on
store/remove), so `redact()` no longer decrypts every credential (PBKDF2 @100k iters) on
each tool call. It also masks longer values first to avoid partial masking.

### M2. `read_scratch` mixed byte offsets with character offsets — ✅ Fixed
The scratchpad moved out of `web_tools.py` into `components/scratch.py`, and
`read_scratch` now opens the file in **binary** mode, so offsets, `length`, `end` and
`total = os.path.getsize(path)` are all bytes. A page boundary landing inside a multi-byte
character is repaired by over-reading up to 3 bytes and trimming back to the longest valid
UTF-8 prefix, so paging with the returned `end` reassembles the content losslessly —
asserted by `tests/test_scratch.py::test_read_non_ascii_byte_offsets_consistent`.

### M3. `_store_memory` shelled out for a timestamp — ✅ Fixed
Now uses `datetime.now(timezone.utc).strftime(...)` instead of `subprocess.getoutput("date ...")`.

### M4. Recall could pass `n_results=0` to Chroma — ✅ Fixed
Recall now lives in `memory_store.search_memories`, which reads `total = collection.count()`
and returns `[]` before querying when the store is empty; `_dense_candidates` additionally
clamps with `n_results=min(k, total)`, and the dedup path guards on `not collection.count()`.
No call site reaches Chroma with `n_results < 1`.

### M5. `vault`/`input`/`console` were not integrated with the agent — ✅ Fixed
`vault.py` now exposes a framework-agnostic adapter (`bootstrap`, `run_action`, `redact`,
`save`); `langbot.py` registers the `vault` tool, auto-loads stored secrets into the
environment at startup, and uses `input.read_input()` + the `console` UI in the REPL.

### M6. Unrestricted shell/file tools with no sandboxing — ⚙️ By design
`execute_shell_command` (`shell=True`) and `read_any_file`/`write_any_file` (any path),
combined with a system prompt telling the model to act without asking, are intentional and
called out by the startup banner and the README security section. If ever exposed beyond a
trusted single-user terminal, add an allowlist / path jail / confirmation gate.

### M7. No context management: the thread grew without bound — ✅ Fixed
A `compact` node (`components/context_budget.py`) now runs before every agent step: once
the thread plus its rolling summary crosses `context.compact_at` of the usable budget,
everything older than `keep_last_messages` is folded into the summary by one cheap LLM
call and removed from the checkpoint with `RemoveMessage`, so the thread shrinks on disk
as well as in the prompt. Budgets are counted in tokens (`tiktoken` when available), the
split never orphans a tool result from its call, and a summarizer failure leaves the turn
uncompacted rather than losing it.

### M8. The nudge budget was spent by tool rounds — ✅ Fixed
`route_agent` now gates on `nudges_since_human`, which counts only messages carrying
`NUDGE_MARKER`, so ordinary tool rounds no longer exhaust the budget. The budget itself
dropped to 3 and the nudge texts are one line each.

### M9. Non-leading system messages ended the session with a 500 — ✅ Fixed (#32)
`nudge_agent` appended a `SystemMessage` to the end of the thread, and `agent()` sent the
rolling summary as a second one; served chat templates reject any system message that is not
first (llama.cpp: *"System message must be at the beginning"*), so the next step 500'd and
the turn died. Nudges are now `HumanMessage`s carrying `NUDGE_MARKER`, and the summary is
folded into the single leading system prompt. The counters (and `distill_knowledge`) skip
nudges via `routing.is_nudge`, so a nudge no longer reads as the user speaking.

### M10. Distillation tiers pinned to models Groq is shutting down — ✅ Fixed (#31)
`llama-3.3-70b-versatile` and `llama-3.1-8b-instant` held the first and last hosted slots;
Groq deprecated both on 2026-06-17 with shutdown on 2026-08-16, after which every call to
them would fail and quietly push distillation onto the local model. The chain now follows
Groq's migration guidance: `openai/gpt-oss-120b` → `qwen/qwen3.6-27b` → `openai/gpt-oss-20b`
→ local. Worth re-checking whenever Groq's deprecation page moves.

---

## Low / Nits

- **L1 (langbot.py):** ✅ Fixed — the duplicated REPL loop is now a single `run_repl(app, config)` helper.
- **L2 (components/utils.py, tool_call_repair.py):** ✅ Fixed — fence stripping is now
  `strip_code_fences()` / `_strip_fences()`, which handle a single-line fenced reply and a
  language tag without the `IndexError`.
- **L3 (components/memory_store.py):** ✅ Fixed — metadata reads go through
  `meta.get("text", "")`; the one `meta["text"]` left (`supabase_sync.py`) is guarded by an
  explicit `"text" in meta`.
- **L4 (components/engines.py):** ✅ Fixed — the path walk is named `repo_root` and explained,
  and the clone is no longer implicit: it logs at warning level (a search turning into a
  network fetch and tens of megabytes of disk deserves saying), and
  `web.searxng_auto_clone: false` refuses it with an error naming every path searched.
- **L5 (components/engines.py):** ✅ Fixed — the module docstring describes this adapter and
  where the source tree comes from; the logger is `__name__` rather than another project's
  `sage.engines`, and the unused `_ENGINE_CACHE` is gone.
- **L6 (components/engines.py):** ✅ Fixed — now
  `engine.categories[0] if engine.categories else "general"`.
- **L7 (components/engines.py):** ✅ Fixed — one process-wide `requests.Session` via
  `_get_session()`, so a search that fans out over several engines pools its connections
  instead of re-handshaking TLS per request.
- **L8 (components/vault.py):** ✅ Fixed — the `MAX_CREDENTIALS` count is read inside the
  lock. Overwriting an existing credential is no longer refused at capacity, since a
  rotation adds nothing.
- **L9 (components/vault.py):** ✅ Fixed — one `mask_value()`, which
  `RedactionFilter.get_masked_value` delegates to. It keeps the fixed-width `abcd...wxyz`
  form: the old asterisk-per-character mask published the secret's length.
- **L10 (components/vault.py):** ✅ Fixed — the redactor skip-list is now
  `_CREDENTIAL_TOOL_NAMES`, which includes the actual `vault` tool name (plus the legacy
  per-action names).
- **L11 (components/input.py):** ✅ Fixed — `_handle_slash` in `langbot.py` implements the
  advertised commands (`/help`, `/quit`, `/new`, `/info`, `/health`, `/config`, `/ls`,
  `/knowledge`, `/save`).
- **L12 (general):** ✅ Fixed — `requirements.txt`, `requirements-dev.txt` and a
  `pyproject.toml` with `requires-python = ">=3.10"` now exist alongside a 629-test suite.
- **L13 (components/scratch.py):** ✅ Fixed — `components/housekeeping.py` sweeps
  `paths.scratch_dir` once per start on the warmup thread: entries older than
  `housekeeping.scratch_max_age_days` (7) go, then oldest-first until the directory fits
  `scratch_max_total_mb` (512). Recent entries are kept regardless of size.
- **L14 (langbot.py):** ✅ Fixed — the same sweep keeps the active thread plus the
  `housekeeping.checkpoint_keep_threads` (20) most recently written threads and deletes the
  rest from every table carrying a `thread_id`, then `VACUUM`s so the disk is actually
  returned. Recency comes from `rowid` order, since the checkpointer's schema stores no
  timestamp. `/health` reports what the last sweep freed.

---

## Remaining suggested priorities
1. Constrain tool-call decoding at the server (GBNF grammar / the model's own chat
  template) so `tool_call_repair` stops being the only defence against malformed calls.
  The change itself is on the server side (`llama-server --jinja`), outside this repo;
  `/health`'s `tool-call repairs` counters (`tool_call_repair.stats()`) are how you tell
  whether it took effect — a session that stays at `0 recovered` is the evidence that
  this is done, and that the repair/nudge layers can start shrinking.
2. Consider password-by-default or a clearer at-rest warning for the vault (C3 remaining).
3. ~~Tidy the remaining engines/vault nits (L4, L5, L7–L9).~~ ✅ Fixed (#41).

*2026-08 follow-up — 7-track improvements landed (see PR):*
- `reserve_tokens` shrunk from 8192 → 2000 (measured overhead ~1.2K).
- `llm.thinking_mode` config with `/no_think` injection and thinking-token tracking.
- Embedding-based tool routing (MiniLM) as an additive signal to regex triggers.
- Multi-engine search with URL dedup, near-duplicate detection, authority scoring,
  and `engine="auto"`.
- Memory quality: extended distillation prompt (preferences/facts/actions/errors),
  confidence scoring, automatic pruning of stale low-confidence facts.
- Eval harness: 15 end-to-end REPL scenarios in `tests/test_eval_harness.py`.
- Open upgrade-plan items A7.4, B3.3, C8.6 now have test coverage.
