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
  These closed M2, M4, L2, L3, L6, L11 and L12; the statuses below were re-verified
  against `main` at `433c331`.

Statuses below reflect the **current** `main`. Legend: ✅ fixed · ⚠️ partially
addressed / mitigated · ⬜ open · ⚙️ intentional by design.

**Repo shape (current):** `langbot.py` (root) is the LangGraph/LangChain entrypoint and
holds only the tool wiring, the graph, and the REPL. Everything else lives in
`components/`: `web_tools`, `engines`, `scratch`, `file_ops`, `code_search`, `tasks`,
`memory_store`, `memory_worker`, `supabase_sync`, `routing`, `tool_call_repair`, `config`,
`logging_setup`, `vault`, `input`, `console`, `utils`. `vault`/`input`/`console` are no
longer orphaned — the vault is exposed as the `vault` tool with startup env auto-load and
output redaction, and the REPL uses the readline input + console UI.

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

---

## Low / Nits

- **L1 (langbot.py):** ✅ Fixed — the duplicated REPL loop is now a single `run_repl(app, config)` helper.
- **L2 (components/utils.py, tool_call_repair.py):** ✅ Fixed — fence stripping is now
  `strip_code_fences()` / `_strip_fences()`, which handle a single-line fenced reply and a
  language tag without the `IndexError`.
- **L3 (components/memory_store.py):** ✅ Fixed — metadata reads go through
  `meta.get("text", "")`; the one `meta["text"]` left (`supabase_sync.py`) is guarded by an
  explicit `"text" in meta`.
- **L4 (components/engines.py):** ⬜ Open — `_script_dir` walks two levels up to look for
  `searxng-src`; the runtime `git clone` side effect is still implicit.
- **L5 (components/engines.py):** ⬜ Open — stale docstrings from another project remain
  (though `web_tools.py` now correctly imports `from .engines import`).
- **L6 (components/engines.py):** ✅ Fixed — now
  `engine.categories[0] if engine.categories else "general"`.
- **L7 (components/engines.py):** ⬜ Open — a fresh `requests.Session()` is created/closed
  per request (no pooling benefit).
- **L8 (components/vault.py):** ⬜ Open — `put()` reads the length for the
  `MAX_CREDENTIALS` check outside the lock; minor TOCTOU if ever multithreaded.
- **L9 (components/vault.py):** ⬜ Open — duplicate masking helpers:
  `RedactionFilter.get_masked_value` vs module-level `_mask_value` (the latter unused).
- **L10 (components/vault.py):** ✅ Fixed — the redactor skip-list is now
  `_CREDENTIAL_TOOL_NAMES`, which includes the actual `vault` tool name (plus the legacy
  per-action names).
- **L11 (components/input.py):** ✅ Fixed — `_handle_slash` in `langbot.py` implements the
  advertised commands (`/help`, `/quit`, `/new`, `/info`, `/health`, `/config`, `/ls`,
  `/knowledge`, `/save`).
- **L12 (general):** ✅ Fixed — `requirements.txt`, `requirements-dev.txt` and a
  `pyproject.toml` with `requires-python = ">=3.10"` now exist alongside a 465-test suite.
- **L13 (components/scratch.py):** ⬜ Open — scratch entries are never pruned. Every large
  tool result leaves a file under `paths.scratch_dir` for the lifetime of the machine, and
  now that saves are uncapped (#25) a few large fetches can be hundreds of MB. Add an age-
  or size-based sweep at startup.
- **L14 (langbot.py):** ⬜ Open — abandoned checkpoint threads accumulate in the SQLite
  checkpoint DB: `/new` and every restart mint a new `thread_id` and nothing ever deletes
  the old rows.

---

## Remaining suggested priorities
1. Constrain tool-call decoding at the server (GBNF grammar / the model's own chat
  template) so `tool_call_repair` stops being the only defence against malformed calls.
2. Reclaim disk: prune scratch entries (L13) and abandoned checkpoint threads (L14).
3. Consider password-by-default or a clearer at-rest warning for the vault (C3 remaining).
4. Tidy the remaining engines/vault nits (L4, L5, L7–L9).
