# Code Review — mcashdevel-ux/langbot

**Scope note:** The original review covered the single initial commit
(`2b40943` "Initial commit", 6 Python files, ~2,451 lines) since there were no pull
requests to review. Several findings have since been addressed in follow-up PRs:

- **#7** — added this review + the README (docs only).
- **#8** — wired `vault`/`input`/`console` into the LangGraph agent and fixed
  C1, C2, M1, M5 (and confirmed the earlier AES-GCM/permission work for C3/C4).
- **#9** — moved the support modules into a `components/` package.

Statuses below reflect the **current** `main`. Legend: ✅ fixed · ⚠️ partially
addressed / mitigated · ⬜ open · ⚙️ intentional by design.

**Repo shape (current):** `langbot.py` (root) is the LangGraph/LangChain entrypoint. The
support modules now live in `components/` (`web_tools`, `engines`, `vault`, `input`,
`console`, `utils`) and are imported via `components.*`. `vault`/`input`/`console` are no
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
| M2 | Medium | components/web_tools.py | `read_scratch` mixes byte/char offsets → breaks on non-ASCII | ⬜ Open |
| M3 | Medium | langbot.py | `_store_memory` shelled out to `date` for a timestamp | ✅ Fixed |
| M4 | Medium | langbot.py | `n_results=0` passed to Chroma on empty memory | ⬜ Open |
| M5 | Medium | components/vault.py | Vault/input/console not integrated with the agent | ✅ Fixed (#8/#9) |
| M6 | Medium | langbot.py | Unrestricted shell/file tools, no sandbox | ⚙️ By design |
| L1–L12 | Low | various | See below | mixed |

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

### M2. `read_scratch` mixes byte offsets with character offsets — ⬜ Open
`components/web_tools.py` opens the file in **text** mode and does `f.seek(offset)` /
`f.read(length)` (character semantics) but compares against `os.path.getsize(path)`
(bytes) and computes `end = offset + len(chunk)`. For non-ASCII content the byte/char
mismatch makes `more`/`end` wrong, and arbitrary `seek()` in text mode isn't guaranteed.
**Fix:** open in binary mode and decode, or track byte offsets consistently.

### M3. `_store_memory` shelled out for a timestamp — ✅ Fixed
Now uses `datetime.now(timezone.utc).strftime(...)` instead of `subprocess.getoutput("date ...")`.

### M4. `_recall_memories` can pass `n_results=0` to Chroma — ⬜ Open
`langbot.py` still uses `n_results=min(n, memory_collection.count())`, which is `0` on an
empty collection (Chroma rejects `n_results < 1`). It's swallowed by the `recall` tool's
try/except, but the guard should be explicit:
`if memory_collection.count() == 0: return []`.

### M5. `vault`/`input`/`console` were not integrated with the agent — ✅ Fixed
`vault.py` now exposes a framework-agnostic adapter (`bootstrap`, `run_action`, `redact`,
`save`); `langbot.py` registers the `vault` tool, auto-loads stored secrets into the
environment at startup, and uses `input.read_input()` + the `console` UI in the REPL.

### M6. Unrestricted shell/file tools with no sandboxing — ⚙️ By design
`execute_shell_command` (`shell=True`) and `read_any_file`/`write_any_file` (any path),
combined with a system prompt telling the model to act without asking, are intentional and
called out by the startup banner and the README security section. If ever exposed beyond a
trusted single-user terminal, add an allowlist / path jail / confirmation gate.

---

## Low / Nits

- **L1 (langbot.py):** ✅ Fixed — the duplicated REPL loop is now a single `run_repl(app, config)` helper.
- **L2 (langbot.py):** ⬜ Open — markdown-fence stripping does `raw.split("\n", 1)[1]`; a
  single-line fenced reply raises `IndexError`, which the surrounding `except
  json.JSONDecodeError` does not catch. Use a regex or `strip("`")`-based cleanup.
- **L3 (langbot.py):** ⬜ Open — `meta["text"]` in `_recall_memories` will `KeyError` if a
  metadata row lacks `text`; use `meta.get("text", "")`.
- **L4 (components/engines.py):** ⬜ Open — `_script_dir` walks two levels up to look for
  `searxng-src`; the runtime `git clone` side effect is still implicit.
- **L5 (components/engines.py):** ⬜ Open — stale docstrings from another project remain
  (though `web_tools.py` now correctly imports `from .engines import`).
- **L6 (components/engines.py):** ⬜ Open — `engine.categories[0]` assumes the attribute
  exists and is non-empty.
- **L7 (components/engines.py):** ⬜ Open — a fresh `requests.Session()` is created/closed
  per request (no pooling benefit).
- **L8 (components/vault.py):** ⬜ Open — `put()` reads the length for the
  `MAX_CREDENTIALS` check outside the lock; minor TOCTOU if ever multithreaded.
- **L9 (components/vault.py):** ⬜ Open — duplicate masking helpers:
  `RedactionFilter.get_masked_value` vs module-level `_mask_value` (the latter unused).
- **L10 (components/vault.py):** ✅ Fixed — the redactor skip-list is now
  `_CREDENTIAL_TOOL_NAMES`, which includes the actual `vault` tool name (plus the legacy
  per-action names).
- **L11 (components/input.py):** ⚠️ Partial — `input.py` is now used by the REPL, but the
  slash commands it advertises for completion (`/help`, `/quit`, …) are still not
  implemented in `langbot.py`.
- **L12 (general):** ⚠️ Partial — a `README.md`, `requirements-dev.txt`, and a `tests/`
  suite (199 tests) now exist. Still missing a runtime dependency manifest
  (`requirements.txt`/`pyproject.toml`) and a `requires-python` pin.

---

## Remaining suggested priorities
1. Fix the `read_scratch` byte/char bug (M2) and the empty-collection `n_results` guard (M4).
2. Harden distillation parsing (L2) and `meta.get("text", "")` (L3).
3. Add a runtime dependency manifest + `requires-python` (L12); tidy engines nits (L4–L7).
4. Consider password-by-default or a clearer at-rest warning for the vault (C3 remaining).
