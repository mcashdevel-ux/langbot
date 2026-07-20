# Code Review — mcashdevel-ux/langbot

**Scope note:** You have **no pull requests** in this repo (any state), so per your fallback
instruction I reviewed your most recent commits. There is only **one commit**
(`2b40943` "Initial commit", authored by you) which adds all 6 Python files
(2,451 lines). This review covers that commit in full.

**Repo shape:** The runnable entrypoint is `langbot.py` (a LangGraph/LangChain agent). It
imports only `web_tools.py`, which imports `engines.py`. The other three files —
`vault.py`, `input.py`, `console.py` — are **not imported by `langbot.py`** and use a
different plugin framework ("SAGE": `agent.register_tool`, `agent._tool_observers`,
`agent.config`). They are effectively orphaned relative to the running app.

## Summary

| ID | Severity | File | Issue |
|----|----------|------|-------|
| C1 | Critical | console.py | Backslash-in-f-string → `SyntaxError` on Python < 3.12 (won't import) |
| C2 | Critical | vault.py | Auto-redaction is a no-op (`redact()` never called) |
| C3 | Critical | vault.py | Master key stored in plaintext beside ciphertext (default path) |
| C4 | High | vault.py | Hand-rolled SHA256-CTR crypto |
| M1 | Medium | vault.py | PBKDF2 (100k iters) storm — decrypts every cred on every tool call |
| M2 | Medium | web_tools.py | `read_scratch` mixes byte/char offsets → breaks on non-ASCII |
| M3 | Medium | langbot.py | `_store_memory` shells out to `date` for a timestamp |
| M4 | Medium | langbot.py | `n_results=0` passed to Chroma on empty memory |
| M5 | Medium | vault.py | Vault/input/console not integrated with the LangGraph agent |
| M6 | Medium | langbot.py | Unrestricted shell/file tools, no sandbox |
| L1–L12 | Low | various | Duplication, stale docstrings, missing deps manifest/README/tests |

---

## Critical / High

### C1. `console.py` does not even import on Python < 3.12 (SyntaxError)
`console.py` uses backslash escapes inside f-string expression parts in ~29 places, e.g.:
```python
f"{Fore.MAGENTA}{'\u2500' * pad}{Style.RESET_ALL}")   # line 141
```
Backslashes inside f-string replacement fields were only allowed starting in **Python
3.12 (PEP 701)**. On 3.10/3.11 this is a hard `SyntaxError: f-string expression part
cannot include a backslash` at import time. Verified locally on Python 3.10.12:
`python3 -m py_compile console.py` → FAIL. Since `input.py` imports `console.py`, both
break on <3.12.
**Fix:** hoist the literals out of the f-string, or define module-level constants:
```python
HLINE = "\u2500"          # ─
# before (fails <3.12):
#   f"{Fore.MAGENTA}{'\u2500' * pad}{Style.RESET_ALL}"
# after:
dash = HLINE * pad
f"{Fore.MAGENTA}{dash}{Style.RESET_ALL}"
```
Also add a `requires-python`/README note.

### C2. Vault "auto-redaction" is a complete no-op
`RedactionFilter.redact()` (vault.py:467) is **never called**. The observer wired into the
agent, `_output_redactor()` (vault.py:584), only calls `refresh_patterns()` and returns
`None`; it never invokes `redact()` and never mutates/returns `result`. So the advertised
feature "auto-redacts credential values from tool outputs and logs" does nothing —
credential values will pass through tool output unredacted.
**Fix:** the observer must actually transform output, and the dispatch layer must use the
returned value:
```python
def _output_redactor(tool_name, args, result):
    if not _redactor or tool_name == "vault":
        return result
    _redactor.refresh_patterns()
    return _redactor.redact(result)   # <-- return the redacted text
```

### C3. Vault master key is stored in plaintext next to the ciphertext (default path)
`start()` (vault.py:514) auto-initializes the vault **with no password** when none exists,
which takes the `type="raw"` branch in `init_vault()` and writes the 32-byte master key
base64-encoded into `memory/vault/.masterkey` — right beside `credentials.json`. Anyone
with read access to that directory has both the key and the ciphertext, so the encryption
provides ~no confidentiality at rest (it only obscures values from casual `cat`). This
undercuts the stated goal of "replacing plaintext .env". At minimum:
- `chmod 0600` the `.masterkey` file (currently created with default umask).
- Document clearly that raw mode ≈ obfuscation, and prefer the password-wrapped path.

### C4. Don't roll your own crypto (SHA256-CTR stream cipher)
`_sha256_ctr_encrypt` (vault.py:96) is a hand-rolled stream cipher using SHA256 as a PRF.
It happens to avoid keystream reuse here (fresh random salt **and** nonce per value, so the
derived `enc_key` differs each time), but hand-rolled constructions are fragile and
unnecessary. Recommend `cryptography`'s Fernet or AES-GCM. If the "zero dependencies" goal
is firm, at least document the threat model and the fact that this is non-standard.

---

## Medium

### M1. Redaction/`get_all_plaintext` triggers PBKDF2 storm on every tool call
`_output_redactor` → `refresh_patterns()` → `get_all_plaintext()` decrypts **every**
credential, and each `decrypt_value()` runs PBKDF2-HMAC-SHA256 at **100,000 iterations**
(keys are derived per-value from the stored salt). With N credentials this is N×100k
iterations after *every* tool execution — a large, avoidable latency hit. `redact()` also
calls `get_all_plaintext()` again. **Fix:** cache derived keys / decrypted values in
memory, or cache the compiled redaction patterns and only rebuild on store/remove.

### M2. `read_scratch` mixes byte offsets with character offsets (breaks on non-ASCII)
`web_tools.py:44` opens the file in **text** mode, does `f.seek(max(0, offset))` and
`f.read(length)` (character semantics), but compares against `os.path.getsize(path)`
(bytes) and computes `end = offset + len(chunk)`. For any non-ASCII content the
byte/char mismatch makes `more`/`end` wrong, and arbitrary `seek()` in text mode is not
guaranteed by Python (only seeks to values from `tell()` or 0). **Fix:** open in binary
mode and decode, or track byte offsets consistently.

### M3. `_store_memory` shells out for a timestamp
`langbot.py:64` uses `subprocess.getoutput("date -u +%Y-%m-%dT%H:%M:%SZ")` to get a
timestamp — spawns a shell per stored memory, is non-portable (no `date -u` on Windows),
and is slow. **Fix:** `datetime.now(timezone.utc).isoformat()`.

### M4. `_recall_memories` can pass `n_results=0` to Chroma
`langbot.py:72` uses `n_results=min(n, memory_collection.count())`. When the collection is
empty this is `0`; Chroma rejects `n_results < 1`. It's swallowed by the `recall` tool's
try/except, but the guard should be explicit: `if memory_collection.count() == 0: return []`.

### M5. `vault.py` is not integrated with the actual agent
It targets an API (`agent.register_tool`, `agent._tool_observers`, `agent.config`) that
`langbot.py` doesn't provide (LangGraph uses `@tool` + `ToolNode`). As written the vault
tools, env auto-load, and config population never run in this app. Either add an adapter to
register these tools with the LangGraph agent, or remove the file if it's not meant to ship.
Same applies to `input.py`/`console.py` (langbot.py uses plain `input()`/`print()`).

### M6. Unrestricted shell/file tools with no sandboxing
`execute_shell_command` (`shell=True`, arbitrary command) and `write_any_file`/
`read_any_file` (any path) plus a system prompt telling the model to act without asking are
intentional per the banner, but combined with a local LLM this is a full RCE surface. If
this is ever exposed beyond a trusted single-user terminal, add an allowlist / path jail /
confirmation gate. Worth an explicit warning in the README.

---

## Low / Nits

- **L1 (langbot.py):** The `SQLITE_AVAILABLE` and non-sqlite branches (lines 266–313)
  duplicate the entire REPL loop. Extract a `run_repl(app, config)` helper.
- **L2 (langbot.py:223):** Markdown-fence stripping assumes a newline after the opening
  ```` ``` ````; a single-line fenced reply raises IndexError (caught, but sloppy). Use a
  regex or `strip("`")`-based cleanup.
- **L3 (langbot.py:76):** `meta["text"]` will KeyError if a metadata row lacks `text`;
  use `meta.get("text", "")`.
- **L4 (engines.py:44):** `_script_dir = dirname(dirname(abspath(__file__)))` walks **two**
  levels up (to the parent of the repo) to look for `searxng-src`; likely meant one level.
  Also `_ensure_searx_initialized` will silently `git clone` SearXNG at runtime (network +
  ~slow, 120s timeout) — surprising side effect; make it opt-in/explicit.
- **L5 (engines.py):** Stale docstrings from another project ("SAGE's HTTP stack",
  "features.engines import", "227 engines"). `web_tools.py` imports `from engines import`,
  contradicting the docstring's `from features.engines import`.
- **L6 (engines.py:221):** `engine.categories[0]` assumes the attribute exists and is
  non-empty; guard with `getattr(engine, "categories", None)`.
- **L7 (engines.py:178):** A fresh `requests.Session()` is created and closed per request —
  no pooling benefit; either reuse a module-level session or just call `requests.get`.
- **L8 (vault.py):** `put()` reads `len(self._credentials)` for the MAX_CREDENTIALS/limit
  checks *outside* the lock, then mutates inside — minor TOCTOU if ever multithreaded.
- **L9 (vault.py):** Duplicate masking helpers: `RedactionFilter.get_masked_value` vs
  module-level `_mask_value` (the latter appears unused).
- **L10 (vault.py:591):** `_output_redactor`'s skip-list uses tool names
  (`store_credential`, `get_credential`, …) that don't match the actual registered tool
  name `vault` — dead condition.
- **L11 (input.py):** `read_input`/`setup_readline` reference slash commands
  (`/help`, `/quit`, …) that `langbot.py` never implements; and `input.py` itself is unused.
- **L12 (general):** No `requirements.txt`/`pyproject.toml`, no README, no tests. Add
  dependency pinning (langchain, langgraph, chromadb, sentence-transformers, colorama,
  rich, httpx, requests) and a `requires-python = ">=3.12"` (see C1).

---

## Suggested priorities
1. Fix C1 (import-blocking on <3.12) or pin `requires-python>=3.12`.
2. Decide whether `vault.py`/`input.py`/`console.py` ship — if yes, wire them in (M5) and
   fix C2/C3/M1; if no, drop them from the commit.
3. Fix the `read_scratch` byte/char bug (M2) and the timestamp shell-out (M3).
4. Add dependency manifest + README + a couple of smoke tests (L12).
