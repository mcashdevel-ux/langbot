# langbot — architecture review (self-analysis)

Subject: `/home/user/ai/repos/langbot` @ `ff9749f` ("vault: warn loudly at startup when the master key is unwrapped").
Scope: the agent itself — graph, safety, memory, routing, context. ~20k LOC Python.

## 1. What it is

A terminal, tool-using agent on **LangGraph + LangChain**, talking to a local
OpenAI-compatible endpoint (config points at `llm-proxy.app.all-hands.dev`,
model `deepseek-v4.1-flash`). Tools: shell, file read/write/patch, web
search/fetch, background tasks, scratch, vault, memory, reflex distillation.
Persistent long-term memory = Chroma + sentence-transformers (MiniLM-L6-v2).

## 2. Graph topology (`langbot.py:1377`)

```
START → reflex ─┬─(route_reflex)→ distill → END
                └→ compact → agent ─┬─(route_agent)→ tools → stuck ─┬→ distill → END
                                    ├→ nudge → agent                 └→ compact → agent
                                    └→ distill → END
```

- **reflex** — deterministic pre-LLM step: keyword-overlap match against
  `./memory/reflexes.json`; a hit runs the stored command via
  `execute_shell_command` (so it still passes the safety layer). Free, fast.
- **compact** — rolling summary / context-budget compaction.
- **agent** — one system message (summary folded in, because llama.cpp rejects a
  second system message), dynamic tool binding, `repair_message()` to recover
  tool calls a small model printed as text.
- **tools** — `ToolNode` + vault redaction + head/tail truncation fallback.
- **stuck** — five-pattern loop detector (repeating_action, repeating_error,
  monologue, alternating, soft_repeating); `nudge` or `halt`.
- **distill** — enqueues a background memory-extraction job (off critical path).

## 3. Safety model — three layers, honestly scoped

1. **Hard denylist** (`components/safety.py:catastrophic_reason`) — fork bombs,
   raw-disk overwrite (`dd of=/dev/sdX`, `mkfs`, `wipefs`, `> /dev/sdX`),
   `rm -rf /|/home|/root|~|$HOME`. Refused outright, no prompt. Splits on
   `&& || ; | \n` so `echo hi; rm -rf /` is caught. The module's own docstring is
   candid: *"a denylist, not a sandbox"* — it can't see base64/octal obfuscation
   or a payload assembled across calls.
2. **Confirmation gate** (`_check_confirm_gate`, Track F) — off by default
   (`tools.confirm_mutating=false`). When on: trusted read-only binaries
   (`ls/cat/grep/…`) fast-pass; gray-zone mutating calls are refused with a
   pending signature, and a whole-message "yes/ok/go ahead" approves exactly that
   signature once. Token-aware, not prefix-aware.
3. **Blast-radius logging** — softer patterns (force-push, DROP TABLE, ordinary
   `rm -rf project/`) are logged, not blocked.

**Assessment:** the layering is sound and the threat model is stated correctly.
The real gap is the one the docstring names: prompt-injection via `fetch_url`
content is not mitigated by a denylist. The right fix is capability limits on
vault secrets + outbound network, not more regex.

## 4. Memory subsystem (`components/memory_store.py`)

Deliberately more than k-NN:
- `min_similarity` floor (0.3) drops noise instead of presenting it as fact.
- over-fetch → dedup by normalized text → **MMR** (`mmr_lambda=0.7`) for n
  *distinct* facts.
- **lexical leg** (substring) fused with the dense leg by **reciprocal rank
  fusion** (k=60) — because MiniLM is weak on exactly what this agent stores:
  paths, env-var names, error codes, command names.
- write-side dedup requires similarity **and** token overlap **and** identical
  identifiers (so "port 8080" ≠ "port 8081").
- writes serialized behind one lock; reads stay lock-free (accepted eventual
  consistency for async distillation).

Distillation is guarded well: only fires when the turn actually ran a
distillable tool (not `remember`/`recall`, not `task_*`/`vault`/`read_scratch`),
so intentions and greetings don't poison memory. There's even a **recall
compliance check** that logs when the final answer references memory without
calling `recall`.

## 5. Context / routing discipline

- **Dynamic tool binding** (`tool_router.py`): core tools always bound; others
  bound on regex keyword triggers ∪ embedding similarity, and once used they stay
  bound for the turn. Rationale is right — a long tool menu degrades a small
  model exactly where the budget is tightest.
- **Nudges** are `HumanMessage`s carrying `[AUTONOMOUS AGENT DIRECTIVE]`, not
  system messages (template constraint), capped at 3/turn.
- **Stagnation guard**: a verbatim repeat of a call already made this turn is
  answered from the transcript instead of re-executed.
- `RECURSION_LIMIT=300` (~148 tool rounds) vs LangGraph's default 25; raised
  from 100 in `9074179` alongside the 1M-token context budget.

## 6. Findings / risks

| # | Severity | Finding |
|---|----------|---------|
| 1 | **High** | **Live API key in plaintext.** `langbot.config.json` and `.env` both contain `sk-REDACTED-ROTATE-ME`. Both are gitignored (good), but the key is now in this session's context and should be **rotated**. |
| 2 | Medium | Vault master key stored recoverably at `./memory/vault/.masterkey` — encryption-at-rest only defends against other host users. TODO.md already tracks this (C3). |
| 3 | Medium | `langbot.py` is 1967 lines / 87 KB and still holds graph wiring, tool defs, REPL, slash handlers, and gate logic. The `components/` split is good; the entrypoint is the remaining monolith. |
| 4 | ~~Low~~ | ~~`_tool_node = ToolNode(tools)` is assigned **twice** in a row (`langbot.py:1253-1254`) — dead duplicate.~~ **Fixed** — single assignment at `langbot.py:1217`. |
| 5 | Low | Denylist bypass by obfuscation (acknowledged in-code). |
| 6 | Low | `tool_call_repair.py` may be deletable once `llama-server --jinja` is confirmed (TODO §3). |

## 7. Verdict

Coherent, well-documented, and unusually honest about its own limits. The
neo-port tracks (safety tiers, confirm gate, `KernelContext` DI, journaled
sessions, stuck detection, reflexes, truncation) are integrated cleanly, with
per-run `KernelContext` isolating concurrent sessions' task managers and
pending-confirm state. The main things to fix are operational (rotate the key,
decide the vault default) rather than architectural.
