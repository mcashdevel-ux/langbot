# Porting Neo Features into langbot — Upgrade Plan

**Status: proposed.** This plan ports the highest-value, lowest-risk features from
the [neo](https://github.com/mcashdevel-ux/neo) agent kernel into langbot.

It is organized into independent tracks, ordered by risk (low → structural).
Each track has a goal, means, tests, and rollback story, mirroring the format
of `langbot-upgrade-plan.md`.

**Source of truth for this plan:** `neo/core/journal.py`, `neo/core/kernel.py`,
`neo/core/stuck.py`, `neo/core/reflex.py`, `neo/core/truncate.py`,
`neo/core/condense.py`, `neo/core/knowledge.py`, `neo/core/toolspec.py`,
`neo/core/safety.py`, `neo/core/context.py`, `neo/tools/essential.py`,
`neo/interfaces/server.py`, `neo/interfaces/daemon.py`.

**Why port at all:** langbot is a LangGraph/LangChain terminal agent with a strong
tool layer (scratch paging, vault, tiered fallback LLM, background memory
worker, Supabase sync, routing nudges). Neo is an async-first, stdlib-lean
kernel whose durable-journal substrate, stuck detection, procedural reflexes,
and head+tail truncation solve problems langbot still papers over.  The
tracks below are the intersection: what neo does that langbot does not, that
fits langbot's existing architecture without a rewrite.

---

## Track A — Head+Tail Truncation for Tool Results (replaces silent `truncate()`)

**Status: DONE.**

**Why.** `components/utils.py::truncate()` silently cuts long tool output at
20,000 chars with no path back to the full content. Neo's `maybe_truncate`
(`neo/core/truncate.py`) keeps the head (~60%) and tail (~40%) inline, saves
the full content to a hash-named file, and tells the model the file path so it
can `read_file`/`view_file` it back — no custom paging protocol, no silent
data loss. langbot already has the scratch store for *tool-authored* saves;
this track is about the *kernel-level* default for every tool result.

**Means.**
1. Add `components/truncate.py` — a direct port of neo's `maybe_truncate`
   (head+tail split, `[... N chars omitted — full output saved to <path> ...]`
   notice, sha256-dedup filename, `save_dir` defaulting to
   `./memory/truncated/` per `docs/policies/MEMORY_POLICY.md`).
2. In `langbot.py`, replace the `truncate()` calls in the tool-result path
   (`execute_shell_command`, `git_diff`, `find_in_files` small-result path) with
   `maybe_truncate(...)`; keep `truncate()` itself for genuinely bounded uses
   (e.g. `remember`'s 200-char preview).
3. Update the tool docstrings that mention truncation to say "head+tail shown
   inline; full output saved to a file — use `read_file`/`view_file` on the
   path in the notice to see it all."
4. `read_any_file`/`read_many_files` already route through scratch (Track A
   of the previous plan); leave them alone — this track only changes the
   kernel-level fallback for results that don't have a scratch path yet.

**Tests.**
- Unit: a 40,000-char synthetic result → return string contains both the
  first and last ~100 chars of the input, contains `"saved to"`, and the
  referenced file exists and round-trips byte-for-byte.

- Unit: a 5,000-char result → returned verbatim, no file created.

- Unit: identical content twice → same filename (dedup), no second write.

**Rollback.** Straight revert; the change is additive (new module, call-site
swap) and no state migration is involved.

---

## Track B — Durable Journaled Sessions (crash-safe resume)

**Status: DONE.**

**Why.** langbot's conversation history lives in the LangGraph SqliteSaver
checkpoint DB. It persists, but there is no *event log*: no way to
browse past sessions by title/activity, search what happened in an old
session, or resume a session after the checkpoint was pruned by
housekeeping. Neo's `Journal` (`neo/core/journal.py`) is an append-only
JSONL event log per session (one dir per session: `events.jsonl`,
`meta.json`, single-writer `session.lock`) with torn-tail recovery, forward-
tolerant readers, optional Fernet encryption, and a `View` that folds the
log into the LLM message list.

**Means.**
1. Add `components/journal.py` — port neo's `EventLog` (append-only JSONL,
   torn-tail truncation, seq recovery) and `Journal` (create/load/list,
   meta.json, single-writer lock, `resume_point()`).  Skip the `View`
   class — langbot's message list already comes from the LangGraph state, so
   the journal is an *audit/observability* layer, not the source of truth.

2. In `langbot.py`, journal every event the graph already produces: user
   message, AI message (content + tool calls), tool call, tool result
   (preview), nudge, condensation (summary/demote), finish.  Hook
   into `_stream_turn` (it already sees every node update) and the slash
   handlers — one `journal.append(Event(...))` per rendered message, best-
   effort (journaling must never break the loop).
3. Session metadata: on `main()` start, create a journal with
   `name="session:<thread_id>"`; set the title from the first user message via
   neo's `_heuristic_title` (no LLM call).
4. Add slash commands `/sessions` (list journals, newest by activity) and
   `/session <id> <query>` (search a past session's events by keyword) —
   direct ports of neo's `session_list`/`session_search` tools, minus the
   tool-calling wrapper.
5. Optional encryption: honor `LANGBOT_SESSION_ENCRYPT` (=1 ephemeral
   key, or a passphrase → PBKDF2, per-session salt in meta.json) exactly
   like neo's `resolve_session_key`.

**Tests.**
- Unit: append → read_all round-trips; a torn trailing line is truncated
  on load; seq counters recover.

- Unit: `Journal.list()` sorts by activity (mtime) descending; meta.json
  corruption skips, doesn't crash.
- Integration: run a synthetic turn through `_stream_turn` (mocked LLM),
  assert the journal contains user/ai/tool/finish events in order.



**Rollback.** Revert; journals are additive files under `./memory/sessions/`
and can be deleted without touching the checkpoint DB.



---

## Track C — Stuck Detection (halt repeating loops, nudge soft loops)

**Status: DONE (tools + router triggers + wiring + tests).**

**Why.** langbot's stagnation guard (`components/routing.py::split_repeated_calls`)
only blocks *verbatim* repeats within one turn. Neo's `StuckDetector`
(`neo/core/stuck.py`) folds the whole recent narrative and catches five
patterns: exact repeating action→result, repeating error, content-only
monologue, A-B-A-B alternation (nudge), and soft repetition with drifting
literals (nudge). — the classic "same command, different PID each time"
loop that exact-match is blind to.

**Means.**
1. Add `components/stuck.py` — port neo's detector verbatim (patterns,
   thresholds, `_VOLATILE` normalization, `_similar` token-overlap,
   `_is_wait` polling exclusion, `_TAIL_WINDOW` slicing).  It operates on
   a message list, not neo events: adapt `_narrative()` to map
   `(AIMessage-with-tool_calls, ToolMessage, AIMessage-without-tool_calls)`
   onto `(tool_call, tool_result, llm_response)`.

2. Wire it into `route_agent` (or a new `stuck` node between `tools` and
   `agent`): after each tool round, run the detector over the current turn's
   messages; on a `halt` verdict, inject a `HumanMessage` directive telling
   the model to stop and answer from what it has (mirroring neo's
   `"(stuck: ...)"` finish); on a `nudge` verdict, inject a softer
   "change approach" message, capped at 3 per turn like the existing
   nudge budget.
3. Journal the verdict (Track B) so the audit trail records why the loop
   stopped.

**Tests.**
- Unit: synthetic message lists for each of the 5 patterns → detector fires
  with the right pattern name; a normal varied turn → no fire.  Port
  neo's own test vectors where applicable.
- Integration: a mocked LLM that repeats the same call 3× → the graph
  terminates the turn with a stuck message instead of a 4th call.

**Rollback.** Revert; the detector is a pure function, no state, no data
migration.

---

## Track D — Procedural Reflexes (deterministic when-X → do-Y rules)

**Status: DONE (store + node + slash command + distill tool + tests).**

**Why.** langbot re-runs the whole LLM for every user message, even ones
that are really a known procedure ("check disk space", "what's the repo
layout"). Neo's `ReflexStore` (`neo/core/reflex.py`) stores distilled
"when trigger words → run this command" rules as JSON and matches by
keyword overlap before the LLM is ever called — deterministic, free, and
fast. Safety still applies to the command it runs (the confirmation gate
included).

**Means.**
1. Add `components/reflex.py` — port neo's store (JSON persistence, atomic
   tmp+rename save, keyword-overlap `match()`, `use()` counter, `disable()`).
2. Add a `reflex` node at the start of the graph (before `compact`): if
   the user message matches a rule, run the command via the existing
   `execute_shell_command` path (safety gate included) and return the
   result as the turn's answer — no LLM call.  If the command needs
   confirmation, route through the existing confirm flow.
3. Add a `/reflex` slash command (list rules, `uses`, disabled state)and
   a `reflex_distill` tool the model can call to store a rule from the
   current turn's successful command (mirroring neo's `distill` tool).

**Tests.**
- Unit: store → match by overlap → run; disabled rules don't match; `use()`
  increments and persists.

- Integration: a rule "check disk → df -h" fires without an LLM call (mock
  the LLM to assert it is never invoked).

**Rollback.** Revert; rules are a JSON file under `./memory/reflexes.json`,
additive and deletable.

.

  **Note:** this is the one track where neo's *tool* (`distill`) has no
  langbot equivalent yet — the model-facing store path is new, not just a
  port.  Keep the first landing to the deterministic match+run path; the
  model-facing distill tool can follow in a fast-follow.



---

## Track E — Session History Tools (`session_list` / `session_search` / `journal_search`)

**Status: DONE (tools + router triggers + wiring + tests).**

**Why.** Once journals exist (Track B), the model should be able to browse
them itself — neo exposes `session_list`, `session_search`, and
`journal_search` as tools so the agent can answer "what did we do last
week?" without the human grepping `./memory/sessions/`.  langbot's
`remember`/`recall` cover *facts*; these cover *events*.



**Means.**
1. Port neo's three tools from `neo/tools/essential.py` into `langbot.py`
   (or `components/session_tools.py`), adapted to langbot's journal
   location and message format.
2. `journal_search` needs access to the current turn's journal — thread it
   through a module-level `_current_journal` set by `_stream_turn` (or read
   the thread_id from the graph config and look up the journal by name).
3. Register them in the `tools` list; add docstrings noting they search
   *events*, not memory facts.



**Tests.**
- Unit: seed a temp journal with known events; each tool returns the expected
  matches and respects `n`/`limit`; unknown session id → clean error, no
  exception.



**Rollback.** Revert; tools are additive.



---

## Track F — Tool Registry with Safety Tiers + Token-Aware Confirmation Gate

**Status: DONE (safety/tier classification + SafetyGate port + tier-gated
binding + config-gated confirmation gate + tests).**  Design notes: the
confirmation gate is **off by default** (`tools.confirm_mutating: false`) so
langbot's autonomy contract ("Never ask for permission") is preserved exactly;
when on, gray-zone mutating calls are refused with a notice, and the user can
approve by replying "yes"/"ok"/"go ahead"/… — the model then re-issues the exact
same call and it runs once.  The safety/tier metadata lives in
`components/tool_router.py` (`_BUILTIN_TOOL_META`, registered at module load)
so any consumer (including unit tests) sees the same classification langbot.py
relies on.  Network tools (`search_web`, `fetch_url`) are tier=server; a device
with `tools.max_tier: 0` gets a core-only, fully-offline agent.  Reflex
rules are exempt from the confirm gate (user-created procedures count as
pre-approved; they still pass through the catastrophic hard-block).

**Why.** langbot binds tools per step via `tool_router.select_tools`, but the
set is chosen by heuristics, not by a declared safety/tier classification. Neo's
`ToolSpec` (`neo/core/toolspec.py`) gives every tool a `safety` class
(`read`/`write`/`exec`) and a `tier` (core/server/vector/mcp/pi) so the
kernel can hide tools whose optional deps aren't installed and route mutating
tools through the confirmation gate uniformly.  Neo's `SafetyGate`
(`neo/core/safety.py`) also improves on langbot's `catastrophic_reason`: it
splits chained commands on separators and checks *every* piece (so
`cat foo; rm -rf /` is not treated as trusted read-only), and it has a
trusted-readonly-binary allowlist that fast-paths `ls`/`cat`/`grep`/… without
a confirmation prompt.



**Means.**
1. Add a `safety` field to langbot's tool definitions (default `read`),
   and a `tier` field (default `core`).  Classify the existing tools:
   `execute_shell_command`, `task_start`, `task_kill`, `patch_file`,
   `write_any_file`, `batch_patch` → `exec`/`write`; everything else →
   `read`.
2. Port neo's `SafetyGate.check_shell` (chained-command splitting,
   trusted-readonly allowlist, catastrophic denylist) into
   `components/safety.py`, replacing the current prefix-based trusted check.
3. In `tools_node`, before running a `write`/`exec` tool, apply the gate:
   hard-block on catastrophic, fast-allow trusted read-only, confirmation
   for the gray zone (reusing the existing confirm flow).  This makes
   the confirmation gate *token-aware*: `cat foo; rm -rf /` gets blocked
   even though it starts with a trusted binary.

4. Tier-gate the tool list: `select_tools` skips tools whose tier exceeds
   the configured max (e.g. no `search_web` on a device without the
   searxng source tree).

**Tests.**
- Unit: `cat foo; rm -rf /` → blocked (the old prefix check would have
  allowed it); `ls -la` → fast-allowed, no confirm; `rm -rf ./build` →
  confirmation path.
- Unit: tier gating hides a high-tier tool from `select_tools` output.



**Rollback.** Revert; classification is declarative data, no migration.



---

## Track G — KernelContext Dependency Injection (multi-agent isolation)

**Status: DONE (context dataclass + config-stash injection + per-thread task
managers + per-run confirm-gate state + tests).**  Design notes: langgraph
1.2.11 auto-injects a ``config: RunnableConfig`` param into any tool that
declares it, strips it from the OpenAI schema sent to the model, and carries
arbitrary ``configurable`` keys — so instead of neo's explicit ``ctx`` param,
``tools_node`` stashes a ``KernelContext`` under
``config["configurable"]["kernel_context"]`` and tools resolve it via
``_resolve_ctx(config)``, falling back to the process-wide singletons when
absent (direct ``.func(...)`` test calls, REPL slash handlers, pre-Track-G
runs)..  Task managers are per-thread (``_thread_tasks`` keyed by thread_id:
a thread reuses its manager across turns; different threads get isolated
managers, so two concurrent sessions don't clobber each other's task lists).
while ``_reflex_store``, ``_current_journal``, and ``_memory_worker`` stay
process-wide singletons by design (documented follow-ups if multi-session
isolation is ever needed)..  The Track F confirmation-gate pending state
(``pending_confirm``/``pending_approved``) lives on the per-run context, so
two sessions can't approve each other's pending calls.  ``_file_backups``/
``_read_cache`` from the original plan turned out not to exist as module-level
globals in langbot (the file layer is stateless/scratch-based), so nothing
to relocate there.

**Why.** langbot's tools reach module-level globals (`_tasks`, `_file_backups`,
`_read_cache`, `_memory_worker`, `_vault_*`).  Neo's `KernelContext`
(`neo/core/context.py`) is a dataclass built per-Agent and injected into every
tool call via a `ctx` parameter — which is what lets neo run many agents in
one process (swarm peers, sessions) without sharing mutable state.  This
track is the enabler for any future langbot multi-session/server work, and it
also makes the tool layer unit-testable without import side effects.



**Means.**
1. Add `components/context.py` — a `KernelContext` dataclass holding the
   task manager, file-backup store, read cache, memory store, vault,
   and the current journal (Track B).
2. Thread it through: langbot's `@tool` wrappers already take explicit
   params; add an optional `ctx` param to the wrappers that need shared state,
   and have `tools_node` build one context per graph invocation (from the
   thread_id) and pass it in.
3. Move the module-level mutable stores (`_tasks`, `_file_backups`,
   `_read_cache`) into the context (or a per-context instance), so two
   concurrent sessions don't clobber each other's tasks/backups/caches.

4. Keep `_memory_worker` and the vault as process-wide singletons for now
   (they're already lock-protected); note it as a follow-up if multi-session
   memory isolation is ever needed.



**Tests.**
- Unit: two contexts → separate task lists, separate file backups, separate
  read caches; no cross-talk.
- Integration: run two graph invocations concurrently (different thread_ids),
  assert no shared-state exceptions.



**Rollback.** Revert; this is the riskiest track (touches every tool wrapper);
land it last, after A–F are stable, and keep it to a pure relocation (no
behavior change) so a revert is mechanical.



---

## Suggested order

**A → B → C → D → E → F → G.**

- **A** first — self-contained, directly fixes a silent-data-loss bug, no
  shared-state risk.
- **B** second — everything else that wants journaling (C's verdicts, E's
  tools) depends on it; it is additive and low-risk.

- **C** third — pure function over the message list, no state; rides on B
  only for the audit trail.
- **D** fourth — new node + JSON store, no shared-state risk; the model-
  facing distill tool can be a fast-follow.

- **E** fifth — depends on B; additive tools.
- **F** sixth — declarative classification + a better gate; touches the
  confirmation path, so it lands after the mechanical tracks are stable.
- **G** last — structural; touches every tool wrapper; only worth it when
  multi-session/server work is actually planned.  **Now DONE** — landed
  after A–F were stable, kept to a pure relocation (no behavior change),
  and covered by `tests/test_langbot_context_wiring.py`.



## Rollback summary

| Track | Rollback |
|---|---|
| A | straight revert (additive module + call-site swap) |
| B | revert; journals are additive files under `./memory/sessions/` |
| C | revert; pure function, no state |
| D | revert; JSON rules file, additive |
| E | revert; additive tools |
| F | revert; declarative classification |
| G | revert; pure relocation, mechanical |

---

## What we deliberately do NOT port

- **Neo's async kernel / step machine** — langbot is LangGraph-based; replacing
  the graph substrate is a rewrite, not a port. The journal (Track B) gives
  the crash-safety benefits without the substrate change.
- **Neo's event bus / typed event stream** — langbot's render path is already
  event-driven enough (`_stream_turn`); a full bus is overkill for a
  single-process terminal agent.
- **Neo's swarm / A2A / worker subprocesses** — langbot has no multi-agent
  story and adding one is a project of its own; Track G is the prerequisite
  if it ever happens, but it is out of scope here.
- **Neo's TF-IDF char-n-gram memory** — langbot's Chroma + lexical search
  already covers this; the MMR re-rank and similarity floor are already
  implemented in `memory_store.py`.
- **Neo's briefing / task distillation / scope filtering** — langbot's
  `distill_knowledge` + rolling summary already cover the distillation half;
  the briefing lane is a knowledge-store feature that would need the Supabase
  sync story reworked, so it stays out of scope.