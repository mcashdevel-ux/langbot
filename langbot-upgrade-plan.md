# langbot Upgrade Plan
### Scratch-registry unification · duplicate-answer guard · background knowledge/embedding queue

This plan is organized into three independent tracks. Tracks A and B are low-risk,
mechanical, and should land first. Track C is a structural change (introduces a
background thread and a new shared-state boundary) and should land after A/B are
stable, since it touches the same `distill_knowledge` node.

**Suggested order:** A → B → C. A and B can be done in parallel by different people;
C should not start until A is merged, because Track C's `memory_store.py` extraction
touches the same import block in `langbot.py` that Track A leaves alone but Track C
restructures further.

---

## Status — implemented

All three tracks landed together in [#14](https://github.com/mcashdevel-ux/langbot/pull/14)
(commit `27537ed`), not in the staged A → B → C order the plan suggested. 333 tests pass
(`python -m pytest`); there is no lint/typecheck config in the repo, so `pyflakes` was used
as the closest check.

Legend: **DONE** · **DONE (deviation)** — implemented differently than written · **OPEN** —
not done.

| Step | Status | Note |
|---|---|---|
| A1 extract `scratch.py` | DONE (deviation) | the write cap was later removed entirely ([#25](https://github.com/mcashdevel-ux/langbot/pull/25)) |
| A2 point `web_tools.py` at it | DONE (superseded) | the 20,000-char cap it preserved was removed by [#25](https://github.com/mcashdevel-ux/langbot/pull/25) |
| A3 `find_in_files` via scratch, no `-m 5` | DONE | both `grep` and the pure-Python fallback |
| A4 `read_many_files` via scratch | DONE | |
| A5 `read_file` via scratch | DONE | |
| A6 tool docstrings | DONE | |
| A7 Track A tests | DONE (deviation) | steps 1–3 automated; step 4 (manual dogfooding repro) still OPEN — no local LLM server available |
| B1 `final_answers_since_human` | DONE (deviation) | lives in new `components/routing.py`, not `langbot.py` |
| B2 guard in `route_agent` | DONE | plus render-side suppression in `_stream_turn` (beyond plan) |
| B3 diagnostics | DONE (deviation) | warning + DEBUG stream trace shipped; step 3 (root-cause follow-up from real logs) still OPEN |
| B4 Track B tests | DONE | |
| C1 `memory_store.py` | DONE (deviation) | no module-level `memory_collection`; lazy `get_collection()`/`get_embeddings()` instead, so importing the module stays cheap and testable |
| C2 `memory_worker.py` | DONE | the C2.5 batching "fast-follow" shipped in the same change (`MAX_BATCH = 5`) |
| C3 `distill_knowledge` enqueues | DONE | |
| C4 graceful shutdown | DONE | `shutdown()` before `_vault_save()` in `main()` |
| C5 `/health` + `/info` queue metrics | DONE | queue depth and dropped-job count |
| C6 `remember`/`recall`/`/save` stay sync | DONE | |
| C7 `supabase_sync.py` lookup | DONE | prefers `components.memory_store`, uses `store_memories_batch`; old `__main__` lookup kept as fallback |
| C8 Track C tests | DONE (deviation) | functional / non-blocking / backpressure / concurrent read-write / shutdown-drain automated; step 6 (manual soak under real LLM latency) still OPEN |

**Rollback deviation:** the plan's Track C rollback called for keeping the synchronous
distillation path behind an `AGENT_ASYNC_DISTILL` flag for one release cycle. That flag was
**not** implemented — the synchronous path was removed outright, so rolling back Track C
means reverting the commit. The `_write_lock` scope also stayed write-only: the concurrent
read-during-write test passed, so C1's contingency (widening the lock to reads) was not needed.

### Follow-on work not in this plan

- Proprietary `LICENSE` + `CONTRIBUTING.md` + README/pyproject updates — [#14](https://github.com/mcashdevel-ux/langbot/pull/14).
- Optional config file `components/config.py` (`langbot.config.json`, defaults preserved when
  absent) — [#15](https://github.com/mcashdevel-ux/langbot/pull/15). Several constants this
  plan introduced as literals (`READ_INLINE_CHARS`, `GREP_INLINE_LINES`,
  `MANYFILES_INLINE_CHARS`, `MAX_QUEUE_SIZE`, `MAX_BATCH`, the shutdown timeout) are now read
  from that file, falling back to the values written below. `SCRATCH_SAVE_CHARS` and
  `FETCH_SAVE_CHARS` were config keys too until
  [#25](https://github.com/mcashdevel-ux/langbot/pull/25) deleted both caps.

### Remaining open items

All three items below were addressed in a follow-up (tests + logging only; no
live LLM server was available for the interactive repros, but the tool-level
verifications and soak simulation are now in the test suite):

1. **A7.4 (DONE — tool-level)** — `tests/test_dogfooding.py` verifies that
   `find_in_files` against the real repo returns >5 matches and routes through
   scratch, and that scratch round-trips preserve the full result set.
2. **B3.3 (DONE)** — `route_agent` now logs the prior answer's content, the
   message count, and the dropped content when the guard fires
   (`tests/test_routing.py::TestDuplicateAnswerGuardB33` covers it).
3. **C8.6 (DONE)** — `tests/test_memory_worker.py::TestSoakC86` simulates 50
   turns of distillation with controlled latency and confirms the queue drains
   between bursts.

---

## Files touched — summary

| File | Track A | Track B | Track C |
|---|---|---|---|
| `components/scratch.py` *(new)* | created | | |
| `components/routing.py` *(new, beyond plan)* | | created | |
| `components/web_tools.py` | modified | | |
| `components/code_search.py` | modified | | |
| `components/file_ops.py` | modified | | |
| `components/memory_store.py` *(new)* | | | created |
| `components/memory_worker.py` *(new)* | | | created |
| `components/supabase_sync.py` | | | modified |
| `langbot.py` | modified | modified | modified |

---

## Track A — Scratch Registry Unification

**Status: DONE** (`27537ed`, [#14](https://github.com/mcashdevel-ux/langbot/pull/14)) — except
the manual repro in A7.4.

**Why this track exists:** `web_tools.py` and `tasks.py` already solve "don't blow
the context window on one tool result" via an on-disk store + preview + on-demand
paging. `file_ops.py` and `code_search.py` never adopted that pattern, so
`read_any_file` and `find_in_files` currently write full (or silently truncated)
results straight into the message thread. This track makes the pattern shared
infrastructure and applies it to the two tools that skip it.

### Step A1 — Extract the scratch store into its own module

> **Status: DONE (deviation).** `components/scratch.py` exists with all five symbols moved.
> The `max_bytes` default became the named constant `SCRATCH_SAVE_CHARS` (200,000), config-driven
> via `tools.scratch_save_chars` ([#15](https://github.com/mcashdevel-ux/langbot/pull/15)) — and
> was then dropped altogether ([#25](https://github.com/mcashdevel-ux/langbot/pull/25)): truncating
> the copy that exists to preserve the full result defeats the point of the store, so writes are
> now verbatim and only the inline preview is capped.

**Goal / purpose.** `save_to_scratch` / `read_scratch` / `SCRATCH_DIR` currently
live inside `web_tools.py`. `code_search.py` and `file_ops.py` need the same
functions, and importing them from `web_tools` would create an unnecessary
dependency (those modules have nothing to do with the web) and a risk of circular
imports later if `web_tools.py` ever imports from `code_search.py`. Pulling the
scratch logic into its own leaf module makes it a shared utility with no
dependents among the tool modules.

**Means to implement.**
1. Create `components/scratch.py`. Move these symbols verbatim from
   `web_tools.py`: `SCRATCH_DIR`, `_new_scratch_id`, `save_to_scratch`,
   `_valid_utf8_prefix_len`, `read_scratch`.
2. Keep the module dependency-free — it only needs `os`, `uuid`.
3. Add one new parameter to `save_to_scratch` that the existing callers don't need
   but the new callers (Step A3/A5) will: an optional `max_bytes` override, since
   `web_tools.py` hardcodes `FETCH_SAVE_CHARS = 20000` as the on-disk cap but code
   files and grep results may legitimately be larger and shouldn't be silently
   clipped a second time after already being clipped once for the context preview.
   ```python
   def save_to_scratch(content: str, prefix: str = "doc", max_bytes: int = 200_000) -> str:
       sid = _new_scratch_id(prefix)
       path = os.path.join(SCRATCH_DIR, f"{sid}.txt")
       with open(path, "w", encoding="utf-8") as f:
           f.write(content[:max_bytes])
       return sid
   ```
4. No behavior change for existing values — `web_tools.py`'s calls continue to pass
   `prefix="search"` / `prefix="fetch"` and get the same 20,000-char default via an
   explicit `max_bytes=FETCH_SAVE_CHARS` argument, preserving current behavior
   exactly.

### Step A2 — Point `web_tools.py` at the shared module

**Goal / purpose.** Remove the now-duplicated definitions from `web_tools.py`
without changing any of its externally visible behavior, so `search_web`/`fetch_url`
continue to work identically.

**Means to implement.**
1. Delete `SCRATCH_DIR`, `_new_scratch_id`, `save_to_scratch`,
   `_valid_utf8_prefix_len`, `read_scratch` from `web_tools.py`.
2. Add `from .scratch import save_to_scratch, read_scratch` at the top.
3. Update the two call sites (`search_web`, `fetch_url`) to pass
   `max_bytes=FETCH_SAVE_CHARS` explicitly so the on-disk cap is unchanged:
   `save_to_scratch(text, prefix="fetch", max_bytes=FETCH_SAVE_CHARS)`.
4. Re-run the module standalone (`python -c "from components import web_tools"`)
   to confirm no import cycle was introduced.

### Step A3 — Route `find_in_files` through scratch, remove the silent 5-match cap

**Goal / purpose.** This is the bug identified in the dogfooding transcript:
`grep -m 5` caps results per file with no indication to the model or user that
anything was cut off, which produced an incomplete "analyze the tools" answer.
The fix has two parts: (1) stop silently truncating — get the *actual* match
count, and (2) don't dump a potentially large match list straight into context —
route it through scratch like everything else.

**Means to implement.**
1. In `components/code_search.py`, remove `"-m", "5"` from the `grep` invocation
   in `find_in_files`, so the full match set is captured.
2. Split the result into `lines = output.strip().splitlines()`.
3. If `len(lines) <= 20`, return the existing plain-text format unchanged (no
   scratch round-trip needed for small result sets — avoids scratch-file clutter
   for the common case).
4. If `len(lines) > 20`, call `save_to_scratch("\n".join(lines), prefix="grep")`
   and return:
   ```python
   sid = save_to_scratch("\n".join(lines), prefix="grep")
   preview = "\n".join(lines[:20])
   return (f"{len(lines)} matches for '{pattern}' "
           f"(showing first 20; full list at scratch:{sid}):\n{preview}")
   ```
5. Apply the identical change to `_find_in_files_py`, the pure-Python fallback —
   it currently caps at 100 results with `if len(results) >= 100: return
   truncate(...)`; replace that cap with the same "save full list to scratch past
   20 lines" logic so both code paths behave identically regardless of whether
   `grep` is installed.
6. Update the `find_in_files` docstring (both the function docstring and the
   `@tool`-decorated wrapper in `langbot.py`, see Step A6) to mention that large
   result sets are pageable via `read_scratch`.

### Step A4 — Apply the same treatment to `read_many_files`

**Goal / purpose.** `read_many_files` already has a truncation cliff (`if total >
50000: break`) but, like the old `find_in_files`, gives no path to see what was
cut. Since it's already assembling per-file chunks in a loop, this is a small
change riding along with Step A3.

**Means to implement.**
1. In `code_search.py`, keep building `parts` exactly as today, but track the
   *untruncated* total separately from what's returned inline.
2. After the loop, if the untruncated concatenation exceeds, say, 4000 chars,
   save the full concatenation to scratch (`prefix="manyfiles"`) and return only
   the first ~4000 chars inline plus a scratch pointer, instead of the current
   50,000-char inline cliff.
3. Keep the existing per-file `max_chars_per_file` truncation as-is (that's a
   different, per-file safety valve, not the one this fix targets).

### Step A5 — Route `read_file` through scratch above a small inline threshold

**Goal / purpose.** This is the other half of the dogfooding bug: a 31,597-byte
file (`langbot.py`) got dumped as a 20,015-character block directly into the
model's context — roughly 60% of a 32K-token budget in one tool result, with a
silent hard cutoff at the file's `MAX_OUTPUT_CHARS` limit. `read_file` needs the
same preview + scratch-id pattern `fetch_url` already uses for web pages.

**Means to implement.**
1. In `components/file_ops.py`, import `save_to_scratch` from `.scratch`.
2. Add a module-level constant near the top: `READ_INLINE_CHARS = 1500` (matches
   `web_tools.py`'s `FETCH_INLINE_CHARS = 1800` order of magnitude — small enough
   that even several file reads in one turn don't dominate a 32K window).
3. Change the tail of `read_file`:
   ```python
   if not content:
       return "(empty file)"
   if len(content) <= READ_INLINE_CHARS:
       return content
   sid = save_to_scratch(content, prefix="file", max_bytes=200_000)
   preview = content[:READ_INLINE_CHARS]
   return (f"{os.path.basename(path)} — {len(content)} chars, showing first "
           f"{READ_INLINE_CHARS} (full file at scratch:{sid}):\n{preview}")
   ```
4. Remove the old blanket `return truncate(content)` call at the end of
   `read_file` for the text branch — `truncate()` (20,000-char cap, silent) is
   superseded by the scratch pointer. Leave `truncate()` itself in `utils.py`
   unchanged since other call sites (`execute_shell_command`, `git_diff`,
   `find_in_files`'s small-result path) still use it for genuinely bounded
   outputs.
5. Binary-file detection logic (the `\0`-in-first-8KB check) is unaffected —
   binary files still return the short `[Binary file: ...]` message, never go
   through scratch.

### Step A6 — Update tool docstrings in `langbot.py`

**Goal / purpose.** The `@tool`-decorated wrapper functions in `langbot.py` are
what the LLM actually sees as the tool's contract (LangChain uses the docstring
as the tool description sent to the model). If the underlying function's
behavior changes but the docstring doesn't, the model won't know `scratch:id`
results exist or how to page through them.

**Means to implement.**
1. `read_any_file`'s docstring: add a line — "Large files are truncated inline;
   call `read_scratch` with the returned id to see the rest."
2. `find_in_files`'s docstring: add — "Result sets over 20 matches are paged via
   `read_scratch`."
3. `read_many_files`'s docstring: same note.
4. No signature changes needed — `read_scratch` is already a registered tool
   (`@tool read_scratch`), so the model can call it once it knows to.

### Step A7 — Testing for Track A

> **Status: DONE (deviation).** Steps 1–3 are automated in `tests/test_code_search.py`,
> `tests/test_file_ops.py`, and `tests/test_scratch.py` (plus `tests/test_web_tools.py`, which now
> asserts the *absence* of a web save cap after [#25](https://github.com/mcashdevel-ux/langbot/pull/25)).
> **Step 4 is OPEN** — no local OpenAI-compatible LLM server was available, so the dogfooding
> transcript was never replayed end to end.

**Goal / purpose.** Confirm the fix actually changes behavior for the exact case
that exposed the bug, and confirm nothing regresses for small results.

**Means to implement.**
1. Unit test `find_in_files` against a directory with a file containing 30+
   matching lines for one pattern: assert the return string contains
   `"scratch:"` and that `read_scratch(id)` (paged, offset 0) returns the full
   30-line set, not 5.
2. Unit test `find_in_files` against a directory with 3 matches: assert the
   return string does **not** contain `"scratch:"` (small-result path unchanged).
3. Unit test `read_file` against a synthetic 40,000-character text file: assert
   the returned string is short, contains `"scratch:"`, and that
   `read_scratch(id, offset=0, length=40000)` reconstructs the original content
   byte-for-byte.
4. Manual repro: re-run the exact dogfooding transcript ("analyze langbot.py"
   then "analyze the tools") and confirm the second answer now reflects more
   than 5 matches.

---

## Track B — Duplicate-Answer Guard

**Status: DONE** (`27537ed`) — the guard, diagnostics, and tests shipped; the root-cause
investigation in B3.3 is still OPEN. Beyond the plan, routing was extracted into a new
`components/routing.py` (so it is unit-testable without importing `langbot.py`) and
`_stream_turn` also suppresses a duplicate no-tool-call answer at render time, so the second
panel cannot appear even if a duplicate is generated upstream of `route_agent`.

**Why this track exists:** the dogfooding transcript showed two separate
"💬 Answer" panels rendered for one human turn — wasted tokens and a confusing
transcript. Root cause is not fully isolated (possibly the permission-phrase
nudge matching a benign closing sentence, possibly a graph re-entry), so the fix
is a structural invariant rather than a point patch: at most one no-tool-call AI
message reaches `distill`/`END` per human turn.

### Step B1 — Add a turn-scoped "already answered" helper

> **Status: DONE (deviation).** Implemented as `final_answers_since_human` in
> `components/routing.py` (alongside the relocated `ai_turns_since_human`), not in
> `langbot.py`. Body is as written below.

**Goal / purpose.** `route_agent` needs to know, before deciding to nudge or to
finalize, whether a final (non-tool-call) AI message has *already* been produced
since the last `HumanMessage`. This mirrors the existing `_ai_turns_since_human`
helper but counts a different thing (final answers, not all AI turns).

**Means to implement.** In `langbot.py`, next to `_ai_turns_since_human`, add:
```python
def _final_answers_since_human(messages) -> int:
    """Count no-tool-call AI messages back to (not including) the last HumanMessage."""
    count = 0
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            break
        if getattr(m, "type", None) == "ai" and not (getattr(m, "tool_calls", None) or []):
            count += 1
    return count
```

### Step B2 — Modify `route_agent` to enforce the invariant

**Goal / purpose.** Prevent a second final answer from ever reaching
`_render_message`/the user, regardless of what caused the first duplicate to be
generated. This is a defensive guard, not a root-cause fix — it should be paired
with the logging in Step B3 so the actual cause can still be diagnosed and
addressed separately.

**Means to implement.** Change the branch in `route_agent` that currently only
handles the nudge-vs-distill decision:
```python
def route_agent(state: MessagesState):
    last_msg = state["messages"][-1]
    if last_msg.type == "ai" and getattr(last_msg, "tool_calls", None):
        return "tools"

    if last_msg.type == "ai" and isinstance(getattr(last_msg, "content", None), str):
        # Guard: never let a second final answer reach the user for one turn.
        if _final_answers_since_human(state["messages"][:-1]) > 0:
            logger.warning(
                "route_agent: dropping a duplicate final answer this turn "
                "(content preview: %r)", last_msg.content[:120]
            )
            return "distill"

        content_lower = last_msg.content.lower()
        needs_nudge = (
            any(phrase in content_lower for phrase in PERMISSION_PHRASES)
            or any(pat in content_lower for pat in TOOL_AVOIDANCE_PATTERNS)
        )
        if needs_nudge and _ai_turns_since_human(state["messages"]) < 5:
            return "nudge"

    return "distill"
```
Note the guard is checked *before* the nudge/permission-phrase logic and looks at
`state["messages"][:-1]` (everything except the message just produced), so it
only fires when a prior final answer already exists earlier in the same turn —
it does not block the first legitimate answer.

### Step B3 — Add diagnostics for root-causing the duplicate

> **Status: DONE (deviation).** Sub-steps 1 and 2 shipped (guard warning with content preview;
> DEBUG-gated `(node, message types)` trace in `_stream_turn`). **Sub-step 3 is OPEN** — the
> logs have not been observed in real use yet, so the root cause is still unknown and the guard
> remains the only mitigation.

**Goal / purpose.** The guard in B2 hides the symptom; this step preserves the
ability to find and fix the actual cause later without re-instrumenting from
scratch.

**Means to implement.**
1. The `logger.warning` added in Step B2 already captures every time the guard
   fires, with a content preview — deploy this first and watch logs in normal
   use to see how often it actually triggers.
2. In `_stream_turn`, add a debug-level log of `(node, [m.type for m in
   update.get("messages", [])])` for every chunk yielded by `app.stream(...)`,
   gated behind `logger.isEnabledFor(logging.DEBUG)` so it's zero-cost in normal
   operation. This makes it possible to see, on the next reproduction, whether
   the duplicate came from `agent` being invoked twice with no `tools`/`nudge`
   message between (→ graph/edge issue) or from the local LLM server itself
   returning two completions in one response object (→ `ChatOpenAI`/server
   issue).
3. Once the pattern is identified from logs, file it as a separate, targeted
   follow-up fix; the guard from B2 stays in place regardless as a permanent
   safety net (cheap, and a hard invariant worth keeping even after the root
   cause is fixed).

### Step B4 — Testing for Track B

**Goal / purpose.** Confirm the guard fires only when it should.

**Means to implement.**
1. Construct a synthetic `MessagesState` with `[HumanMessage(...), AIMessage(content="answer 1"), AIMessage(content="answer 2")]` and assert `route_agent` on this state returns `"distill"` (guard fires on the second).
2. Construct `[HumanMessage(...), AIMessage(content="Would you like me to proceed?")]` (single answer, permission phrase, first occurrence) and assert `route_agent` still returns `"nudge"` (guard must not block the *first* answer).
3. Construct a normal single-answer turn and assert `route_agent` returns `"distill"` as before (no regression for the common case).

---

## Track C — Background Knowledge Extraction + Embedding

**Status: DONE** (`27537ed`) — including the C2.5 batching fast-follow, which shipped in the
same change rather than later. The `AGENT_ASYNC_DISTILL` rollback flag from the rollback plan
was **not** implemented. The manual soak in C8.6 is still OPEN.

**Why this track exists:** `distill_knowledge` currently runs synchronously as
the last node before `END` — one extra LLM round-trip plus per-fact embedding
and ChromaDB writes, all blocking the REPL from returning control. Moving this
off the critical path removes that latency from every tool-using turn, and lets
embedding calls be batched instead of issued one-by-one.

### Step C1 — Extract shared memory state into `memory_store.py`

> **Status: DONE (deviation).** `components/memory_store.py` owns the client, collection,
> embeddings, `_write_lock`, `store_memory`, `store_memories_batch`, `recall_memories`, and
> `count`. The canonical collection is **not** a module-level `memory_collection` global as
> sketched below: it is created lazily by `get_collection()` (embeddings likewise via
> `get_embeddings()`, warmed once at startup from `langbot.py`), so importing the module does
> not touch disk or load a model — which is what makes the unit tests possible. Callers that
> the plan expected to import `memory_collection` call `get_collection()` instead.

**Goal / purpose.** Today `chroma_client`, `memory_collection`, `embeddings`,
`_store_memory`, and `_recall_memories` are module-level state inside
`langbot.py`. A background worker thread needs to read/write the same
collection the synchronous `recall`/`remember` tools use — putting the state in
its own module (rather than reaching into `langbot`'s globals via
`sys.modules["__main__"]`, the way `supabase_sync.py` currently does) gives both
the main thread and the worker thread one canonical, lockable access point.

**Means to implement.**
1. Create `components/memory_store.py`. Move from `langbot.py`: the
   `CHROMA_PERSIST_DIR` constant, `_load_embeddings()`, the `chroma_client` /
   `memory_collection` construction, `_store_memory`, `_recall_memories`.
2. Add a module-level `threading.Lock()` (`_write_lock`) guarding
   `memory_collection.add(...)` calls specifically. ChromaDB's
   `PersistentClient` is SQLite-backed under the hood; serializing writes through
   one lock avoids relying on assumptions about its internal concurrency
   guarantees rather than testing them under load.
   ```python
   import threading
   _write_lock = threading.Lock()

   def store_memory(text: str) -> str:
       mem_id = str(uuid.uuid4())
       vector = embeddings.embed_query(text)
       with _write_lock:
           memory_collection.add(ids=[mem_id], embeddings=[vector],
                                  metadatas=[{"text": text, "timestamp": _now_iso()}])
       return mem_id

   def store_memories_batch(texts: list[str]) -> list[str]:
       """Batched variant for the background worker — one embed_documents() call."""
       if not texts:
           return []
       ids = [str(uuid.uuid4()) for _ in texts]
       vectors = embeddings.embed_documents(texts)
       now = _now_iso()
       with _write_lock:
           memory_collection.add(ids=ids, embeddings=vectors,
                                  metadatas=[{"text": t, "timestamp": now} for t in texts])
       return ids
   ```
3. Reads (`_recall_memories` → rename `recall_memories`) don't need the lock —
   ChromaDB reads during a write are the collection's own concern, and worst case
   is a `recall` mid-batch-write momentarily missing the newest facts, which is
   already the accepted eventual-consistency tradeoff of this whole track.
4. `langbot.py` now imports from `memory_store`:
   `from components.memory_store import memory_collection, store_memory, recall_memories, store_memories_batch`.
5. Delete the moved definitions from `langbot.py`; update the `remember`/`recall`
   `@tool` wrappers to call `store_memory`/`recall_memories` from the new module
   instead of the old local closures. Behavior is unchanged — this step is a
   pure relocation plus the added lock.

### Step C2 — Build the background worker: `memory_worker.py`

**Goal / purpose.** This is the actual queue/thread that moves distillation and
embedding off the graph's execution path. Modeled loosely on
`tasks.BackgroundTaskManager` for the "own thread + status tracking" shape, but
purpose-built for in-process Python calls (an LLM invocation + embedding calls)
rather than subprocess management, so it's a new module rather than an
extension of `tasks.py`.

**Means to implement.**
1. Create `components/memory_worker.py` with a single-consumer design: one
   `queue.Queue(maxsize=N)` (not a raw `collections.deque`, since `queue.Queue`
   already provides the thread-safe blocking/bounded semantics this needs
   out of the box) and one daemon worker thread that drains it.
2. Define what gets enqueued — a lightweight, already-extracted snapshot, not
   raw `MessagesState`, so the worker has zero LangChain/LangGraph coupling and
   is trivially testable in isolation:
   ```python
   @dataclass
   class DistillJob:
       user_text: str
       ai_text: str
       tool_context: str        # pre-formatted, same format distill_knowledge builds today
       enqueued_at: float
   ```
3. Core class:
   ```python
   import logging, queue, threading, time
   from .memory_store import store_memories_batch

   logger = logging.getLogger(__name__)
   MAX_QUEUE_SIZE = 50

   class MemoryWorker:
       def __init__(self, llm, max_queue_size: int = MAX_QUEUE_SIZE):
           self._llm = llm
           self._q: "queue.Queue[DistillJob]" = queue.Queue(maxsize=max_queue_size)
           self._stop = threading.Event()
           self._thread = threading.Thread(target=self._run, daemon=True, name="memory-worker")
           self._dropped = 0
           self._thread.start()

       def enqueue(self, job: DistillJob) -> bool:
           """Non-blocking. Returns False (and logs) if the queue is full —
           never blocks the calling graph node."""
           try:
               self._q.put_nowait(job)
               return True
           except queue.Full:
               self._dropped += 1
               logger.warning("memory_worker: queue full, dropping distillation "
                               "job (%d dropped so far)", self._dropped)
               return False

       def qsize(self) -> int:
           return self._q.qsize()

       def dropped_count(self) -> int:
           return self._dropped

       def _run(self):
           while not self._stop.is_set():
               try:
                   job = self._q.get(timeout=0.5)
               except queue.Empty:
                   continue
               try:
                   self._process(job)
               except Exception:
                   logger.exception("memory_worker: job failed, skipping")
               finally:
                   self._q.task_done()

       def _process(self, job: DistillJob):
           facts = self._distill(job)          # LLM call, same prompt as today
           if facts:
               store_memories_batch(facts)      # batched embedding, one call

       def _distill(self, job: DistillJob) -> list[str]:
           # same distillation_prompt construction + JSON parsing that
           # distill_knowledge does today, unchanged logic — moved here verbatim.
           ...

       def shutdown(self, timeout: float = 10.0):
           """Drain remaining jobs (bounded wait), then stop the thread."""
           deadline = time.time() + timeout
           while not self._q.empty() and time.time() < deadline:
               time.sleep(0.1)
           self._stop.set()
           self._thread.join(timeout=2.0)
           remaining = self._q.qsize()
           if remaining:
               logger.warning("memory_worker: shutdown with %d jobs still queued", remaining)
   ```
4. The `_distill` method is a direct move of the existing prompt-building +
   `llm.invoke(...)` + `strip_code_fences` + `json.loads` logic currently inside
   `distill_knowledge` in `langbot.py` — no prompt or parsing changes, just a
   relocation so it runs on the worker thread instead of the graph thread.
5. Batch opportunity: because jobs sit in a queue, a slightly more advanced
   version of `_run` can drain up to K queued jobs at once (`get` one, then
   `get_nowait()` in a loop until empty or K reached) and issue one distillation
   call per job but one **combined** `store_memories_batch` call for all facts
   extracted across the batch — this is what turns "batching" from a
   theoretical benefit into a real one. Include this as a fast-follow, not
   required for the first landing, to keep the initial change reviewable.

### Step C3 — Wire `distill_knowledge` to enqueue instead of block

**Goal / purpose.** This is the actual removal of the synchronous work from the
graph's execution path — the whole point of the track.

**Means to implement.**
1. In `langbot.py`, replace the body of `distill_knowledge` with the same
   *guard* logic it has today (only proceed if the current turn actually
   produced tool results — this check stays, it's still correct and still
   cheap) but replace the LLM call + `_store_memory` loop with a single
   `enqueue` call:
   ```python
   def distill_knowledge(state: MessagesState) -> MessagesState:
       user_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
       ai_msgs = [m for m in state["messages"] if m.type == "ai" and m.content]
       if not user_msgs or not ai_msgs:
           return state

       last_human_idx = max(i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage))
       turn_msgs = state["messages"][last_human_idx:]
       tool_results = [m for m in turn_msgs if getattr(m, "type", None) == "tool"]
       if not tool_results:
           return state

       tool_context = "\n".join(
           f"[{getattr(m, 'name', 'tool')}]: "
           f"{(m.content if isinstance(m.content, str) else str(m.content))[:400]}"
           for m in tool_results
       )
       job = DistillJob(
           user_text=user_msgs[-1].content,
           ai_text=ai_msgs[-1].content,
           tool_context=tool_context,
           enqueued_at=time.time(),
       )
       _memory_worker.enqueue(job)   # non-blocking; drops + logs if the queue is full
       return state
   ```
2. `_memory_worker` is a module-level singleton constructed once, alongside the
   existing `llm`/`embeddings` construction near the top of `langbot.py`:
   `_memory_worker = MemoryWorker(llm=llm)`.
3. Net effect: the `distill` node now does string formatting only (cheap,
   synchronous, no I/O) and returns essentially immediately; `app.stream(...)`
   reaches its final chunk and `_stream_turn` returns control to `run_repl` as
   soon as the agent's rendered answer is done, with distillation happening
   concurrently on the worker thread.

### Step C4 — Graceful shutdown wiring

**Goal / purpose.** Without this, quitting the REPL right after a tool-heavy
turn silently drops that turn's facts — the worker thread is a daemon thread
that dies with the process the instant `main()` returns, mid-job or not.

**Means to implement.**
1. In `langbot.py`'s `main()`, the existing `finally: _vault_save()` block gets
   a sibling call before it:
   ```python
   finally:
       _memory_worker.shutdown(timeout=10.0)
       _vault_save()
   ```
2. `shutdown()`'s bounded wait (Step C2) means a hung LLM call on the worker
   thread can't hang process exit indefinitely — after `timeout` seconds it logs
   a warning about however many jobs are still queued and proceeds with normal
   shutdown. This is a deliberate choice: losing the last turn's facts on a slow
   shutdown is an acceptable tradeoff against making `Ctrl+D` unresponsive.
3. Also handle the `KeyboardInterrupt`/`EOFError` paths in `run_repl` — confirm
   they fall through to `main()`'s `finally` (they do, since `run_repl` is called
   inside the `try` in `main()`) so no separate wiring is needed there.

### Step C5 — Expose queue health via `/health` and `/info`

> **Status: DONE.** `/health` shows memory count, queue depth, dropped jobs, vault credentials,
> and background tasks; `/info` shows memory count and queue depth. (A separate `/config`
> command was added later in [#15](https://github.com/mcashdevel-ux/langbot/pull/15).)

**Goal / purpose.** With distillation now asynchronous, there's no more
inline confirmation that a fact was stored — operators/users need a way to see
queue depth and whether jobs are being dropped, otherwise a stuck or overloaded
worker fails silently.

**Means to implement.** In `_handle_slash`, extend the existing `health` and
`info` branches:
```python
if cmd == "health":
    ...
    ui.kv("memory queue depth", str(_memory_worker.qsize()))
    ui.kv("memory jobs dropped", str(_memory_worker.dropped_count()))
    return False
```
No new slash command needed — this rides along with the existing dashboard.

### Step C6 — Update `remember`/`recall` tool wrappers

**Goal / purpose.** These tools currently call the local `_store_memory`/
`_recall_memories` closures directly (synchronous, user-invoked — e.g. `/save`
or the model calling `remember(...)` explicitly). They should keep working
exactly as before; only the *automatic* distillation path moves to the
background queue. Manual `remember` calls stay synchronous because the user/
model is explicitly asking for an immediate, confirmable write.

**Means to implement.**
1. Update the `@tool remember` and `@tool recall` wrapper bodies in `langbot.py`
   to call `memory_store.store_memory` / `memory_store.recall_memories` (the
   relocated Step C1 functions) instead of the old local closures.
2. The `/save` slash command in `_handle_slash` gets the same import swap.
3. No behavior change from the user's perspective — `remember`/`recall`/`/save`
   remain synchronous and immediately confirmable; only the *automatic*
   post-turn distillation is now async.

### Step C7 — Update `supabase_sync.py`'s memory lookup

**Goal / purpose.** `supabase_sync.py` currently reaches into
`sys.modules["__main__"].memory_collection` via `getattr` to find the live
ChromaDB collection. Since the collection now lives in `memory_store.py`
instead of as a `langbot.py` global, this lookup needs to point at the new
location — otherwise `supabase_sync` silently falls back to its "construct a
fresh PersistentClient" path on every call, which still works but bypasses the
in-memory instance and the write lock from Step C1.

**Means to implement.**
1. In `supabase_sync.py`'s `_local_facts()` and `_store_facts_locally_batch()`,
   change the lookup order: try `import components.memory_store as _ms; return
   _ms.memory_collection` first, falling back to the existing
   `sys.modules["__main__"]` lookup only for backward compatibility, then the
   existing "construct fresh client" fallback last.
2. For `_store_facts_locally_batch`, prefer calling the new
   `memory_store.store_memories_batch(facts)` directly instead of
   hand-rolling its own embedding + `collection.add()` logic — this also gets
   Supabase-pulled facts the same write-lock protection as Step C1, which they
   don't currently have.

### Step C8 — Testing and concurrency validation for Track C

> **Status: DONE (deviation).** Sub-steps 1–5 are automated in `tests/test_memory_worker.py`
> and `tests/test_memory_store.py` (real temporary Chroma collection + deterministic fake
> embedder). Sub-step 4 passed with the write-only lock, so the lock scope was left as-is.
> **Sub-step 6 (manual soak) is OPEN.**

**Goal / purpose.** This track introduces a background thread touching shared
state — the highest-risk change of the three tracks — so it needs targeted
concurrency testing, not just functional testing.

**Means to implement.**
1. **Functional:** enqueue a `DistillJob` with known tool_context, wait (poll
   `qsize() == 0`), then `recall_memories(...)` and assert the expected fact
   is retrievable — confirms the end-to-end path works outside the graph.
2. **Non-blocking guarantee:** call `enqueue()` in a tight loop from the main
   thread while the worker is deliberately slowed (mock `_distill` with a
   `time.sleep`), and assert each `enqueue()` call returns in well under the
   sleep duration (proves the graph node truly isn't blocked).
3. **Backpressure:** fill the queue past `max_queue_size` and assert `enqueue()`
   returns `False` and `dropped_count()` increments, rather than raising or
   blocking.
4. **Concurrent read/write:** spin up several threads calling `recall_memories`
   while the worker is actively writing batches, run for N seconds, and assert
   no exceptions surface from ChromaDB — validates the `_write_lock` approach
   from Step C1 is sufficient (or reveals it needs to also wrap reads, if
   ChromaDB turns out not to tolerate concurrent read-during-write on the
   `PersistentClient` backend, in which case the lock scope should widen).
5. **Shutdown drain:** enqueue several jobs, immediately call `shutdown(timeout=10)`,
   and assert `qsize() == 0` after it returns (all jobs drained) for a
   fast-completing mock distillation, and assert the warning log fires (not an
   exception) when `timeout` is set too short for a deliberately slow mock.
6. **Manual soak test:** run a long interactive session with several
   tool-heavy turns in a row and watch `/health`'s queue-depth line to confirm
   it drains between turns under realistic (non-mocked) LLM latency, rather
   than growing unbounded.

---

## Rollout order and dependency notes

> **Status: not followed.** All three tracks were implemented and landed in a single commit
> (`27537ed`) in one PR, with the full suite green, rather than staged A → B → C across
> separate merges.

1. **Track A** first — self-contained, no shared-state risk, directly fixes an
   observed bug. Land and verify via Step A7 before touching anything else.
2. **Track B** can land in parallel with A (touches only `route_agent`) — verify
   via Step B4.
3. **Track C** last, and only after A is merged — `memory_store.py`'s extraction
   touches the same top-of-file import block in `langbot.py` that a fast-follow
   to Track A might also touch, and Track C is materially higher-risk
   (background thread + shared mutable state) so it benefits from landing on a
   stable base rather than concurrently with other `langbot.py` edits.

## Rollback plan

> **Status: partially applicable.** A and B are still straight reverts. For C, the
> `AGENT_ASYNC_DISTILL` feature flag below was **not** implemented — the synchronous
> distillation path was deleted, so rolling Track C back means reverting the commit.

- **Track A:** each of A3/A4/A5 is an isolated function-body change with a clear
  before/after; revert is a straight `git revert` per commit, no data migration
  involved (scratch files are ephemeral and additive — un-reverting doesn't
  strand any state).
- **Track B:** the guard is additive logic in `route_agent`; reverting removes
  the guard and restores prior (buggy but understood) behavior with no other
  side effects.
- **Track C:** riskiest to roll back mid-session because `memory_store.py`
  becomes the single source of truth for `memory_collection`. Rollback plan:
  keep `distill_knowledge`'s synchronous code path in a feature-flagged branch
  (`AGENT_ASYNC_DISTILL` env var, default on) for one release cycle rather than
  deleting it outright, so a bad interaction in production can be reverted with
  a config flip instead of a code revert + redeploy.
