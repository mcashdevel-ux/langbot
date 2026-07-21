Todo: Async migration (branch: async)

Overview

Create a safe, non-breaking async orchestration and warmup foundation for langbot. Deliver a minimal-impact implementation that improves startup responsiveness by parallelizing heavy initialization and prepares the codebase for a larger async refactor.

Goals

- Make startup warmups (embeddings, chroma, checkpointer) run concurrently without blocking REPL.
- Keep current behavior unchanged for users (backwards-compatible): REPL should start even if warmups are in progress.
- Provide clear status/fallbacks for tools that depend on warming resources.
- Add observability and tests to assert behavior.

Scope (what this todo covers)

1) Implement background warmup orchestration (threads + optional asyncio glue) for heavy components.
2) Add "is_ready" checks and short waits/fallbacks to memory-related tools (store/recall).
3) Add UI indicators (/health, simple banner) to show warmup state and memory availability.
4) Add unit and integration tests that simulate warmup slow/failure modes.
5) Documentation: README/CHANGELOG note and the new todo.md.

Out of scope for this task (future work)

- Full async refactor of the REPL, message graph, and tool node execution.
- Replacing sync libraries with async-native equivalents (will evaluate later).

Deliverables

- todo.md (this file) explaining the plan and tasks
- A small, safe code change on branch `async` implementing background warmup
- Tests validating behavior and graceful degradation
- CI steps (reuse existing pytest pipeline)

Detailed tasks

1) Design & architecture (1 day)
   - Enumerate dependencies and which can be initialized concurrently.
   - Decide thread vs asyncio approach for warmup (threads recommended for native libs).
   - Define readiness semantics and timeouts.

2) Implementation (1-2 days)
   - Add WarmupManager or lightweight background thread init in langbot.py.
   - Introduce flags/events: embeddings_ready, chroma_ready, checkpointer_ready.
   - Ensure vault bootstrap remains synchronous and runs before any network calls.
   - Modify memory functions (_store_memory, _recall_memories) to start warmup and return helpful messages when not ready.
   - Add monitoring hooks: simple counters and log messages; expose via /health command.

3) Tests (1 day)
   - Unit tests:
     - When warmup succeeds: memory store/recall works end-to-end (mock embeddings/chroma to be fast).
     - When warmup is delayed: functions return the expected fallback messages/empty results.
     - Failure modes: chroma initialization fails -> fallback to in-memory or benign error.
   - Integration/smoke tests:
     - Start main() with warmup thread in background and assert REPL prompt appears quickly (use harness or call run_repl with mocked input).
   - Run full pytest suite and ensure no regressions (existing tests should pass).

4) Safety & rollback (hours inline)
   - Add defensive try/except around warmup tasks; log and surface errors but do not crash.
   - Rollback plan: revert commit on branch async if tests fail or users report regressions.

5) Documentation & release notes (a few hours)
   - Add /health message updates, README note that startup is now non-blocking and memory may be initializing.

Files to change (primary)

- langbot.py: warmup orchestration + guard checks
- components/* (if needed): minor wrappers for threads/async
- tests/: add new tests for warmup behavior
- README.md or docs/CHANGELOG.md: note change

Testing commands & CI

- Local tests:
  - pytest -q
  - python -m pytest tests/test_warmup.py::TestWarmup::test_store_before_ready (new test)
- Linting: reuse existing project linters if any

Risk analysis

- Thread-safety of chromadb and SQLite: if a library requires main-thread initialization, warmup must run in main thread. Mitigation: initialize DB/checkpointer in main() and start embeddings in background.
- OOM / GPU duplication: avoid multiprocessing; use threads only.
- Silent failures: add explicit logging and surface the cause via /health and startup messages.

Estimated effort

- Design & implementation: 1-2 days
- Tests & QA: 0.5-1 day
- Docs & release notes: 0.5 day

Acceptance criteria

- REPL becomes responsive faster than before when launching the app on machines where embeddings/chroma are slow to init.
- Memory-storing tools behave gracefully when memory system is still initializing.
- All tests pass (existing suite + new tests).

Commands to implement locally (for reviewers)

- Create branch and run tests:
  git checkout -b async
  pytest -q

- Run and observe:
  python -m langbot
  # Observe banner and /health showing warmup status

Rollback

- If issues arise:
  git checkout main
  git branch -D async

Notes

- This todo intentionally keeps changes small and reversible. After this landing, follow-up work can migrate to an asyncio foundation and parallel tool execution.


End of todo.md
