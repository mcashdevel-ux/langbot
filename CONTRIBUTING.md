# Contributing to langbot

langbot is **proprietary software** (see [`LICENSE`](./LICENSE)), but contributions are
welcome. By opening an issue or pull request you agree to the contribution terms in
[section 3 of the license](./LICENSE): your contribution is assigned to the copyright
holder (or, where assignment isn't permitted, licensed to it without restriction).

## Where help is most wanted

- **Tool coverage** *（PR #45: plugin tool system — new tools now go in `tools/plugins/`）* — additional tools
  in {{langbot}} are always welcome. (each tool is a small, testable
  module; `langbot.py` only wires it up as a `@tool`). Anything returning potentially large
  output should go through `components/scratch.py` rather than into the model's context.
- **Search engines** *（PR #42: multi-engine + dedup + authority scoring; PR #44: StackExchange + PubMed）* — more engine adapters and deeper ranking improvements still welcome.
  See `components/engines.py` / `components/web_tools.py`.
- **Memory quality** *（PR #42: improved distillation + confidence scoring + auto-pruning）* — further improvements welcome, especially staleness detection and recall-frequency-based re-ranking.
  See `components/memory_store.py` / `components/memory_worker.py` / `housekeeping.py`.
- **Model compatibility** — prompt and nudge tuning in `components/routing.py` for local
  models that stall, ask for permission, or emit tool calls as code blocks.
- **Terminal UX** *（PR #44: context health bar, streaming search progress, tool timing）* — `components/console.py` and `components/input.py` (rendering, streaming,
  editing, paste handling).
- **Safety** — sandboxing options for `execute_shell_command`, and stronger secret redaction
  in `components/vault.py`.
- **Docs and tests** — coverage for edge cases, plus keeping `README.md` and
  `MEMORY_POLICY.md` accurate.

Good first issues: anything labelled `good first issue`, or picking one unaddressed item
from [`CODE_REVIEW.md`](./CODE_REVIEW.md).

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

A live LLM server is **not** required for the test suite — the `components/` modules are
tested in isolation. To run the agent itself you need a local OpenAI-compatible endpoint
(see [`README.md`](./README.md)).

## Ground rules

1. **Keep changes focused.** One concern per pull request; no drive-by reformatting.
2. **Add tests.** Every behaviour change needs a test under `tests/`. Tests must not touch
   the network, a live LLM, or state outside `tmp_path` — see the scratch-dir and Chroma-dir
   fixtures in `tests/conftest.py` and `tests/test_memory_store.py` for the pattern.
3. **Respect the memory policy.** New persistent state goes under `./memory/` and is
   configurable by environment variable — see [`MEMORY_POLICY.md`](./MEMORY_POLICY.md).
4. **Never block the interactive loop.** Slow work (LLM calls, embeddings, network I/O)
   belongs in a background worker, as with `components/memory_worker.py`.
5. **Keep tool output context-cheap.** Return a short preview plus a `scratch:` id instead of
   dumping large payloads into the conversation.
6. **No secrets.** Never commit credentials, tokens, or files under `memory/vault/`. Values
   read from the vault must stay redacted in tool output.
7. **Match the surrounding style.** Standard-library-only where practical, module-level
   docstrings explaining *why*, and comments only where the code can't speak for itself.
8. **Run the suite before pushing:** `python -m pytest` (all tests must pass). There is no
   CI, so this is the only gate: the suite must also pass with an empty Hugging Face cache
   and `HF_HUB_OFFLINE=1`, which is what rule 2 means in practice.

## Pull requests

- Describe *what* changed and *why*, and note any behaviour visible to the agent or the
  user (new tool, changed tool output shape, new slash command, new env var).
- Update `README.md`, `MEMORY_POLICY.md`, and this file when your change makes them stale.
- Include the relevant test output in the description.

## Reporting bugs and security issues

- **Bugs:** open an issue with the failing command or prompt, the expected vs. actual
  behaviour, your Python version, and any relevant `/health` output.
- **Security:** do **not** open a public issue for a vulnerability that could expose
  credentials or allow remote code execution. Contact the copyright holder privately via
  the repository's issue tracker (a minimal, non-exploitable report requesting a private
  channel) instead.
