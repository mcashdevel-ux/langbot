# Memory & State Files Policy

## Overview
All agent state, memory, and runtime data files **must** be stored in the `./memory/` directory.
The project root must remain clean — it contains only source code, configuration, and documentation.

## Designated Paths

| State Type | Path | Owner |
|-----------|------|-------|
| Conversation checkpoints | `./memory/agent_checkpoints.db*` | langbot.py / LangGraph SqliteSaver |
| Long-term memory (embeddings) | `./memory/agent_memory_chroma/` | components/memory_store.py (Chroma) |
| Encrypted credentials | `./memory/vault/` | components/vault.py |
| Vault master key | `./memory/vault/.masterkey` | components/vault.py (0600 perms) |
| Tool-result scratchpad | `./memory/agent_scratch/` | components/scratch.py |
| Knowledge distillation log | `./memory/knowledge.md` | (user-created docs) |

## Enforcement

### Code Review Checklist
Before committing changes to `langbot.py` or `components/*.py`:
- [ ] Any new state file creation uses a path under `./memory/`
- [ ] No files are created in project root or CWD (e.g., `./myfile.db`)
- [ ] All paths are relative to the repo root and prefixed with `./memory/`
- [ ] Test fixtures do not create root-level state files

### .gitignore
The following entries ensure state files are never committed:
```
# Agent state — never commit
memory/
memory/vault/
agent_memory_chroma/          # legacy, kept for safe cleanup
agent_checkpoints.db*         # legacy, kept for safe cleanup
/memory/agent_checkpoints.db
/memory/agent_checkpoints.db-*
```

### Config File
The optional `langbot.config.json` (see `components/config.py`) can relocate any of these
under its `paths` section: `checkpoint_db`, `chroma_dir`, `scratch_dir`, `tasks_dir`,
`vault_dir`. Defaults stay inside `./memory/`; overrides should too.

### Environment Variables
Environment variables take precedence over the config file.
- `AGENT_SCRATCH_DIR` — defaults to `./memory/agent_scratch` (also `paths.scratch_dir`)
- `AGENT_CHROMA_DIR` — defaults to `./memory/agent_memory_chroma` (also `paths.chroma_dir`)
- `AGENT_TASKS_DIR` — defaults to `./memory/agent_tasks` (also `paths.tasks_dir`)
- `LANGBOT_CONFIG` — path to the config file to load
- `LANGBOT_VAULT_PASSWORD` — optional, for vault key wrapping

The checkpoint DB is config-only (`paths.checkpoint_db`, default
`./memory/agent_checkpoints.db`).

All new dynamic paths should be configurable via the config file (and an environment
variable where a per-run override is useful) with defaults in `./memory/`.

## Future Changes

When adding new persistent state:
1. Create the path under `./memory/{feature_name}/`
2. Add to `.gitignore` if needed
3. Add to this policy table
4. Document in `langbot.py` constants at the top
5. Make it configurable via `paths.*` in `langbot.config.json`, keeping the default here

## Why This Matters
- **Clean repository** — diff and version control focus on source, not runtime state
- **Reproducibility** — state is isolated and can be wiped/reset easily
- **CI/CD** — tests run cleanly without polluting the root directory
- **User clarity** — clear separation: `./memory/` is ephemeral, everything else is code
