# langbot

A terminal-based, tool-using AI agent built on **LangGraph** + **LangChain**. It talks to
a local OpenAI-compatible LLM endpoint, can run shell commands, read/write files, search
the web (via SearXNG engines) and fetch pages, and keeps a persistent, searchable
long-term memory (Chroma + sentence-transformers).

> ⚠️ **Security warning:** the agent has **unrestricted shell, file, and web access** and
> is prompted to act without asking. Only run it in a trusted, single-user, sandboxed
> environment. See "Security notes" below.

> 📄 **Proprietary software.** All rights reserved — see [`LICENSE`](./LICENSE). Use requires
> written permission from the copyright holder. Contributions are welcome under the terms in
> [`CONTRIBUTING.md`](./CONTRIBUTING.md).

## Features

- **Autonomous agent loop** (`langbot.py`) — LangGraph `StateGraph` with an agent node, a
  `ToolNode`, and an automatic knowledge-distillation node that extracts durable facts
  from each exchange into long-term memory. Distillation is handed to a background
  worker (`components/memory_worker.py`), so it never blocks the REPL from returning;
  `/health` shows its queue depth and dropped-job count.
- **Long-term memory** — Chroma vector store (`agent_memory_chroma/`) with
  `remember` / `recall` tools plus automatic distillation, all behind
  `components/memory_store.py`.
- **Context-cheap tool results** (`components/scratch.py`) — every tool that can return a
  large payload (page fetches, file reads, grep hits) saves the full result to an on-disk
  scratchpad and returns a short preview plus a `scratch:id`; `read_scratch` pages through
  the rest, so nothing is silently truncated and nothing large is force-fed into context.
- **File & code tools** (`components/file_ops.py`, `components/code_search.py`) — hardened
  `read_any_file`/`write_any_file` (binary detection, idempotent writes), surgical
  `patch_file`/`batch_patch` (find/replace with `.py` syntax-check + auto-rollback),
  `git_diff`, and `find_in_files`/`read_many_files`/`glob_list` for navigation.
- **Background task manager** (`components/tasks.py`) — run long-lived commands (servers,
  watchers) with `task_start` and actively manage them via `task_list`/`task_status`/
  `task_output`/`task_kill`; output is captured to `./memory/agent_tasks` and a monitor
  thread updates each task's status the moment it exits.
- **Web tools** (`components/web_tools.py`) — `search_web` and `fetch_url` (via Jina Reader),
  scratchpad-backed like the file and search tools.
- **SearXNG engine adapter** (`components/engines.py`) — runs individual SearXNG engine
  modules directly (no SearXNG web app), supporting many search engines.
- **Conversation persistence** — LangGraph SQLite checkpointer when
  `langgraph-checkpoint-sqlite` is installed, else in-memory.
- **Encrypted credential vault** (`components/vault.py`) — AES-256-GCM encrypted secrets
  store, exposed as the `vault` tool (`store`/`get`/`list`/`remove`/`status`). Stored
  credentials are auto-loaded into the environment at startup, and their values are
  automatically **redacted** from other tools' output before it re-enters the model.
- **Terminal UX** — `components/input.py` (readline history, arrow-key editing, multi-line
  paste detection, Esc-to-cancel) and `components/console.py` (colored output, banners,
  spinners) power the interactive REPL. Every step of a turn — agent thoughts, tool calls,
  tool results and the final answer — streams to the console as its own Rich panel the moment
  it is produced, and the final answer is rendered as Markdown. Ctrl+C interrupts the current
  turn and returns to the prompt; Ctrl+D (or `quit`/`exit`) ends the session. The embedding
  model loads quietly (its `Loading weights` progress bars are suppressed).

## Requirements

- **Python 3.10+** (the code uses `X | None` type syntax).
- A local OpenAI-compatible LLM server (default `http://127.0.0.1:8080/v1`).

### Python dependencies

Runtime dependencies are declared in `requirements.txt` (and `pyproject.toml`,
which also pins `requires-python >= 3.10`). Install them with:

```bash
pip install -r requirements.txt
# or, as a package:  pip install .
```

For development and running the test suite, use `requirements-dev.txt` (it
includes the runtime deps plus `pytest`):

```bash
pip install -r requirements-dev.txt
```

`components/engines.py` additionally needs the SearXNG source on disk. Place it at one of
`./searxng-src`, `~/searxng-src`, or `/usr/local/searxng/searxng-src`, or let the module
clone it automatically on first use:

```bash
git clone --depth 1 https://github.com/searxng/searxng ~/searxng-src
```

## Configuration

The config file is **optional** — every setting has a built-in default, so langbot runs
unchanged with no config file at all. To override something, copy the template and edit
only the keys you care about:

```bash
cp langbot.config.example.json langbot.config.json
```

Lookup order (first file found wins):

1. `$LANGBOT_CONFIG` — explicit path to a JSON config file
2. `./langbot.config.json` — current working directory
3. `~/.config/langbot/config.json`

A missing, malformed, or partially filled file is never fatal: langbot logs a warning
where relevant and falls back to defaults key by key (a wrong-typed value falls back on
its own, the rest of the file still applies). `/config` in the REPL shows which file (if
any) is in use and the values actually in effect.

Precedence for a single setting: **environment variable > config file > default.**

| Section | Keys | Notes |
|---------|------|-------|
| `llm` | `base_url`, `model`, `temperature`, `max_retries` | OpenAI-compatible endpoint |
| `paths` | `checkpoint_db`, `chroma_dir`, `scratch_dir`, `tasks_dir`, `vault_dir` | keep inside `./memory/` per `MEMORY_POLICY.md` |
| `memory` | `collection_name`, `embedding_model`, `embedding_device`, `worker_queue_size`, `worker_batch_size`, `worker_shutdown_timeout` | background distiller + vector store |
| `tools` | `read_inline_chars`, `grep_inline_lines`, `manyfiles_inline_chars`, `scratch_save_chars`, `max_output_chars` | how much tool output goes inline vs. to scratch |
| `web` | `search_snippet_chars`, `search_max_results`, `fetch_inline_chars`, `fetch_save_chars`, `jina_timeout`, `jina_retry_on_429`, `searxng_settings_path`, `searxng_source_dir` | search/fetch behaviour |
| `routing` | `max_nudges_per_turn` | autonomy nudge budget per turn |
| `compat` | `repair_json_tool_calls`, `repair_max_candidates` | recover tool calls from models that print them as text (see below) |

See [`langbot.config.example.json`](./langbot.config.example.json) for every key with its
default value.

Environment variables (override the config file):

- `LANGBOT_CONFIG` — path to the config file to load.
- `SEARXNG_SETTINGS_PATH` — path to a SearXNG `settings.yml` (defaults to
  `/etc/searxng/settings.yml`, then the source's bundled settings).
- `AGENT_SCRATCH_DIR` — where scratch files are written (default
  `./memory/agent_scratch`, per `MEMORY_POLICY.md`).
- `AGENT_CHROMA_DIR` — where the long-term memory store lives (default
  `./memory/agent_memory_chroma`).
- `AGENT_TASKS_DIR` — where background task logs are written (default
  `./memory/agent_tasks`).
- `LANGBOT_VAULT_PASSWORD` — if set, the vault master key is wrapped with a
  password-derived key instead of being stored in recoverable form on disk.

## Usage

Start your local LLM server, then run:

```bash
python langbot.py
```

You'll get an interactive prompt:

```
You: search the web for the latest langgraph release and summarize it
```

Type `quit` or `exit` (or Ctrl+C / Ctrl+D) to leave. Conversation state persists across
runs via the SQLite checkpointer.

Local REPL commands (not sent to the model): `/help`, `/new` (or `/clear`, starts a fresh
conversation thread), `/info`, `/health`, `/config`, `/ls [dir]`, `/knowledge <query>`,
`/save <fact>`, `/quit`.

### Tests

The `components/` modules have a unit-test suite (the heavy LLM deps and a live LLM server
are not required):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

### Available tools

`execute_shell_command`, `read_any_file`, `write_any_file`, `patch_file`, `batch_patch`,
`git_diff`, `find_in_files`, `read_many_files`, `glob_list`, `task_start`, `task_list`,
`task_status`, `task_output`, `task_kill`, `search_web`, `fetch_url`, `read_scratch`,
`remember`, `recall`, `vault`.

### Weak / fine-tuned local models

Some small models (and LoRA fine-tunes) ignore the tool-calling channel and print the
call into the answer instead:

```json
{"content": "Analyzing the current working directory.",
 "tool_calls": [{"name": "glob_list", "args": {"pattern": "."}}]}
```

Qwen-family models use their own text protocol for the same thing, which arrives as
content whenever the server does not parse the tags:

```
<tool_call>
{"name": "glob_list", "arguments": {"pattern": "*.py"}}
</tool_call>
```

langbot detects both, converts them into real tool calls, and executes them, logging a
warning each time (`tool_call_repair: model emitted N tool call(s) as text ...`). The
repair only fires when the message has no native tool calls and the tool name is actually
registered, so ordinary answers that merely discuss JSON are untouched. The same models
wrap plain answers (`{"content": "...", "tool_calls": []}`) and the distiller's fact array
in that envelope; both are unwrapped as well, so the user sees prose rather than JSON and
knowledge distillation still works. Leaked chat-template markup (`<tool_response>`,
`<|im_end|>`) is stripped from answers, and `<think>` reasoning blocks are shown as
Thought panels rather than as part of the answer. Set
`compat.repair_json_tool_calls` to `false` to see the raw output instead — a stream of
those warnings is a good signal to fix the model's chat template or prompt format.

## Security notes

- The agent can run **arbitrary shell commands** and read/write **any file**. Treat it as
  giving the model a shell on your machine. Run only in a sandbox/VM you control.
- The credential vault encrypts values with AES-256-GCM and restricts its files to
  `0600`. By default the master key is stored (file-protected) alongside the ciphertext,
  so encryption at rest primarily protects against other users on the host; set
  `LANGBOT_VAULT_PASSWORD` for password-wrapped key protection.

## Project layout

```
langbot.py              # agent loop, tools, memory, LangGraph wiring (entrypoint)
components/
  file_ops.py           # read/write/patch/batch_patch/git_diff file tools
  code_search.py        # find_in_files / read_many_files / glob_list
  tasks.py              # background task manager (start/list/status/output/kill)
  config.py             # optional config file (langbot.config.json) with default fallbacks
  tool_call_repair.py   # recovers tool calls from models that emit them as text
  scratch.py            # shared on-disk scratchpad + read_scratch paging
  memory_store.py       # embeddings + Chroma collection (store/recall, write lock)
  memory_worker.py      # background distillation queue (off the graph's critical path)
  routing.py            # agent routing, autonomy nudges, duplicate-answer guard
  web_tools.py          # search_web / fetch_url (scratchpad-backed)
  engines.py            # SearXNG engine adapter used by web_tools
  vault.py              # AES-256-GCM credential vault (the `vault` tool + env auto-load + redaction)
  input.py              # readline input UX used by the REPL
  console.py            # terminal UI helpers used by the REPL
  utils.py              # shared helpers (atomic JSON writes, truncation)
langbot.config.example.json  # template listing every setting and its default
CODE_REVIEW.md          # review of the initial commit with known issues + fixes
MEMORY_POLICY.md        # where persistent state may live (./memory/ only)
CONTRIBUTING.md         # how to contribute + where help is wanted
LICENSE                 # proprietary license
```

## Known issues

See [`CODE_REVIEW.md`](./CODE_REVIEW.md) for the original review. Several items have since
been addressed (console 3.12 import break, vault AES-GCM migration + `0600` perms, active
output redaction, `read_scratch` handling).

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the development
setup, ground rules, and the areas where help is most wanted (new tools, SearXNG engine
adapters, memory quality, local-model prompt tuning, terminal UX, sandboxing, docs and
tests). Opening a pull request means agreeing to the contribution terms in the license.

## License

Proprietary — Copyright (c) 2026 mcashdevel-ux, all rights reserved. See
[`LICENSE`](./LICENSE); use outside a written agreement with the copyright holder is not
permitted. Third-party dependencies remain under their own licenses.
