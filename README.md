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
  `/health` shows its queue depth and dropped-job count. The embedding model and
  vector store are warmed on a background thread (`components/warmup.py`) so the
  prompt appears immediately; `/health` reports their state.
- **Long-term memory** — Chroma vector store (`agent_memory_chroma/`) with
  `remember` / `recall` tools plus automatic distillation, all behind
  `components/memory_store.py`. Memory is only ever read when the model asks for it
  (`recall`) or you do (`/knowledge`) — nothing is injected into every turn.
  Retrieval is not a bare k-NN: hits below `memory.min_similarity` are dropped
  instead of being passed off as facts, candidates are over-fetched and then
  narrowed by MMR so `n` results are `n` *distinct* facts, a lexical leg finds the
  paths / env-var names / ports / error codes that sentence embeddings miss, and
  writes skip duplicates so one repeated fact cannot own every result slot.
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
| `paths` | `checkpoint_db`, `chroma_dir`, `scratch_dir`, `tasks_dir`, `vault_dir`, `log_file` | keep inside `./memory/` per `MEMORY_POLICY.md` |
| `logging` | `level`, `console`, `max_bytes`, `backup_count` | log destination and verbosity (see below) |
| `memory` | `collection_name`, `embedding_model`, `embedding_device`, `worker_queue_size`, `worker_batch_size`, `worker_shutdown_timeout`, `max_facts_per_turn` | background distiller + vector store |
| `memory` (search) | `min_similarity`, `recall_overfetch`, `mmr_lambda`, `dedup_similarity`, `dedup_token_overlap`, `lexical_search`, `max_tags`, `auto_tags` | retrieval precision and tags (see below) |
| `memory` (distiller) | `non_distillable_tools` | tools whose output is machine state, so the turn skips its distillation call |
| `distill` | `tiers`, `cooldown_seconds`, `temperature`, `timeout`, `reserve_output_tokens` | hosted models tried for distillation before the local one (see below) |
| `context` | `budget_tokens`, `reserve_tokens`, `compact_at`, `keep_last_messages`, `summary_max_chars`, `chars_per_token` | history compaction (see below) |
| `tools` | `read_inline_chars`, `grep_inline_lines`, `manyfiles_inline_chars`, `max_output_chars` | how much tool output goes inline; the full result always reaches scratch |
| `tools` (binding) | `dynamic_binding`, `core` | which tool schemas are sent each step (see below) |
| `web` | `search_snippet_chars`, `search_max_results`, `fetch_inline_chars`, `jina_timeout`, `jina_retry_on_429`, `searxng_settings_path`, `searxng_source_dir` | search/fetch behaviour |
| `routing` | `max_nudges_per_turn`, `recursion_limit` | nudges (not tool rounds) allowed per turn; graph steps per turn (two per tool round) |
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
- `LANGBOT_LOG_FILE` — where log records are written (default `./memory/langbot.log`).
- `LANGBOT_LOG_LEVEL` — log verbosity (default `WARNING`).
- `LANGBOT_LOG_CONSOLE` — set to `1` to also stream log records to stderr.
- `GROQ_API_KEY` — enables the hosted distillation tiers (see below). Without it,
  distillation runs on the local model exactly as before. The vault loads it into the
  environment automatically once stored (`vault store GROQ_API_KEY`).

### Logs

The terminal is reserved for the UI: diagnostics from the agent and its background
threads (distillation warnings, tool-call repairs, routing guards) go to a rotating log
file — `./memory/langbot.log` by default — instead of being printed between the REPL's
panels and prompt. `/config` and `/health` show the active log file.

```bash
tail -f memory/langbot.log                 # watch diagnostics next to the REPL
LANGBOT_LOG_LEVEL=DEBUG python langbot.py  # more detail (per-chunk graph tracing)
LANGBOT_LOG_CONSOLE=1 python langbot.py    # mirror records to stderr as well
```

### Context budget

The checkpointer replays the whole thread on every step, so history is capped by
tokens rather than left to grow until the server truncates it (which silently drops
the system prompt) or rejects the turn. Before each agent step, if the thread plus its
summary exceeds `compact_at` (0.7) of `budget_tokens - reserve_tokens`, everything
older than the last `keep_last_messages` messages is folded into a rolling summary in
one cheap LLM call, and those messages are deleted from the checkpoint as well as from
the prompt. The split never separates a tool result from the message that requested it,
and the details survive regardless: tool output is already on disk under a `scratch:` id.
Counting uses `tiktoken` when installed and a `chars_per_token` estimate otherwise.

Set `budget_tokens` to the context length the server is actually serving (`llama-server
-c`), not the model's theoretical maximum.

### Serve tool calls, don't repair them

`components/tool_call_repair.py` exists because a small model asked for a tool by
*printing* the call into its answer, where nothing executes it. That is a decoding
problem, and the server can prevent it outright: start `llama-server` with `--jinja`, so
the model's own chat template is applied and tool calls are generated under a grammar
constraint and parsed back as real `tool_calls`.

`/health` reports how often the repair layer had to step in:

```
tool-call repairs   0 recovered, 0 answers cleaned
```

A non-zero `recovered` count means the server is still emitting tool calls as text —
check `--jinja` first. Zero across long sessions is the evidence that the repair layer,
the nudges, and the code-block patterns are belt-and-braces rather than load-bearing.

### Tool binding

Every bound tool costs its JSON schema in the prompt on every step, and a small model
picks worse from a longer menu. So only `tools.core` is always bound; the rest are added
for a turn when the conversation mentions what they do, or once they have been used in
that turn (see `components/tool_router.py`). Set `tools.dynamic_binding` to `false` to
bind all twenty every step.

### Memory search

Long-term memory is read **on demand only** — when the model calls `recall`, or when you
run `/knowledge <query>`. Nothing is injected into every turn, so an unrelated question
never arrives carrying unrelated "facts".

What a lookup does, and the knobs for each part:

| Setting | Default | Effect |
|---------|---------|--------|
| `min_similarity` | `0.3` | cosine similarity a hit must reach to count as relevant; below it, `recall` answers "nothing above the relevance threshold" rather than returning the nearest rows regardless of distance |
| `recall_overfetch` | `4` | candidates fetched per requested result, before dedup and MMR narrow them |
| `mmr_lambda` | `0.7` | relevance-vs-diversity tradeoff when picking the final `n` (`1.0` = pure relevance) |
| `lexical_search` | `true` | run a literal-token leg beside the dense one and fuse the two by reciprocal rank; this is what finds `~/code/myapp`, `OPENAI_API_KEY` or `port 8080`, which sentence embeddings rank poorly |
| `dedup_similarity` | `0.95` | how close a new fact must be to an existing one to be treated as a duplicate and not stored again |
| `dedup_token_overlap` | `0.6` | vocabulary overlap also required for that duplicate verdict; on top of it the two must name the same identifiers, so `port 8080` never absorbs `port 8081` |
| `max_tags` | `5` | tags kept per fact |
| `auto_tags` | `true` | give untagged facts deterministic fallback tags (`preference`, `filesystem`, `credentials`, `web`) based on what they visibly contain |

`/knowledge <query>` prints each hit's score, matched legs, tags, source and timestamp —
the view to use when tuning `min_similarity` for your own store.

### Distillation tiers

Distillation is the one LLM call that is both off the critical path and cheap — a
truncated turn summary in, a short JSON array of facts out — and it is where a small
local model is weakest, since an unparseable answer silently loses the turn's
knowledge. So it runs on its own fallback chain instead of the agent's model, tried
in order, with the local model always last:

| Tier | Free-tier limits (RPM / RPD / TPM / TPD) | Why here |
|------|------------------------------------------|----------|
| `openai/gpt-oss-120b` | 30 / 1K / 8K / 200K | best remaining free instruction-follower |
| `qwen/qwen3.6-27b` | 30 / 1K / 8K / 200K | native to the prompt's `/no_think` hint |
| `openai/gpt-oss-20b` | 30 / 1K / 8K / 200K (unconfirmed, see note) | weakest, last before local |
| local model | — | always available, no network, no quota |

`llama-3.3-70b-versatile` and `llama-3.1-8b-instant` previously held the first and
last slots; Groq deprecated both on June 17, 2026 with a shutdown date of August 16,
2026 (see [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations)),
and its own migration guidance points to the three models above. `openai/gpt-oss-20b`'s
TPD is confirmed from a live 429 response; its RPM/RPD/TPM are carried over from
`gpt-oss-120b` as an estimate pending confirmation against
[console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits).

Quotas are per model *and* per organization, so the chain multiplies the available
budget rather than re-hitting one bucket. Limits are respected *before* a call, not
discovered by failing one: each tier keeps sliding 60-second and 24-hour windows of
the requests and tokens it has spent, and Groq's own
[rate-limit headers](https://console.groq.com/docs/rate-limits)
(`x-ratelimit-remaining-tokens` is per minute, `x-ratelimit-remaining-requests` is
per **day**) override that local accounting on every response. A tier that would not
fit is skipped silently; one that fails, gets a 429, or returns output the fact parser
cannot read is put on a cooldown — `retry-after` when the server sends one, otherwise
`distill.cooldown_seconds`. `/health` and `/config` print the chain with each tier's
current state.

A tier without its `api_key_env` set is skipped, so the default chain is inert until
`GROQ_API_KEY` exists. Setting `"distill": {"tiers": []}` disables hosted distillation
entirely and leaves the local model doing the work.

Note what a hosted tier implies: the distillation prompt — the user request, the
assistant's reply, and the turn's tool output, each truncated — leaves the machine.
Everything else (the agent's own turns, the memory store, embeddings) stays local. Keep
`tiers` empty if that trade is not acceptable for your data.

### Tags

Every fact can carry short category tags. They come from three places: an explicit
`tags` argument to the `remember` tool, trailing `#tag` tokens on `/save`
(`/save prefers dark mode #preference #ui`), and the background distiller, which asks
the model for `{"fact", "tags"}` objects; deterministic fallback rules cover facts
that arrive untagged.

Tags are stored two ways: comma-joined in the row's metadata (for display) and as
`#tag` tokens inside the lexical document — so tag search is just the existing
lexical leg. A plain query whose words match a tag finds tagged facts, and a `#tag`
token makes it exact: `recall("#preference")` or `/knowledge #preference` returns only
facts carrying that tag. Tags are never embedded, so they cannot distort semantic
similarity, and they are ignored by dedup — a duplicate write merges its new tags
into the existing fact instead. Rows written before tags existed keep working,
simply untagged.

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
warning to the log file each time (`tool_call_repair: model emitted N tool call(s) as
text ...`). The
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
  logging_setup.py      # routes log records to ./memory/langbot.log, off the REPL
  tool_call_repair.py   # recovers tool calls from models that emit them as text
  scratch.py            # shared on-disk scratchpad + read_scratch paging
  memory_store.py       # embeddings + Chroma collection (store/recall, write lock)
  memory_worker.py      # background distillation queue (off the graph's critical path)
  fallback_llm.py       # tiered distillation LLM: hosted models (rate-limit aware) then local
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
