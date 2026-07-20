# langbot

A terminal-based, tool-using AI agent built on **LangGraph** + **LangChain**. It talks to
a local OpenAI-compatible LLM endpoint, can run shell commands, read/write files, search
the web (via SearXNG engines) and fetch pages, and keeps a persistent, searchable
long-term memory (Chroma + sentence-transformers).

> ⚠️ **Security warning:** the agent has **unrestricted shell, file, and web access** and
> is prompted to act without asking. Only run it in a trusted, single-user, sandboxed
> environment. See "Security notes" below.

## Features

- **Autonomous agent loop** (`langbot.py`) — LangGraph `StateGraph` with an agent node, a
  `ToolNode`, and an automatic knowledge-distillation node that extracts durable facts
  from each exchange into long-term memory.
- **Long-term memory** — Chroma vector store (`agent_memory_chroma/`) with
  `remember` / `recall` tools plus automatic distillation.
- **Web tools** (`components/web_tools.py`) — `search_web` and `fetch_url` (via Jina Reader)
  that save full payloads to an on-disk scratchpad and return short, context-cheap previews;
  `read_scratch` pages through the rest.
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

There is no runtime dependency manifest yet (dev/test deps are in `requirements-dev.txt`).
Install the packages the code imports:

```bash
pip install \
  langchain-core langchain-openai langchain-huggingface \
  langgraph langgraph-checkpoint-sqlite \
  chromadb sentence-transformers \
  cryptography requests httpx colorama rich
```

`components/engines.py` additionally needs the SearXNG source on disk. Place it at one of
`./searxng-src`, `~/searxng-src`, or `/usr/local/searxng/searxng-src`, or let the module
clone it automatically on first use:

```bash
git clone --depth 1 https://github.com/searxng/searxng ~/searxng-src
```

## Configuration

Edit the constants at the top of `langbot.py`:

| Constant | Default | Purpose |
|----------|---------|---------|
| `BASE_URL` | `http://127.0.0.1:8080/v1` | OpenAI-compatible LLM endpoint |
| `LLM_MODEL` | `local-model` | Model name sent to the endpoint |
| `SQLITE_DB_PATH` | `./agent_checkpoints.db` | Conversation checkpoint DB |
| `CHROMA_PERSIST_DIR` | `./agent_memory_chroma` | Long-term memory store |

Environment variables:

- `SEARXNG_SETTINGS_PATH` — path to a SearXNG `settings.yml` (defaults to
  `/etc/searxng/settings.yml`, then the source's bundled settings).
- `AGENT_SCRATCH_DIR` — where web scratch files are written (default `/tmp/agent_scratch`;
  note `/tmp` may be cleared between reboots).
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

### Tests

The `components/` modules have a unit-test suite (the heavy LLM deps and a live LLM server
are not required):

```bash
pip install -r requirements-dev.txt
python -m pytest
```

### Available tools

`execute_shell_command`, `read_any_file`, `write_any_file`, `search_web`, `fetch_url`,
`read_scratch`, `remember`, `recall`, `vault`.

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
  web_tools.py          # search_web / fetch_url / read_scratch (scratchpad-backed)
  engines.py            # SearXNG engine adapter used by web_tools
  vault.py              # AES-256-GCM credential vault (the `vault` tool + env auto-load + redaction)
  input.py              # readline input UX used by the REPL
  console.py            # terminal UI helpers used by the REPL
  utils.py              # shared helpers (atomic JSON writes, truncation)
CODE_REVIEW.md          # review of the initial commit with known issues + fixes
```

## Known issues

See [`CODE_REVIEW.md`](./CODE_REVIEW.md) for the original review. Several items have since
been addressed (console 3.12 import break, vault AES-GCM migration + `0600` perms, active
output redaction, `read_scratch` handling).
