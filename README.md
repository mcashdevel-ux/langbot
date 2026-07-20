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
- **Web tools** (`web_tools.py`) — `search_web` and `fetch_url` (via Jina Reader) that
  save full payloads to an on-disk scratchpad and return short, context-cheap previews;
  `read_scratch` pages through the rest.
- **SearXNG engine adapter** (`engines.py`) — runs individual SearXNG engine modules
  directly (no SearXNG web app), supporting many search engines.
- **Conversation persistence** — LangGraph SQLite checkpointer when
  `langgraph-checkpoint-sqlite` is installed, else in-memory.

### Not currently wired in

`vault.py` (encrypted credential vault), `input.py` (readline UX), and `console.py`
(rich terminal UI) target a different plugin framework (`agent.register_tool`,
`agent._tool_observers`) and are **not imported by `langbot.py`**. They ship as standalone
modules; integrating them requires an adapter. See `CODE_REVIEW.md` for details and known
issues before relying on them.

## Requirements

- **Python 3.12+** (required — `console.py` uses f-string syntax introduced in 3.12; see
  `CODE_REVIEW.md` C1).
- A local OpenAI-compatible LLM server (default `http://127.0.0.1:8080/v1`).

### Python dependencies

There is no dependency manifest yet. Install the packages the code imports:

```bash
pip install \
  langchain-core langchain-openai langchain-huggingface \
  langgraph langgraph-checkpoint-sqlite \
  chromadb sentence-transformers \
  requests httpx colorama rich
```

`engines.py` additionally needs the SearXNG source on disk. Place it at one of
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

### Available tools

`execute_shell_command`, `read_any_file`, `write_any_file`, `search_web`, `fetch_url`,
`read_scratch`, `remember`, `recall`.

## Security notes

- The agent can run **arbitrary shell commands** and read/write **any file**. Treat it as
  giving the model a shell on your machine. Run only in a sandbox/VM you control.
- The credential vault (`vault.py`), if you integrate it, currently stores its master key
  in plaintext next to the ciphertext by default and its output-redaction is a no-op — see
  `CODE_REVIEW.md` (C2/C3) before using it to hold real secrets.

## Project layout

```
langbot.py    # agent loop, tools, memory, LangGraph wiring (entrypoint)
web_tools.py  # search_web / fetch_url / read_scratch (scratchpad-backed)
engines.py    # SearXNG engine adapter used by web_tools
vault.py      # encrypted credential vault (standalone; not wired in)
input.py      # readline input UX (standalone; not wired in)
console.py    # terminal UI helpers (standalone; not wired in)
CODE_REVIEW.md# review of the initial commit with known issues + fixes
```

## Known issues

See [`CODE_REVIEW.md`](./CODE_REVIEW.md) for a full list, including the Python 3.12
requirement, the vault crypto/redaction issues, and the `read_scratch` non-ASCII bug.
