"""Background knowledge distillation + embedding.

Distillation (one LLM round-trip) and the embedding/ChromaDB writes it produces
used to run synchronously in the graph's last node, blocking the REPL from
returning control after every tool-using turn. This module moves that work onto
a single daemon consumer thread fed by a bounded queue: the graph node only
formats a job and enqueues it (never blocks, drops with a warning when the queue
is full), and facts land in memory shortly after the answer is shown.

Jobs carry a plain snapshot of the turn rather than graph state, so this module
has no LangGraph coupling and is testable in isolation.
"""

import json
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from .config import config
from .tool_call_repair import strip_reasoning, unwrap_content
from .utils import strip_code_fences

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = config.get("memory.worker_queue_size", 50)
# Jobs merged into one embedding/write batch. Distillation is still one LLM call
# per job; only the store is batched.
MAX_BATCH = config.get("memory.worker_batch_size", 5)
SHUTDOWN_TIMEOUT = config.get("memory.worker_shutdown_timeout", 10.0)


@dataclass
class DistillJob:
    user_text: str
    ai_text: str
    tool_context: str          # pre-formatted "[tool]: result" lines
    enqueued_at: float = field(default_factory=time.time)


def _distillation_prompt(job: DistillJob) -> str:
    return f"""
You are a knowledge extraction module. Look at the following user request, the tool
results that were actually returned this turn, and the assistant's final response.
Extract only factual information that is GROUNDED IN THE TOOL RESULTS — do not infer
or store anything the assistant merely described doing without evidence in the tool output.
Useful facts include:
- User preferences (e.g., "the user prefers DuckDuckGo")
- Confirmed facts from tool output (e.g., "project located at ~/code/myapp")
- Decisions or conclusions that are supported by evidence
- Context helpful for future interactions

User request: {job.user_text}
Tool results this turn:
{job.tool_context}
Assistant response: {job.ai_text}

Return ONLY a JSON array of strings, each a standalone factual statement grounded in
the tool results above. If nothing is clearly supported by evidence, return [].
Do not include explanations, markdown, or extra text.
/no_think
"""  # /no_think disables Qwen3-style reasoning blocks; harmless to other models.


_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$")
# Keys a model may hide the fact list under when it wraps the array in an object.
_LIST_KEYS = ("facts", "memories", "content", "items", "results", "data", "knowledge")
_ARRAY_RE = re.compile(r"\[.*?\]", re.DOTALL)


def _strings(items) -> "list[str] | None":
    """Coerce a list of facts to clean strings, accepting {"fact": "..."} entries."""
    out = []
    for item in items:
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict):
            text = next(
                (item[k] for k in ("fact", "text", "statement", "content")
                 if isinstance(item.get(k), str)),
                None,
            )
            if text is None:
                return None
        else:
            return None
        text = text.strip()
        if text:
            out.append(text)
    return out


def parse_facts(raw, _depth: int = 0) -> "list[str] | None":
    """Read a fact list out of the distiller's output, or None if unreadable.

    The prompt asks for a bare JSON array of strings, but small local models
    return it wrapped in a chat envelope, nested under an object key, as objects
    instead of strings, embedded in prose, or as a Markdown bullet list. Since a
    miss silently loses the whole turn's knowledge, every one of those shapes is
    accepted. An empty list is a valid answer ("nothing worth storing") and is
    distinct from None ("could not read this at all").
    """
    if not isinstance(raw, str) or _depth > 3:
        return None
    text = strip_code_fences(strip_reasoning(raw).strip())
    # A chat envelope may itself contain fences/JSON, hence the recursion.
    inner = unwrap_content(text)
    if inner is not None:
        return parse_facts(inner, _depth + 1)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        return _strings(parsed)
    if isinstance(parsed, dict):
        for key in _LIST_KEYS:
            value = parsed.get(key)
            if isinstance(value, list):
                return _strings(value)
            if isinstance(value, str):
                return parse_facts(value, _depth + 1)
        # Any other single list value, e.g. {"extracted_facts": [...]}.
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if len(lists) == 1:
            return _strings(lists[0])
        return None
    if isinstance(parsed, str):
        return parse_facts(parsed, _depth + 1)

    # Not JSON as a whole: an array embedded in prose, else a bullet list.
    match = _ARRAY_RE.search(text)
    if match:
        try:
            embedded = json.loads(match.group(0))
        except json.JSONDecodeError:
            embedded = None
        if isinstance(embedded, list):
            return _strings(embedded)
    bullets = [m.group(1) for m in (_BULLET_RE.match(ln) for ln in text.splitlines()) if m]
    if bullets:
        logger.debug("memory_worker: read %d fact(s) from a bullet list", len(bullets))
        return bullets
    return None


class MemoryWorker:
    """Single-consumer queue draining distillation jobs on its own thread."""

    def __init__(self, llm, store_fn=None, max_queue_size: int = MAX_QUEUE_SIZE):
        self._llm = llm
        self._store_fn = store_fn
        self._q: "queue.Queue[DistillJob]" = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True, name="memory-worker")
        self._thread.start()

    # ── Producer side (called from the graph thread) ──

    def enqueue(self, job: DistillJob) -> bool:
        """Non-blocking. Returns False (and logs) when the queue is full, so a
        slow or stuck worker can never block the calling graph node."""
        try:
            self._q.put_nowait(job)
            return True
        except queue.Full:
            self._dropped += 1
            logger.warning(
                "memory_worker: queue full, dropping distillation job "
                "(%d dropped so far)", self._dropped
            )
            return False

    def qsize(self) -> int:
        return self._q.qsize()

    def dropped_count(self) -> int:
        return self._dropped

    # ── Consumer side (worker thread) ──

    def _run(self):
        while not self._stop.is_set():
            try:
                job = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [job]
            try:
                while len(batch) < MAX_BATCH:
                    batch.append(self._q.get_nowait())
            except queue.Empty:
                pass
            try:
                self._process(batch)
            except Exception:
                logger.exception("memory_worker: batch failed, skipping")
            finally:
                for _ in batch:
                    self._q.task_done()

    def _process(self, batch: "list[DistillJob]") -> None:
        facts = []
        for job in batch:
            try:
                facts.extend(self._distill(job))
            except Exception:
                logger.exception("memory_worker: job failed, skipping")
        if facts:
            self._store(facts)

    def _store(self, facts: "list[str]") -> None:
        store_fn = self._store_fn
        if store_fn is None:
            from .memory_store import store_memories_batch

            store_fn = store_memories_batch
        store_fn(facts)

    def _distill(self, job: DistillJob) -> "list[str]":
        raw = self._llm.invoke(_distillation_prompt(job)).content
        facts = parse_facts(raw)
        if facts is None:
            # Log the output itself: the shape a given model emits is the only
            # way to tell a prompt/template problem from a parser gap.
            logger.warning(
                "memory_worker: distillation skipped, could not read a fact list from "
                "model output: %r",
                (raw if isinstance(raw, str) else str(raw))[:300],
            )
            return []
        return facts

    # ── Lifecycle ──

    def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT) -> None:
        """Drain remaining jobs (bounded wait), then stop the thread.

        The bound matters more than completeness: losing the last turn's facts on
        a slow shutdown beats making Ctrl+D unresponsive behind a hung LLM call.
        """
        deadline = time.time() + timeout
        while not self._q.empty() and time.time() < deadline:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=2.0)
        remaining = self._q.qsize()
        if remaining:
            logger.warning("memory_worker: shutdown with %d job(s) still queued", remaining)
