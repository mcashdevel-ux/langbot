"""KernelContext — dependency injection for tools (Track G of docs/neo-port-plan.md).

Ported from neo's ``neo/core/context.py``: a per-agent (per graph run) dataclass
that replaces module-level mutable globals, so two concurrent sessions (or swarm
peers) can run in one process without sharing mutable state.

langbot's graph is LangGraph-based, so the injection mechanism differs from neo's
explicit ``ctx`` parameter: langgraph 1.2.11 auto-injects a ``config:
RunnableConfig`` param into any tool (or node) that declares it, strips it from the
OpenAI schema sent to the model, and carries arbitrary ``configurable`` keys.  A
``KernelContext`` is stashed under ``config["configurable"]["kernel_context"]`` by
``tools_node`` before the ToolNode runs; tools resolve it from there, falling back
to the process-wide singletons when absent (so direct ``.func(...)`` test calls
and the REPL's slash handlers keep working unchanged).

What lives in the context (per plan):
- ``tasks`` — the background-task manager (per-session isolation of task lists).
- ``reflex_store`` — procedural reflex rules (shared store, per-run handle).
- ``journal`` — the current session's durable journal (Track B).
- ``memory_worker`` — process-wide singleton by design (lock-protected); kept
  as the default so multi-session memory isolation stays a documented follow-up.

- ``pending_confirm`` / ``pending_confirm_tool`` / ``pending_approved`` — the
  Track F confirmation-gate state (per-session: two sessions must not approve
  each other's pending calls).
- ``bound_llms`` / ``schema_tokens`` — tool-binding caches (per-session: the
  cache is keyed by tool-set names, so sharing it is harmless, but per-run keeps
  the isolation story uniform).
- ``last_prompt`` — prompt-accounting state (per-session: two interleaved
  runs must not corrupt each other's cache-reuse measurement).

Process-wide singletons that stay module-level (documented in the plan as
follow-ups if multi-session isolation is ever needed): the vault, the memory
store/embeddings, the warmup, the distill/summarize LLMs.
.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class KernelContext:
    """Per-graph-run dependency bundle injected into tools via ``config``.


    Every field defaults to ``None`` — a context built by ``tools_node`` fills in
    what it can from the graph config; consumers fall back to the module-level
    singletons when a field is ``None``, so the context is purely additive and a
    revert is mechanical.
    """

    thread_id: Optional[str] = None
    """The LangGraph thread_id for this run (from ``config["configurable"]``)."""

    tasks: Any = None
    """Background-task manager (``components.tasks.BackgroundTaskManager``)."""

    reflex_store: Any = None
    """Procedural reflex store (``components.reflex.ReflexStore``)."""

    journal: Any = None
    """The current session's durable journal (``components.journal.Journal``)."""

    memory_worker: Any = None
    """Background memory distiller (``components.memory_worker.MemoryWorker``)."""

    # Track F confirmation-gate state — per-session so two sessions can't approve
    # each other's pending calls.
    pending_confirm: Optional[str] = None
    pending_confirm_tool: Optional[str] = None
    pending_approved: bool = False

    # Tool-binding caches (keyed by tool-set names; per-run for a uniform story).
    bound_llms: Dict[Any, Any] = field(default_factory=dict)

    schema_tokens: Dict[Any, int] = field(default_factory=dict)


    # Prompt-accounting state (per-session: interleaved runs must not corrupt
    # each other's cache-reuse measurement).
    last_prompt: list = field(default_factory=list)



def context_from_config(config: Optional[Dict[str, Any]] = None) -> KernelContext:
    """Build a ``KernelContext`` from a graph ``config`` (or empty if none)."""
    ctx = KernelContext()
    if config:
        configurable = config.get("configurable") or {}
        ctx.thread_id = configurable.get("thread_id")
        existing = configurable.get("kernel_context")
        if isinstance(existing, KernelContext):
            return existing
    return ctx


def stash_context(config: Dict[str, Any], ctx: KernelContext) -> None:
    """Stash ``ctx`` into ``config["configurable"]["kernel_context"]`` in place.


    ``config`` may be a plain dict (as passed to graph nodes); ``configurable``
    is created if missing.  The graph propagates ``configurable`` to every tool
    call unchanged, so tools resolve the same context for the whole run."""
    configurable = config.setdefault("configurable", {})
    configurable["kernel_context"] = ctx