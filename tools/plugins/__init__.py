"""Plugin tool discoverer — auto-loads tools from sibling .py files.

Each plugin module (e.g. ``py_eval.py``) exports three module-level attributes:

    TOOLS: list[Callable]     — ``@tool``-decorated functions
    DESCRIPTIONS: dict[str, str]  — tool_name → one-sentence embedding description
    TRIGGERS: dict[str, str]      — tool_name → regex pattern for keyword routing

``discover_plugins()`` scans all non-``__init__`` modules and returns a flat
``(tools, descriptions, triggers)`` triple ready for wiring into ``langbot.py``
and ``components.tool_router.register()``.
"""

import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)


def discover_plugins():
    """Scan this package for plugin modules and return ``(tools, descriptions, triggers)``.

    Any ``.py`` file is imported; modules whose names start with ``_`` are skipped.
    A module is a valid plugin if it exports ``TOOLS``. ``DESCRIPTIONS`` and
    ``TRIGGERS`` are optional.

    Returns:
        ``(tools, descriptions, triggers, core_tools)`` where each is a merged
        list/dict/set from all discovered plugins.  ``core_tools`` is a set of
        tool names whose plugins declared ``CORE = ["tool_name"]``.
    """
    all_tools: list = []
    all_descriptions: dict[str, str] = {}
    all_triggers: dict[str, str] = {}

    for _, name, _ in pkgutil.iter_modules(__path__):
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except Exception:
            logger.warning("tools/plugins: failed to import %s (skipping)", name,
                           exc_info=True)
            continue

        mod_tools = getattr(mod, "TOOLS", [])
        if mod_tools:
            all_tools.extend(mod_tools)
            logger.debug("tools/plugins: loaded %d tool(s) from %s",
                         len(mod_tools), name)

        mod_descs = getattr(mod, "DESCRIPTIONS", {})
        if isinstance(mod_descs, dict):
            all_descriptions.update(mod_descs)

        mod_triggers = getattr(mod, "TRIGGERS", {})
        if isinstance(mod_triggers, dict):
            all_triggers.update(mod_triggers)

    # Collect CORE tool names from plugins that declare them.
    all_core: set[str] = set()
    for _, name, _ in pkgutil.iter_modules(__path__):
        if name.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{__name__}.{name}")
        except Exception:
            continue
        mod_core = getattr(mod, "CORE", [])
        if isinstance(mod_core, list):
            all_core.update(mod_core)

    return all_tools, all_descriptions, all_triggers, all_core
