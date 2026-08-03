"""Sandboxed Python evaluator — run small snippets without a subprocess.

Only a allowlisted subset of builtins is available: no ``__import__``, ``open``,
``exec``, ``eval``, ``compile``, ``breakpoint``, or ``input``.  The expression is
compiled in ``'eval'`` mode (or ``'exec'`` for multi-line) and executed with a
restricted globals dict.  Large results are saved to the scratchpad.
"""

import ast
import logging

from langchain_core.tools import tool

from components.scratch import save_to_scratch

logger = logging.getLogger(__name__)

# Builtins the sandbox can use.  ``__import__``, ``open``, ``exec``, ``eval``,
# ``compile``, ``breakpoint``, and ``input`` are deliberately excluded.
_SAFE_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "complex", "dict", "dir", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "getattr", "globals",
    "hasattr", "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "locals", "map", "max", "memoryview", "min",
    "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "super", "tuple", "type", "vars", "zip",
    "__build_class__",
}

# Additional names available in the sandbox for convenience.
_EXTRA_NAMES = {
    "json": __import__("json"),
    "re": __import__("re"),
    "math": __import__("math"),
    "datetime": __import__("datetime"),
    "itertools": __import__("itertools"),
    "collections": __import__("collections"),
    "functools": __import__("functools"),
    "statistics": __import__("statistics"),
}


def _make_sandbox_globals() -> dict:
    """Build a restricted globals dict with safe builtins + extras."""
    import builtins as _builtins
    safe = {}
    for name in _SAFE_BUILTINS:
        if hasattr(_builtins, name):
            safe[name] = getattr(_builtins, name)
    safe.update(_EXTRA_NAMES)
    # Prevent the sandbox from escaping via __builtins__
    safe["__builtins__"] = safe
    return safe


_SANDBOX_GLOBALS = _make_sandbox_globals()

INLINE_CHARS = 2000


@tool
def py_eval(code: str) -> str:
    """Evaluate a Python expression or statement in a sandboxed environment.

    The sandbox has access to a safe subset of builtins plus json, re, math,
    datetime, itertools, collections, functools, and statistics.  Use this for
    calculations, data transformations, JSON parsing, string manipulation, and
    quick arithmetic — anything that doesn't need a shell subprocess.

    Args:
        code: The Python code to evaluate.  Single-line expressions are evaluated
              with ``eval()`` and their result is returned directly.  Multi-line
              blocks are executed with ``exec()`` and their stdout is captured.

    Returns:
        The result as a string.  Large results (>{INLINE_CHARS} chars) are saved to
        the scratchpad and a reference is returned instead.
    """
    if not code or not code.strip():
        return "(empty)"
    code = code.strip()

    # Detect multi-line vs single expression
    lines = code.split("\n")
    is_multi = len(lines) > 1

    try:
        if is_multi:
            # Capture stdout during exec
            import io
            buf = io.StringIO()
            _SANDBOX_GLOBALS["print"] = lambda *a, **kw: print(*a, **kw, file=buf)
            try:
                exec(compile(code, "<py_eval>", "exec"), _SANDBOX_GLOBALS)
                result = buf.getvalue()
            finally:
                _SANDBOX_GLOBALS["print"] = print
            if not result.strip():
                result = "(executed, no output)"
        else:
            obj = eval(compile(code, "<py_eval>", "eval"), _SANDBOX_GLOBALS)
            result = repr(obj) if not isinstance(obj, str) else obj
    except Exception as e:
        return f"py_eval error: {type(e).__name__}: {e}"

    if len(result) > INLINE_CHARS:
        sid = save_to_scratch(result, prefix="pyeval")
        return (f"{result[:INLINE_CHARS]}\n\n... [{len(result):,} total chars] "
                f"(full output at scratch:{sid})")
    return result


# Plugin contract — discovered by tools.plugins.discover_plugins()
TOOLS = [py_eval]

DESCRIPTIONS = {
    "py_eval": (
        "Evaluate a Python expression in a sandboxed environment for calculations, "
        "data transformations, JSON parsing, and quick arithmetic."
    ),
}

TRIGGERS = {
    "py_eval": (
        r"\b(calculate|compute|evaluate|run python|python code|arithmetic|count|sum|average|"
        r"convert|parse|transform|filter|sort|format|math|statistics|round)\b"
    ),
}
