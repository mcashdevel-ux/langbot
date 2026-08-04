"""Sandboxed Python evaluator — run small snippets without a subprocess.

Only a allowlisted subset of builtins is available: no ``__import__``, ``open``,
``exec``, ``eval``, ``compile``, ``breakpoint``, or ``input``.  The expression is
compiled in ``'eval'`` mode (or ``'exec'`` for multi-line/statements) and executed
with a restricted globals dict.  Large results are saved to the scratchpad.
"""

import ast
import logging
import signal
from langchain_core.tools import tool

from components.scratch import save_to_scratch
from components.config import config

logger = logging.getLogger(__name__)

# Builtins the sandbox can use.  ``__import__``, ``open``, ``exec``, ``eval``,
# ``compile``, ``breakpoint``, and ``input`` are deliberately excluded.
# ``globals``, ``locals``, and ``vars`` are also removed to prevent traversal escapes.
# ``getattr`` and ``hasattr`` are overridden with custom safe implementations.
_SAFE_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "complex", "dict", "dir", "divmod", "enumerate",
    "filter", "float", "format", "frozenset",
    "hash", "hex", "id", "int", "isinstance", "issubclass",
    "iter", "len", "list", "map", "max", "memoryview", "min",
    "next", "object", "oct", "ord", "pow", "print", "property", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "super", "tuple", "type", "zip",
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


def safe_getattr(obj, name, default=None):
    """Restricted getattr that forbids access to dunder attributes."""
    if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
        raise ValueError(f"Access to dunder attribute '{name}' is forbidden")
    return getattr(obj, name, default)


def safe_hasattr(obj, name):
    """Restricted hasattr that forbids check of dunder attributes."""
    if isinstance(name, str) and name.startswith("__") and name.endswith("__"):
        raise ValueError(f"Access to dunder attribute '{name}' is forbidden")
    return hasattr(obj, name)


def _make_sandbox_globals() -> dict:
    """Build a restricted globals dict with safe builtins + extras."""
    import builtins as _builtins
    safe = {}
    for name in _SAFE_BUILTINS:
        if hasattr(_builtins, name):
            safe[name] = getattr(_builtins, name)
    safe["getattr"] = safe_getattr
    safe["hasattr"] = safe_hasattr
    safe.update(_EXTRA_NAMES)
    # Prevent the sandbox from escaping via __builtins__
    safe["__builtins__"] = safe
    return safe


class SafetyValidator(ast.NodeVisitor):
    """AST validator that inspects compiled trees for illegal dunder traversal."""

    def visit_Attribute(self, node):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            raise ValueError(f"Access to dunder attribute '{node.attr}' is forbidden")
        self.generic_visit(node)

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            # Block dunder lookups inside format strings, e.g. "{0.__class__}"
            if "{" in node.value and "}" in node.value and "__" in node.value:
                raise ValueError("Access to dunder attributes in format strings is forbidden")
        self.generic_visit(node)


class EvaluationTimeout(Exception):
    """Exception raised when code evaluation exceeds the specified timeout."""

    pass


def _timeout_handler(signum, frame):
    raise EvaluationTimeout("Evaluation timed out (loop or execution exceeded time limit)")


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

    # Pre-flight compile and safety check the AST
    try:
        tree = ast.parse(code)
        SafetyValidator().visit(tree)
    except Exception as e:
        return f"py_eval error: {type(e).__name__}: {e}"

    # Auto-detect eval (expression) vs exec (statements) based on AST structure
    try:
        compile(code, "<py_eval>", "eval")
        is_expression = True
    except SyntaxError:
        is_expression = False

    # Build a fresh, isolated copy of globals for this execution to prevent state leakage
    sandbox_globals = _make_sandbox_globals()

    # Configure execution timeout
    timeout_seconds = config.get("tools.py_eval_timeout", 2)
    original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        if not is_expression:
            # Capture stdout during exec
            import io
            buf = io.StringIO()
            sandbox_globals["print"] = lambda *a, **kw: print(*a, **kw, file=buf)
            try:
                exec(compile(code, "<py_eval>", "exec"), sandbox_globals)
                result = buf.getvalue()
            finally:
                # Restore standard print function
                sandbox_globals["print"] = print
            if not result.strip():
                result = "(executed, no output)"
        else:
            obj = eval(compile(code, "<py_eval>", "eval"), sandbox_globals)
            result = repr(obj) if not isinstance(obj, str) else obj
    except EvaluationTimeout as e:
        return f"py_eval error: TimeoutException: {e}"
    except Exception as e:
        return f"py_eval error: {type(e).__name__}: {e}"
    finally:
        # Cancel the alarm and restore original signal handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)

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

# Tool names from this plugin that should always be bound (added to tools.core).
CORE = []
