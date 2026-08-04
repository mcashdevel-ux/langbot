import pytest
from tools.plugins.py_eval import py_eval


def test_py_eval_basic():
    # Test simple expression evaluation
    res = py_eval.invoke({"code": "3 * (10 + 5)"})
    assert "45" in res

    # Test multi-line stdout capture
    code = "x = 10\ny = 20\nprint(x + y)"
    res = py_eval.invoke({"code": code})
    assert "30" in res


def test_py_eval_blocked_builtins():
    # Attempting to call blocked builtins should fail
    res = py_eval.invoke({"code": "open('test.txt', 'r')"})
    assert "NameError" in res

    res = py_eval.invoke({"code": "__import__('os')"})
    assert "NameError" in res

    res = py_eval.invoke({"code": "eval('1+1')"})
    assert "NameError" in res

    res = py_eval.invoke({"code": "globals()"})
    assert "NameError" in res

    res = py_eval.invoke({"code": "locals()"})
    assert "NameError" in res


def test_py_eval_isolation():
    # Ensure variables defined in one execution do not leak to the next
    py_eval.invoke({"code": "my_secret_var = 12345\nprint(my_secret_var)"})
    res2 = py_eval.invoke({"code": "print(my_secret_var)"})
    # Since my_secret_var shouldn't exist, it should raise a NameError
    assert "NameError" in res2


def test_py_eval_timeout():
    # Ensure an infinite loop does not block but raises a TimeoutException
    res = py_eval.invoke({"code": "while True: pass"})
    assert "TimeoutException" in res or "timed out" in res


def test_py_eval_subclass_escape_attempt():
    # Verify that trying to walk class trees is blocked completely at compile-time by AST validator
    res = py_eval.invoke({"code": "().__class__.__bases__[0].__subclasses__()"})
    assert "py_eval error" in res
    assert "dunder" in res or "forbidden" in res

    # Verify that trying to use getattr with a dunder attribute string is blocked at runtime
    res2 = py_eval.invoke({"code": "getattr((), '__class__')"})
    assert "py_eval error" in res2
    assert "dunder" in res2 or "forbidden" in res2

    # Verify that format strings attempting dunder lookups are blocked
    res3 = py_eval.invoke({"code": "'{0.__class__}'.format(())"})
    assert "py_eval error" in res3
    assert "dunder" in res3 or "forbidden" in res3
