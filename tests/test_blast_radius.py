import pytest
from langbot import execute_shell_command


def test_execute_shell_command_safe():
    # Ordinary safe commands should not have warning prefix
    res = execute_shell_command.invoke({"command": "echo 'hello world'"})
    assert "destructive" not in res
    assert "hello world" in res


def test_execute_shell_command_destructive():
    # Destructive commands should be prefixed with a warning marker
    res = execute_shell_command.invoke({"command": "rm -rf /some/dummy/path"})
    assert "destructive command detected" in res

    res2 = execute_shell_command.invoke({"command": "git push origin --force"})
    assert "destructive command detected" in res2
