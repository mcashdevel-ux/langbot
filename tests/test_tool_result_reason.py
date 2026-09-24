"""Tests for the tool-result panel's intent subtitle.

A tool result is easier to read in the context of why it was fetched, so the
reasoning that preceded the call is shown as the panel's subtitle. That reasoning
is deliberately *not* put into message state: the model has no reason to read its
own narration back, and repeating it every round is quadratic prompt cost.
"""

from components import console


class TestToolResultReason:
    def test_reason_becomes_the_subtitle(self, capsys):
        console.tool_result_panel("execute_shell_command", "85G free",
                                  reason="Checking disk usage before the cleanup.")
        out = capsys.readouterr().out
        assert "Checking disk usage before the cleanup." in out

    def test_no_reason_means_no_subtitle(self, capsys):
        console.tool_result_panel("execute_shell_command", "85G free")
        out = capsys.readouterr().out
        assert "85G free" in out

    def test_reason_is_collapsed_to_one_line(self, capsys):
        console.tool_result_panel("read_any_file", "content",
                                  reason="first line\n\nsecond   line")
        out = capsys.readouterr().out
        assert "first line second line" in out

    def test_long_reason_is_truncated(self, capsys):
        console.tool_result_panel("read_any_file", "content", reason="x" * 500)
        out = capsys.readouterr().out
        assert "x" * 500 not in out
        assert "..." in out

    def test_error_panel_still_carries_the_reason(self, capsys):
        console.tool_result_panel("execute_shell_command", "boom", is_error=True,
                                  reason="Trying the build.")
        out = capsys.readouterr().out
        assert "Trying the build." in out
