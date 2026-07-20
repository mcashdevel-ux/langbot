"""Unit tests for console.py — terminal formatting helpers.

These focus on the pure/near-pure logic: ANSI handling, truncation, paste
detection, progress bars, markdown rendering, unicode-safety and the writer
helpers (captured via capsys). Colorama may or may not be installed; tests
assert on behaviour that holds either way (e.g. by stripping ANSI).
"""

import console


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

class TestAnsiHelpers:
    def test_strip_ansi(self):
        assert console.strip_ansi("\033[31mred\033[0m") == "red"

    def test_strip_ansi_no_codes(self):
        assert console.strip_ansi("plain") == "plain"

    def test_ansi_len_ignores_codes(self):
        assert console.ansi_len("\033[31mred\033[0m") == 3

    def test_truncate_short_unchanged(self):
        assert console.truncate("hi", max_len=100) == "hi"

    def test_truncate_long_adds_ellipsis(self):
        out = console.truncate("x" * 200, max_len=10)
        assert out == "x" * 10 + "..."

    def test_truncate_preserves_ansi_codes(self):
        text = "\033[31m" + "a" * 50 + "\033[0m"
        out = console.truncate(text, max_len=5)
        # The visible chars are truncated but the color code is retained.
        assert "\033[31m" in out
        assert console.strip_ansi(out) == "aaaaa..."

    def test_truncate_unterminated_escape(self):
        # A stray ESC without a terminating 'm' shouldn't crash.
        out = console.truncate("\033abcdefghij", max_len=3)
        assert out.endswith("...")


# ---------------------------------------------------------------------------
# Paste detection
# ---------------------------------------------------------------------------

class TestPaste:
    def test_is_large_paste_empty(self):
        assert console.is_large_paste("") is False

    def test_is_large_paste_by_lines(self):
        assert console.is_large_paste("a\nb\nc\nd\ne\nf") is True

    def test_is_large_paste_by_chars(self):
        assert console.is_large_paste("x" * 501) is True

    def test_is_large_paste_small(self):
        assert console.is_large_paste("just one line") is False

    def test_collapse_paste_small_unchanged(self):
        assert console.collapse_paste("small") == "small"

    def test_collapse_paste_large(self):
        text = "line\n" * 10
        out = console.collapse_paste(text)
        assert out.startswith("[Pasted Text:")
        assert "lines" in out


# ---------------------------------------------------------------------------
# Progress bar
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_clamps_low(self):
        assert "0%" in console.strip_ansi(console.progress_bar(-1.0))

    def test_clamps_high(self):
        assert "100%" in console.strip_ansi(console.progress_bar(2.0))

    def test_midpoint(self):
        assert "50%" in console.strip_ansi(console.progress_bar(0.5))

    def test_label_included(self):
        assert "loading" in console.strip_ansi(console.progress_bar(0.5, label="loading"))

    def test_print_progress_writes(self, capsys):
        console.print_progress(0.5, label="task")
        assert "50%" in console.strip_ansi(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# Ghost text
# ---------------------------------------------------------------------------

class TestGhostText:
    def test_no_completion_returns_prefix(self):
        assert console.ghost_text("abc", "") == "abc"

    def test_matching_completion_appends_ghost(self):
        out = console.ghost_text("ab", "abcdef")
        assert console.strip_ansi(out) == "abcdef"

    def test_nonmatching_completion_returns_prefix(self):
        assert console.ghost_text("xyz", "abcdef") == "xyz"

    def test_equal_completion_returns_prefix(self):
        # completion == prefix → no ghost tail to add
        assert console.ghost_text("abc", "abc") == "abc"


# ---------------------------------------------------------------------------
# Status messages / dividers (writer helpers)
# ---------------------------------------------------------------------------

class TestWriters:
    def test_info(self, capsys):
        console.info("hello")
        assert "hello" in console.strip_ansi(capsys.readouterr().out)

    def test_success(self, capsys):
        console.success("done")
        assert "done" in console.strip_ansi(capsys.readouterr().out)

    def test_warning(self, capsys):
        console.warning("careful")
        assert "careful" in console.strip_ansi(capsys.readouterr().out)

    def test_error(self, capsys):
        console.error("boom")
        assert "boom" in console.strip_ansi(capsys.readouterr().out)

    def test_rule_plain(self, capsys):
        console.rule()
        assert capsys.readouterr().out.strip() != ""

    def test_rule_with_text(self, capsys):
        console.rule("Section")
        assert "Section" in console.strip_ansi(capsys.readouterr().out)

    def test_header(self, capsys):
        console.header("Title")
        assert "Title" in console.strip_ansi(capsys.readouterr().out)

    def test_banner(self, capsys):
        console.banner("Boot", subtitle="v1")
        out = console.strip_ansi(capsys.readouterr().out)
        assert "Boot" in out and "v1" in out

    def test_blank(self, capsys):
        console.blank()
        assert capsys.readouterr().out == "\n"

    def test_separator(self, capsys):
        console.separator()
        assert capsys.readouterr().out.strip() != ""

    def test_kv(self, capsys):
        console.kv("key", "value")
        out = console.strip_ansi(capsys.readouterr().out)
        assert "key" in out and "value" in out

    def test_knowledge_hint_zero_silent(self, capsys):
        console.knowledge_hint(0)
        assert capsys.readouterr().out == ""

    def test_knowledge_hint_singular(self, capsys):
        console.knowledge_hint(1)
        out = console.strip_ansi(capsys.readouterr().out)
        assert "1 relevant fact" in out and "facts" not in out

    def test_knowledge_hint_plural(self, capsys):
        console.knowledge_hint(3)
        assert "3 relevant facts" in console.strip_ansi(capsys.readouterr().out)

    def test_startup_tip(self, capsys):
        console.startup_tip("gpt")
        assert "/help" in console.strip_ansi(capsys.readouterr().out)

    def test_session_resume_banner(self, capsys):
        console.session_resume_banner(5, knowledge_count=2, is_truncated=True)
        out = console.strip_ansi(capsys.readouterr().out)
        assert "Resumed session (5 messages)" in out
        assert "knowledge facts" in out


# ---------------------------------------------------------------------------
# Tool call / result display
# ---------------------------------------------------------------------------

class TestToolDisplay:
    def test_tool_icon_known(self):
        icon, _ = console._tool_icon("shell")
        assert icon == "$"

    def test_tool_icon_unknown_default(self):
        icon, _ = console._tool_icon("does-not-exist")
        assert icon == "\u25cf"

    def test_tool_call_no_args(self, capsys):
        console.tool_call("shell")
        assert "shell()" in console.strip_ansi(capsys.readouterr().out)

    def test_tool_call_with_args(self, capsys):
        console.tool_call("read", {"path": "/etc/hosts"})
        out = console.strip_ansi(capsys.readouterr().out)
        assert "read(" in out and "path=" in out

    def test_tool_call_truncates_long_arg(self, capsys):
        console.tool_call("read", {"path": "x" * 100})
        assert "..." in capsys.readouterr().out

    def test_tool_result_ok(self, capsys):
        console.tool_result("read", "file contents", elapsed_ms=100)
        out = console.strip_ansi(capsys.readouterr().out)
        assert "file contents" in out
        assert "100ms" in out

    def test_tool_result_slow_seconds(self, capsys):
        console.tool_result("read", "x", elapsed_ms=3000)
        assert "3.0s" in console.strip_ansi(capsys.readouterr().out)

    def test_tool_result_error(self, capsys):
        console.tool_result("read", "nope", is_error=True)
        assert "nope" in console.strip_ansi(capsys.readouterr().out)

    def test_tool_result_truncates_large(self, capsys):
        console.tool_result("read", "y" * 600, full_len=10000)
        out = console.strip_ansi(capsys.readouterr().out)
        assert "10,000 total chars" in out


# ---------------------------------------------------------------------------
# Agent response
# ---------------------------------------------------------------------------

class TestAgentResponse:
    def test_empty_silent(self, capsys):
        console.agent_response("")
        assert capsys.readouterr().out == ""

    def test_response_label(self, capsys):
        console.agent_response("hi there")
        assert "hi there" in console.strip_ansi(capsys.readouterr().out)

    def test_thought_label(self, capsys):
        console.agent_response("thinking...", label="Thought")
        out = console.strip_ansi(capsys.readouterr().out)
        assert "Thought" in out and "thinking..." in out


# ---------------------------------------------------------------------------
# Tables & code
# ---------------------------------------------------------------------------

class TestTableCode:
    def test_table_empty(self, capsys):
        console.table([])
        assert "(empty)" in console.strip_ansi(capsys.readouterr().out)

    def test_table_with_headers(self, capsys):
        console.table([["a", "b"], ["c", "d"]], headers=["H1", "H2"])
        out = console.strip_ansi(capsys.readouterr().out)
        assert "H1" in out and "a" in out and "d" in out

    def test_table_no_headers(self, capsys):
        console.table([["x", "y"]])
        assert "x" in console.strip_ansi(capsys.readouterr().out)

    def test_code_block(self, capsys):
        console.code("print('hi')")
        assert "print('hi')" in console.strip_ansi(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# Status row
# ---------------------------------------------------------------------------

class TestStatusRow:
    def test_loading(self):
        out = console.strip_ansi(console.status_row(loading_msg="working", is_loading=True))
        assert "working" in out

    def test_not_loading(self):
        out = console.strip_ansi(console.status_row(loading_msg="idle"))
        assert "idle" in out

    def test_tip_and_context(self):
        out = console.strip_ansi(console.status_row(tip="press q", context="ctx"))
        assert "press q" in out and "ctx" in out

    def test_print_status_row(self, capsys):
        console.print_status_row(loading_msg="go")
        assert "go" in console.strip_ansi(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

class TestMarkdown:
    def test_apply_md_simple_empty(self):
        assert console._apply_md_simple("") == ""

    def test_apply_md_simple_strips_or_renders_bold(self):
        out = console._apply_md_simple("**bold**")
        # Either markers stripped (no color) or wrapped in ANSI; the word remains.
        assert "bold" in console.strip_ansi(out)
        assert "**" not in console.strip_ansi(out)

    def test_apply_md_simple_link(self):
        out = console._apply_md_simple("[text](http://x)")
        assert "text" in console.strip_ansi(out)

    def test_apply_md_simple_header(self):
        out = console._apply_md_simple("# Heading")
        assert "Heading" in console.strip_ansi(out)

    def test_render_markdown_returns_str(self):
        assert isinstance(console._render_markdown("**x**"), str)

    def test_md_print_writes(self, capsys):
        console.md_print("**hello**")
        assert "hello" in console.strip_ansi(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class TestPanel:
    def test_panel_basic(self, capsys):
        console.panel(title="T", content="body")
        out = console.strip_ansi(capsys.readouterr().out)
        assert "body" in out

    def test_panel_markdown(self, capsys):
        console.panel(title="T", content="**bold** text", render_md=True)
        out = console.strip_ansi(capsys.readouterr().out)
        assert "bold" in out

    def test_panel_tiny_width_floor(self, capsys):
        console.panel(content="hi", width=5)
        assert "hi" in console.strip_ansi(capsys.readouterr().out)


# ---------------------------------------------------------------------------
# Unicode safety
# ---------------------------------------------------------------------------

class TestUnicodeSafe:
    def test_plain_ascii(self):
        assert console._unicode_safe("hello") == "hello"

    def test_non_string_coerced(self):
        assert console._unicode_safe(123) == "123"

    def test_keeps_utf8_on_non_windows(self):
        # On non-win32 valid utf-8 is returned unchanged.
        assert console._unicode_safe("café") == "café"


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class TestSpinner:
    def test_start_stop(self):
        sp = console.GradientSpinner("busy")
        sp.start()
        assert sp._running is True
        sp.stop()
        assert sp._running is False

    def test_double_start_is_safe(self):
        sp = console.GradientSpinner()
        sp.start()
        sp.start()  # should not spawn a second thread / raise
        sp.stop()

    def test_update_message(self):
        sp = console.GradientSpinner("a")
        sp.update("b")
        assert sp.msg == "b"

    def test_context_manager(self):
        with console.GradientSpinner("ctx") as sp:
            assert sp._running is True
        assert sp._running is False

    def test_spinner_alias(self):
        assert console.Spinner is console.GradientSpinner
