"""Unit tests for context_budget.py — token accounting and history compaction."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import components.context_budget as ctx


class TestTokenAccounting:
    def test_empty_text_is_free(self):
        assert ctx.estimate_tokens("") == 0

    def test_longer_text_costs_more(self):
        assert ctx.estimate_tokens("word " * 100) > ctx.estimate_tokens("word")

    def test_tool_calls_are_counted(self):
        plain = AIMessage(content="")
        calling = AIMessage(content="", tool_calls=[
            {"name": "read_any_file", "args": {"file_path": "/tmp/x"}, "id": "1"},
        ])
        assert ctx.message_tokens(calling) > ctx.message_tokens(plain)

    def test_total_sums_the_thread(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert ctx.total_tokens(msgs) == sum(ctx.message_tokens(m) for m in msgs)


class TestSplit:
    def test_short_thread_is_untouched(self):
        msgs = [HumanMessage(content=str(i)) for i in range(3)]
        older, recent = ctx.split_for_compaction(msgs, keep_last=12)
        assert older == []
        assert recent == msgs

    def test_split_keeps_the_tail(self):
        msgs = [HumanMessage(content=str(i)) for i in range(10)]
        older, recent = ctx.split_for_compaction(msgs, keep_last=4)
        assert len(older) == 6
        assert len(recent) == 4

    def test_tool_result_is_never_orphaned_from_its_call(self):
        # A ToolMessage without the AI message that requested it is rejected by
        # the chat API, so the boundary must move before the whole tool round.
        msgs = [
            HumanMessage(content="go"),
            AIMessage(content="", tool_calls=[
                {"name": "recall", "args": {"query": "q"}, "id": "1"},
            ]),
            ToolMessage(content="r1", tool_call_id="1", name="recall"),
            ToolMessage(content="r2", tool_call_id="2", name="recall"),
        ]
        older, recent = ctx.split_for_compaction(msgs, keep_last=2)
        assert [m.type for m in recent] == ["ai", "tool", "tool"]
        assert len(older) == 1


class TestNeedsCompaction:
    def test_small_thread_is_under_budget(self):
        assert not ctx.needs_compaction([HumanMessage(content="hi")])

    def test_large_thread_crosses_the_threshold(self):
        big = [HumanMessage(content="x " * 20_000) for _ in range(4)]
        assert ctx.needs_compaction(big)


class TestCompact:
    def _thread(self, n=20):
        return [HumanMessage(content=f"message {i}") for i in range(n)]

    def test_summarizes_the_older_half(self):
        seen = {}

        def summarize(prompt):
            seen["prompt"] = prompt
            return "the user counted to twenty"

        dropped, recent, summary = ctx.compact(
            self._thread(), summarize, keep_last=5)
        assert len(dropped) == 15
        assert len(recent) == 5
        assert summary == "the user counted to twenty"
        assert "message 0" in seen["prompt"]

    def test_previous_summary_is_carried_into_the_prompt(self):
        seen = {}

        def summarize(prompt):
            seen["prompt"] = prompt
            return "new summary"

        ctx.compact(self._thread(), summarize, previous_summary="older notes",
                    keep_last=5)
        assert "older notes" in seen["prompt"]

    def test_summary_is_capped(self, monkeypatch):
        monkeypatch.setattr(ctx, "SUMMARY_MAX_CHARS", 20)
        _dropped, _recent, summary = ctx.compact(
            self._thread(), lambda p: "s" * 500, keep_last=5)
        assert len(summary) == 20

    def test_summarizer_failure_keeps_history(self):
        # Losing the turn to a summarizer error would be worse than being over
        # budget for a step, so nothing is dropped.
        def boom(_prompt):
            raise RuntimeError("no server")

        dropped, recent, summary = ctx.compact(
            self._thread(), boom, previous_summary="kept", keep_last=5)
        assert dropped == []
        assert len(recent) == 20
        assert summary == "kept"

    def test_empty_summary_keeps_history(self):
        dropped, recent, _summary = ctx.compact(
            self._thread(), lambda p: "   ", keep_last=5)
        assert dropped == []
        assert len(recent) == 20

    def test_nothing_to_drop_is_a_no_op(self):
        msgs = self._thread(3)
        dropped, recent, summary = ctx.compact(
            msgs, lambda p: "unused", keep_last=12)
        assert dropped == []
        assert recent == msgs
        assert summary == ""
