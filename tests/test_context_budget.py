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


class TestInstrumentation:
    def setup_method(self):
        ctx.reset_stats()

    def teardown_method(self):
        ctx.reset_stats()

    def test_no_steps_yet_is_said_plainly(self):
        assert ctx.stats_summary() == "no agent steps yet"

    def test_peaks_are_peaks_not_last_values(self):
        ctx.record_step(history_tokens=9000, overhead_tokens=1200, schema_tokens=800)
        ctx.record_step(history_tokens=100, overhead_tokens=900, schema_tokens=500)
        stats = ctx.stats()
        assert stats["steps"] == 2
        assert stats["peak_history"] == 9000
        assert stats["peak_overhead"] == 1200
        assert stats["peak_schemas"] == 800
        assert stats["peak_prompt"] == 10200

    def test_reuse_is_reported_as_a_share_of_the_prompt(self):
        # Half of a 1000-token prompt reused, then none of the next one.
        ctx.record_step(history_tokens=900, overhead_tokens=100, prefix_tokens=500)
        ctx.record_step(history_tokens=900, overhead_tokens=100, prefix_tokens=0)
        stats = ctx.stats()
        assert stats["prefix_reused"] == 500
        assert stats["prefix_reprocessed"] == 1500
        assert "cache reuse 25%" in ctx.stats_summary()

    def test_a_prefix_larger_than_the_prompt_cannot_inflate_reuse(self):
        ctx.record_step(history_tokens=10, overhead_tokens=10, prefix_tokens=999)
        assert ctx.stats()["prefix_reused"] == 20
        assert ctx.stats()["prefix_reprocessed"] == 0

    def test_compactions_are_counted_with_what_they_dropped(self):
        ctx.record_compaction(7, 4200)
        ctx.record_compaction(2, 800)
        stats = ctx.stats()
        assert stats["compactions"] == 2
        assert stats["tokens_dropped"] == 5000

    def test_summary_names_the_reserve_it_is_judged_against(self):
        ctx.record_step(history_tokens=1000, overhead_tokens=2000, schema_tokens=1500)
        summary = ctx.stats_summary()
        assert f"of reserve {ctx.RESERVE_TOKENS}" in summary
        assert "schemas 1500" in summary


class TestSharedPrefixTokens:
    def test_an_identical_prompt_is_fully_reusable(self):
        rendered = ["system:prompt", "human:hello", "ai:hi"]
        assert ctx.shared_prefix_tokens(rendered, rendered) == sum(
            ctx.estimate_tokens(m) for m in rendered
        )

    def test_appending_keeps_the_whole_previous_prefix(self):
        before = ["system:prompt", "human:hello"]
        after = ["system:prompt", "human:hello", "ai:hi"]
        assert ctx.shared_prefix_tokens(before, after) == ctx.shared_prefix_tokens(
            before, before
        )

    def test_a_changed_lead_invalidates_everything_after_it(self):
        # This is what compaction does: the summary lives in the system message, so
        # rewriting it means the server reuses nothing.
        before = ["system:prompt", "human:hello", "ai:hi"]
        after = ["system:prompt\n\nEarlier in this session:...", "human:hello", "ai:hi"]
        assert ctx.shared_prefix_tokens(before, after) == 0

    def test_a_change_in_the_middle_keeps_the_lead(self):
        before = ["system:p", "human:a", "ai:x"]
        after = ["system:p", "human:b", "ai:x"]
        assert ctx.shared_prefix_tokens(before, after) == ctx.estimate_tokens("system:p")

    def test_no_previous_prompt_means_no_reuse(self):
        assert ctx.shared_prefix_tokens([], ["system:p"]) == 0
