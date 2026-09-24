"""Tests for surfacing model reasoning in the console.

The bug these pin down: the configured model returns its reasoning in a
``reasoning_content`` field *beside* the answer (DeepSeek/OpenRouter-style
OpenAI-compatible endpoints), and ``langchain_openai.ChatOpenAI`` drops that
field by design — so the agent reasoned on every step and the console showed
none of it. The Qwen-style think tags inside content worked, which is why the
gap was easy to miss.

Note: the think tags below are assembled from pieces rather than written as
literals, because the agent's own tool layer strips a literal opening tag out of
tool arguments (a bug tracked separately). Do not "simplify" them.
"""

from langchain_core.messages import AIMessage

from components import tool_call_repair as repair
from components.utils import reasoning_tokens, thinking_content

OPEN = "<" "think>"
CLOSE = "</" "think>"


class TestThinkingContent:
    def test_openai_style_reasoning_content_field(self):
        msg = AIMessage(content="391", additional_kwargs={"reasoning_content": "17*20+17*3"})
        assert thinking_content(msg) == "17*20+17*3"

    def test_provider_specific_fields_nesting(self):
        # What the hosted proxy returns: reasoning under provider_specific_fields.
        msg = AIMessage(content="hi", additional_kwargs={
            "provider_specific_fields": {"reasoning": "weighing the options"},
        })
        assert thinking_content(msg) == "weighing the options"

    def test_reasoning_details_block_list(self):
        msg = AIMessage(content="hi", additional_kwargs={
            "provider_specific_fields": {
                "reasoning_details": [{"type": "reasoning.text", "text": "step one"}],
            },
        })
        assert thinking_content(msg) == "step one"

    def test_top_level_attribute(self):
        class Resp:
            content = "answer"
            reasoning_content = "top level thought"

        assert thinking_content(Resp()) == "top level thought"

    def test_content_block_type_thinking(self):
        msg = AIMessage(content=[{"type": "thinking", "text": "block thought"},
                                 {"type": "text", "text": "answer"}])
        assert thinking_content(msg) == "block thought"

    def test_no_reasoning_is_empty(self):
        assert thinking_content(AIMessage(content="just an answer")) == ""
        assert thinking_content(AIMessage(content="x", additional_kwargs={})) == ""

    def test_plain_string_additional_kwargs_is_not_mistaken_for_reasoning(self):
        # A tool-call repair can leave arbitrary keys around; only the known
        # reasoning keys count, and an unrelated key must not be echoed as thought.
        msg = AIMessage(content="x", additional_kwargs={"refusal": "no"})
        assert thinking_content(msg) == ""


class TestReasoningTokens:
    def test_reads_output_token_details(self):
        msg = AIMessage(content="x", usage_metadata={
            "input_tokens": 10, "output_tokens": 20, "total_tokens": 30,
            "output_token_details": {"reasoning": 62},
        })
        assert reasoning_tokens(msg) == 62

    def test_missing_usage_is_zero(self):
        assert reasoning_tokens(AIMessage(content="x")) == 0


class TestInlineReasoning:
    def test_closed_block_is_split_out(self):
        thought, answer = repair.inline_reasoning(
            f"{OPEN}weighing it up{CLOSE}The answer."
        )
        assert thought == "weighing it up"
        assert answer == "The answer."

    def test_thinking_tag_alias(self):
        thought, answer = repair.inline_reasoning(
            f"<thinking>hmm</thinking>Done."
        )
        assert thought == "hmm" and answer == "Done."

    def test_unclosed_block_is_all_reasoning(self):
        # A generation cut off mid-thought must not be shown as an answer.
        thought, answer = repair.inline_reasoning(f"Preamble {OPEN}still going")
        assert thought == "still going"
        assert answer == "Preamble"

    def test_no_block_passes_through(self):
        assert repair.inline_reasoning("plain answer") == ("", "plain answer")

    def test_markup_after_a_block_is_cleaned_from_the_answer(self):
        thought, answer = repair.inline_reasoning(
            f"{OPEN}x{CLOSE}<|im_end|>real answer"
        )
        assert thought == "x"
        assert answer == "real answer"


class TestRenderMessageShowsReasoning:
    def _render(self, msg, monkeypatch):
        import langbot
        shown = []
        monkeypatch.setattr(langbot.ui, "thought_panel", lambda c: shown.append(c))
        monkeypatch.setattr(langbot.ui, "final_answer_panel", lambda c, **k: None)
        monkeypatch.setattr(langbot.ui, "tool_call_panel", lambda *a, **k: None)
        langbot._render_message(msg)
        return shown

    def test_side_channel_reasoning_gets_a_panel(self, monkeypatch):
        msg = AIMessage(content="The answer is 391.",
                        additional_kwargs={"reasoning_content": "17*20+17*3=391"})
        shown = self._render(msg, monkeypatch)
        assert "17*20+17*3=391" in shown

    def test_inline_reasoning_gets_a_panel_and_not_the_answer(self, monkeypatch):
        msg = AIMessage(content=f"{OPEN}private deliberation{CLOSE}Public answer.")
        shown = self._render(msg, monkeypatch)
        assert "private deliberation" in shown
        assert "Public answer." not in shown

    def test_show_thinking_off_suppresses_the_panel(self, monkeypatch):
        import langbot
        monkeypatch.setattr(langbot, "SHOW_THINKING", False)
        msg = AIMessage(content="391",
                        additional_kwargs={"reasoning_content": "deliberation"})
        assert self._render(msg, monkeypatch) == []

    def test_reasoning_does_not_become_the_answer(self, monkeypatch):
        # Regression: the raw tags used to be shown verbatim in the answer panel
        # when only one of the two tags survived.
        import langbot
        answers = []
        monkeypatch.setattr(langbot.ui, "final_answer_panel",
                            lambda c, **k: answers.append(c))
        monkeypatch.setattr(langbot.ui, "thought_panel", lambda c: None)
        langbot._render_message(AIMessage(content="Just the answer."))
        assert answers == ["Just the answer."]
