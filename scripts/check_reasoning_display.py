"""End-to-end check: does a real turn surface reasoning to the console?

Runs the real graph (real model, real tools, real console renderers) with the
renderers wrapped to capture what a user would actually see. Nothing is asserted
about model wording — only that the reasoning channel arrives at the panel.

    python scripts/check_reasoning_display.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import langbot  # noqa: E402  — imports the graph, tools and console wiring

captured = {"thought": [], "answer": [], "tool": []}


def _wrap(name, sink):
    original = getattr(langbot.ui, name)

    def wrapper(content, *args, **kwargs):
        sink.append(str(content))
        return original(content, *args, **kwargs)

    setattr(langbot.ui, name, wrapper)


_wrap("thought_panel", captured["thought"])
_wrap("final_answer_panel", captured["answer"])
_wrap("tool_result_panel", captured["tool"])

PROMPT = os.environ.get(
    "CHECK_PROMPT",
    "In one line: how much free disk space is there?",
)
# The graph recurses once per tool round; the app's own limit is the honest number.
LIMIT = int(os.environ.get("CHECK_RECURSION_LIMIT", "40"))


def main() -> int:
    config = {"configurable": {"thread_id": "check_reasoning_display"},
              "recursion_limit": LIMIT}
    print(f"model: {langbot.LLM_MODEL}  |  show_thinking: {langbot.SHOW_THINKING}")
    print(f"prompt: {PROMPT}\n")
    langbot._stream_turn(langbot.builder.compile(), config, PROMPT)

    print("\n--- what the console received ---")
    print(f"thought panels : {len(captured['thought'])}")
    for text in captured["thought"]:
        print(f"    {text[:100]!r}")
    print(f"tool panels    : {len(captured['tool'])}")
    print(f"answer panels  : {len(captured['answer'])}")
    stats = langbot._ctx.stats()
    print(f"reasoning tokens recorded: {stats['thinking_tokens']} "
          f"in {stats['thinking_calls']} step(s)")

    # The honest criterion: reasoning the model actually produced must be shown.
    # A trivial prompt may legitimately produce none, so "no panels" is only a
    # failure when the provider reported reasoning tokens for the turn.
    if stats["thinking_tokens"] and not captured["thought"]:
        print("\nFAIL: the model reasoned but no Thought panel was rendered")
        return 1
    if not stats["thinking_tokens"]:
        print("\nINCONCLUSIVE: the model produced no reasoning for this prompt "
              "(try a prompt that needs a step or two)")
        return 0
    print("\nOK: reasoning was surfaced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
