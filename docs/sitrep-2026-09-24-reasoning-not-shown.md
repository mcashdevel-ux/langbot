# Situation Report — the console showed no reasoning

**Date:** 2026-09-24 09:40 +06
**Repo:** `~/ai/repos/langbot` @ `7eab593` + uncommitted work
**Severity:** medium — the agent reasoned on every step and none of it was visible
**Status:** fixed, verified live

---

## 1. Symptom

The console rendered tool calls, tool results and answers, but never a `🧠 Thought`
panel, even though the model reasons on every step.

## 2. Root cause A — the reasoning was dropped before rendering

A live request to the configured endpoint (`llm.base_url` = the hosted proxy,
`deepseek-v4.1-flash`) returns the reasoning **beside** the answer:

```json
{"message": {"content": "17 × 23 = 391",
             "reasoning_content": "We need answer simple multiplication…",
             "provider_specific_fields": {"reasoning_details": [...]}}}
```

`langchain_openai.ChatOpenAI` **drops non-standard provider fields by design** — its
own docstring says `reasoning_content` / `reasoning_details` are "not extracted or
preserved" and points at provider-specific subclasses. So the field was gone before
any renderer could see it. Confirmed by inspecting a persisted checkpoint: the
`AIMessage` in state carried `additional_kwargs = {"refusal": None}` and no reasoning.

This is why the Qwen-style inline `think` tags worked and this did not: inline tags
live in `content`, which langchain keeps.

**Fix:** `components/llm_reasoning.ReasoningChatOpenAI` — the smallest version of the
subclass the library recommends. It parses the response normally, then copies the
reasoning into `additional_kwargs`, which is part of the message and therefore
survives checkpointing (so a resumed session can still show the reasoning it has).

## 3. Root cause B — the renderer only understood one channel, and only half of it

`_render_message` replaced the tags with `<thought>` and split on the closing tag,
taking `parts[0]`. A lone opening tag (truncated generation) therefore fell through
and was rendered **verbatim as the answer**.

**Fix:** `tool_call_repair.inline_reasoning(text) -> (reasoning, answer)` splits the
two properly, treating an unclosed tag as all-reasoning, and `_render_message`
renders the reasoning as a Thought panel *before* the answer it explains.

## 4. Also fixed

- `utils.thinking_content` reads every shape the reasoning arrives in — top-level
  attribute, `additional_kwargs`, `provider_specific_fields`, `reasoning_details`
  blocks, and content blocks typed `thinking` — so the display does not depend on
  which provider is configured.
- `/health`'s reasoning-token count now prefers the provider's reported
  `reasoning_tokens`. The old text heuristic only saw inline tags, so a separate
  `reasoning_content` field scored **zero** and the overhead was invisible.
- A tool-result panel now carries the reasoning that preceded the call as its
  subtitle. That reasoning is kept **out of message state** deliberately: the model
  has no reason to read its own narration back, and repeating it every round is
  quadratic prompt cost.
- `llm.show_thinking` (default `true`) quiets the panels without turning reasoning
  off — that remains `llm.thinking_mode`.

## 5. Verified

- Live endpoint: the subclass recovers the reasoning text and the reported token
  count (`reasoning tokens reported: 58`).
- Real turn through the graph (`scripts/check_reasoning_display.py`): 2 Thought
  panels, 2 tool panels, 1 answer panel; a second prompt recorded 144 reasoning
  tokens in 1 step and rendered its Thought panel.
- 832 tests pass (5 new test modules' worth: `test_llm_reasoning.py`,
  `test_thinking_display.py`, `test_tool_result_reason.py`).

## 6. Residual / known

- **Streaming is not covered.** Only `_create_chat_result` is overridden; this agent
  calls `.invoke()`. A streaming consumer would also need
  `_convert_delta_to_message_chunk`, since deltas take a different route.
- **A trivial prompt may produce no reasoning at all** (the model answers directly),
  in which case no Thought panel is correct. The check script reports that as
  INCONCLUSIVE rather than a failure.
- **The agent's own tool layer strips a literal opening think tag out of tool
  arguments.** Writing `"<think>x</think>"` through `write_any_file`/`patch_file`
  lands as `"x</think>"`; the closing tag survives. It is not the app (the app's own
  regexes and the repo's existing tests are intact) — it is the write path, and it
  silently corrupted two of my own edits before I noticed. Tests that need the tag
  assemble it from pieces (`"<" "think>"`). Worth fixing in the tool layer.
