"""A ``ChatOpenAI`` that keeps the provider's reasoning instead of dropping it.

``langchain_openai.ChatOpenAI`` targets the official OpenAI schema on purpose and
documents that non-standard provider fields — ``reasoning_content``,
``reasoning_details`` — are "not extracted or preserved". The hosted proxy this
agent talks to returns exactly those fields, so a whole reasoning channel was
being thrown away before anything could display it: the model reasoned on every
step and the console showed none of it.

The library's own advice for that case is to use a provider-specific subclass, so
this is the smallest version of one: parse the response normally, then copy the
reasoning text into ``additional_kwargs``, which is part of the message and
therefore survives checkpointing — a resumed session can still show the reasoning
it already has.

Only the non-streaming path is covered (``_create_chat_result``); this agent
calls ``.invoke()``. A streaming consumer would also need
``_convert_delta_to_message_chunk``, since deltas take a different route.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from .utils import thinking_content

logger = logging.getLogger(__name__)

# Field names a provider may use for the reasoning channel, in priority order.
REASONING_KEYS = ("reasoning_content", "reasoning", "reasoning_details")


def _raw_reasoning(response) -> str:
    """Reasoning text carried by a raw chat-completion response, or ``""``.

    Reads from the raw response rather than the converted message, because the
    conversion is what drops the field in the first place. Both shapes langchain
    accepts are handled: a plain dict (what a non-pydantic endpoint yields) and an
    ``openai`` model, where the extra fields live in ``model_extra`` (or as real
    attributes when the SDK version declares them).
    """
    try:
        message = (
            response["choices"][0]["message"] if isinstance(response, dict)
            else response.choices[0].message
        )
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""

    if isinstance(message, dict):
        payload = message
    else:
        payload = dict(getattr(message, "model_extra", None) or {})
        for key in REASONING_KEYS:
            value = getattr(message, key, None)
            if value and key not in payload:
                payload[key] = value
    return thinking_content(payload)


class ReasoningChatOpenAI(ChatOpenAI):
    """``ChatOpenAI`` that surfaces provider reasoning as ``reasoning_content``."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        reasoning = _raw_reasoning(response)
        if not reasoning:
            return result
        for generation in result.generations:
            message = getattr(generation, "message", None)
            kwargs = getattr(message, "additional_kwargs", None)
            if isinstance(kwargs, dict) and not kwargs.get("reasoning_content"):
                kwargs["reasoning_content"] = reasoning
        return result
