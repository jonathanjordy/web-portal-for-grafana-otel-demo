"""Shared LLM client built on Pydantic AI, using Google Gemini as the provider.

This is the single place that owns the model/provider configuration. Both the
chatbot and the diagnostic RCA summary call into here instead of hitting the
Gemini REST API directly. Swapping models is a one-line change (MODEL_NAME).

`gemini-3.5-flash` is a reasoning model: its thinking tokens count against the
output-token budget, so we set a generous `max_tokens` and keep thinking at the
LOW level to avoid the model truncating its actual answer mid-output.
"""

import os
from functools import lru_cache
from typing import TypeVar

from fastapi import HTTPException
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

MODEL_NAME = "gemini-3.5-flash"

# Generous default so reasoning tokens don't truncate the visible answer.
DEFAULT_MAX_TOKENS = 4096

T = TypeVar("T", bound=BaseModel)


@lru_cache()
def get_model() -> GoogleModel:
    """Build (and cache) the Gemini model wrapper. Requires GEMINI_API_KEY."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY not set in .env.",
        )
    provider = GoogleProvider(api_key=api_key)
    return GoogleModel(MODEL_NAME, provider=provider)


def _settings(temperature: float, max_tokens: int) -> GoogleModelSettings:
    return GoogleModelSettings(
        temperature=temperature,
        max_tokens=max_tokens,
        # Keep reasoning light so it doesn't eat the output budget.
        google_thinking_config={"thinking_level": "LOW"},
    )


def _build_history(history: list[dict] | None) -> list[ModelMessage]:
    """Convert the frontend's [{role, content}] turns into Pydantic AI messages.

    `user` turns become ModelRequest/UserPromptPart; anything else (e.g.
    `assistant`) becomes ModelResponse/TextPart. Pydantic AI maps these to the
    provider's required `user`/`model` roles, which is what fixes the multi-turn
    bug where Gemini rejected the literal role string `assistant`.
    """
    messages: list[ModelMessage] = []
    for turn in history or []:
        content = turn.get("content", "")
        if turn.get("role") == "user":
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        else:
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
    return messages


async def _run(agent: Agent, prompt: str, history, temperature, max_tokens):
    """Run an agent, translating provider/key failures into the HTTP contract."""
    try:
        return await agent.run(
            prompt,
            message_history=_build_history(history),
            model_settings=_settings(temperature, max_tokens),
        )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - surface any model/transport error as 502
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {str(exc)[:200]}",
        )


async def generate(
    prompt: str,
    history: list[dict] | None = None,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Run a single Gemini turn and return the plain-text output."""
    agent = Agent(get_model(), instructions=system or None)
    result = await _run(agent, prompt, history, temperature, max_tokens)
    return result.output


async def generate_structured(
    prompt: str,
    output_type: type[T],
    history: list[dict] | None = None,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> T:
    """Run a single Gemini turn and return a validated instance of `output_type`.

    Uses Pydantic AI structured output (native schema-constrained generation),
    so callers get typed fields instead of having to parse text/markdown.
    """
    agent = Agent(get_model(), output_type=output_type, instructions=system or None)
    result = await _run(agent, prompt, history, temperature, max_tokens)
    return result.output
