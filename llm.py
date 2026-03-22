"""
LLM API call helper with built-in token accounting.
This file is STATIC -- the agent does NOT modify it.

All LLM calls MUST go through `call_llm()`. It tracks every call's
token usage and cost so that evaluation metrics are trustworthy.

Built on PydanticAI for unified model access across OpenAI and Anthropic.

Usage:
    from llm import call_llm, get_usage, reset_usage

    # Plain text response
    response = await call_llm(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-5-mini",
    )

    # Structured output with a Pydantic model
    from pydantic import BaseModel

    class Analysis(BaseModel):
        pain_points: list[str]
        score: int

    data = await call_llm(
        messages=[{"role": "user", "content": "analyze this lead"}],
        model="gpt-5-mini",
        output_type=Analysis,
    )
    # data is an Analysis instance: data.pain_points, data.score

    usage = get_usage()  # cumulative token counts and cost
"""

from __future__ import annotations

import contextvars
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, overload

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Load .env file
# ---------------------------------------------------------------------------


def _load_dotenv() -> None:
    """Load key=value pairs from .env file into os.environ."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=BaseModel)

MODELS: dict[str, dict[str, Any]] = {
    # OpenAI (via Responses API)
    "gpt-5-mini": {
        "pydantic_ai_model": "openai-responses:gpt-5-mini",
        "provider": "openai",
        "pricing": (0.25, 2.00),
        "valid_efforts": {"minimal", "low", "medium", "high"},
    },
    "gpt-5-nano": {
        "pydantic_ai_model": "openai-responses:gpt-5-nano",
        "provider": "openai",
        "pricing": (0.05, 0.40),
        "valid_efforts": {"minimal", "low", "medium", "high"},
    },
    "gpt-5.4": {
        "pydantic_ai_model": "openai-responses:gpt-5.4",
        "provider": "openai",
        "pricing": (2.50, 15.00),
        "valid_efforts": {"none", "low", "medium", "high", "xhigh"},
    },
    # Anthropic
    "sonnet-4.6": {
        "pydantic_ai_model": "anthropic:claude-sonnet-4-6",
        "provider": "anthropic",
        "pricing": (3.00, 15.00),
        "valid_efforts": {"low", "medium", "high"},
    },
    "haiku-4.5": {
        "pydantic_ai_model": "anthropic:claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "pricing": (1.00, 5.00),
        "valid_efforts": set(),  # effort param not supported; uses budget_tokens
    },
    "opus-4.6": {
        "pydantic_ai_model": "anthropic:claude-opus-4-6",
        "provider": "anthropic",
        "pricing": (5.00, 25.00),
        "valid_efforts": {"low", "medium", "high", "max"},
    },
}

DEFAULT_MODEL = "gpt-5-mini"

# ---------------------------------------------------------------------------
# Usage tracking (global, cumulative per reset cycle)
# ---------------------------------------------------------------------------


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0
    call_log: list[dict[str, Any]] = field(default_factory=list)


_usage_var: contextvars.ContextVar[LLMUsage] = contextvars.ContextVar("llm_usage")


def get_usage() -> LLMUsage:
    """Get cumulative usage since last reset (context-local)."""
    try:
        return _usage_var.get()
    except LookupError:
        u = LLMUsage()
        _usage_var.set(u)
        return u


def reset_usage() -> None:
    """Reset cumulative usage counters (context-local)."""
    _usage_var.set(LLMUsage())


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODELS[model]["pricing"] if model in MODELS else (1.0, 3.0)
    return (input_tokens / 1_000_000) * prices[0] + (output_tokens / 1_000_000) * prices[1]


# ---------------------------------------------------------------------------
# Build model settings per provider
# ---------------------------------------------------------------------------


def _build_settings(
    model: str,
    provider: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
) -> dict[str, Any]:
    """Build provider-specific model_settings dict for PydanticAI."""
    settings: dict[str, Any] = {"max_tokens": max_tokens}

    valid_efforts = MODELS[model].get("valid_efforts", set())
    effort = reasoning_effort if reasoning_effort in valid_efforts else None

    if provider == "openai":
        if effort:
            settings["openai_reasoning_effort"] = effort
        # OpenAI reasoning models don't support temperature
    else:
        settings["temperature"] = temperature
        if effort:
            settings["anthropic_effort"] = effort

    return settings


# ---------------------------------------------------------------------------
# Public API: the ONE function everyone must use
# ---------------------------------------------------------------------------


@overload
async def call_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = ...,
    temperature: float = ...,
    max_tokens: int = ...,
    reasoning_effort: str = ...,
    system_prompt: str | None = ...,
    output_type: None = ...,
) -> str: ...


@overload
async def call_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = ...,
    temperature: float = ...,
    max_tokens: int = ...,
    reasoning_effort: str = ...,
    system_prompt: str | None = ...,
    output_type: type[T] = ...,
) -> T: ...


async def call_llm(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    reasoning_effort: str = "medium",
    system_prompt: str | None = None,
    output_type: type[T] | None = None,
) -> str | T:
    """Call an LLM and return the response.

    Token usage and cost are automatically tracked. Use get_usage() to read.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
                  System messages are extracted and used as system_prompt if
                  system_prompt is not explicitly provided.
        model: One of "gpt-5-mini", "gpt-5-nano", "gpt-5.4", "sonnet-4.6", "haiku-4.5", "opus-4.6".
        temperature: Sampling temperature (Anthropic only; ignored for OpenAI reasoning models).
        max_tokens: Maximum output tokens.
        reasoning_effort: OpenAI: "none", "minimal", "low", "medium", "high", "xhigh".
                          Anthropic: "low", "medium", "high", "max".
        system_prompt: System prompt override. If not set, extracted from messages.
        output_type: A Pydantic BaseModel class for structured output.
                     If None, returns plain text (str).

    Returns:
        str if output_type is None, otherwise an instance of output_type.
    """
    from pydantic_ai import Agent

    model = model or DEFAULT_MODEL
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Valid: {list(MODELS.keys())}")

    spec = MODELS[model]
    pydantic_ai_model: str = spec["pydantic_ai_model"]
    provider: str = spec["provider"]

    # Extract system prompt and user messages
    sys_prompt = system_prompt
    user_messages: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system" and sys_prompt is None:
            sys_prompt = m["content"]
        else:
            user_messages.append(m)

    # Build the user prompt from remaining messages
    user_prompt = "\n\n".join(m["content"] for m in user_messages)

    # Build settings
    settings = _build_settings(model, provider, temperature, max_tokens, reasoning_effort)

    # Create agent
    agent_kwargs: dict[str, Any] = {
        "model": pydantic_ai_model,
        "model_settings": settings,
    }
    if sys_prompt:
        agent_kwargs["system_prompt"] = sys_prompt
    if output_type is not None:
        agent_kwargs["output_type"] = output_type

    agent: Agent[None, Any] = Agent(**agent_kwargs)

    # Run
    t0 = time.time()
    result = await agent.run(user_prompt)
    dt = time.time() - t0

    # Extract usage
    usage = result.usage()
    in_tok = usage.input_tokens or 0
    out_tok = usage.output_tokens or 0
    cost = _estimate_cost(model, in_tok, out_tok)

    # Track usage (context-local, safe for parallel execution)
    _u = get_usage()
    _u.input_tokens += in_tok
    _u.output_tokens += out_tok
    _u.calls += 1
    _u.cost_usd += cost
    _u.duration_s += dt
    _u.call_log.append({
        "model": model,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": cost,
        "duration_s": dt,
    })

    output: Any = result.output
    return output  # type: ignore[no-any-return]
