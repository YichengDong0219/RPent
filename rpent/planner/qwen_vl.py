"""Qwen multimodal adapter for the user's OpenAI-compatible vLLM service."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from openai import DefaultAsyncHttpxClient
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

QWEN_VL_MODEL_PREFIX = "qwen-vl:"
# Keep these defaults in sync with /home/dongyicheng/set_qwen9b.sh.
# vLLM requests use --served-model-name rather than MODEL_PATH.
DEFAULT_QWEN_VL_MODEL = "Qwen3.5-9B"
DEFAULT_QWEN_VL_BASE_URL = "http://127.0.0.1:8000/v1"


def is_qwen_vl_model(model: str) -> bool:
    """Return whether a provider-qualified model selects local Qwen-VL."""
    return model.startswith(QWEN_VL_MODEL_PREFIX)


def build_qwen_vl_model(
    model: str,
    *,
    base_url: str | None = None,
) -> OpenAIChatModel:
    """Build a Pydantic AI chat model backed by a Qwen-VL API server.

    The server must implement OpenAI Chat Completions, including ``image_url``
    content blocks and tool calls. The local ``set_qwen9b.sh`` and
    ``set_qwen4b.sh`` launchers provide that interface for Qwen3.5.
    """
    if not is_qwen_vl_model(model):
        raise ValueError(f"Qwen-VL model ids must start with {QWEN_VL_MODEL_PREFIX!r}")

    model_name = (
        model.removeprefix(QWEN_VL_MODEL_PREFIX).strip()
        or os.environ.get("QWEN_VL_MODEL", "").strip()
        or DEFAULT_QWEN_VL_MODEL
    )
    endpoint = _normalize_base_url(
        base_url or os.environ.get("QWEN_VL_BASE_URL") or DEFAULT_QWEN_VL_BASE_URL
    )
    api_key = os.environ.get("QWEN_VL_API_KEY") or "EMPTY"
    # This provider targets a local service. Ignoring HTTP(S)/ALL_PROXY avoids
    # both unnecessary loopback proxying and an optional socksio dependency.
    http_client = DefaultAsyncHttpxClient(trust_env=False)
    provider = OpenAIProvider(
        base_url=endpoint,
        api_key=api_key,
        http_client=http_client,
    )
    return OpenAIChatModel(model_name, provider=provider)


def _normalize_base_url(value: str) -> str:
    """Validate an HTTP endpoint and add vLLM's default ``/v1`` path."""
    endpoint = value.strip().rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            f"Qwen-VL base URL must be an absolute http(s) URL, got {value!r}"
        )
    if parsed.path in {"", "/"}:
        endpoint += "/v1"
    return endpoint
