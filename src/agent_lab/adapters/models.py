"""Lazy provider constructors keep API credentials out of the core path."""

from __future__ import annotations

from typing import Any


def openai_compatible_model(
    *,
    model: str,
    api_key: str,
    base_url: str | None = None,
    temperature: float = 0,
) -> Any:
    """Build an OpenAI-compatible chat model when the optional extra is installed."""

    try:
        from langchain_openai import ChatOpenAI  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "provider-openai 未安装；运行 `uv sync --extra provider-openai`"
        ) from exc

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
    )
