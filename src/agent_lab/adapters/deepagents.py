"""Optional Deep Agents factory; the core package never imports it eagerly."""

from __future__ import annotations

from typing import Any


def create_experimental_deep_agent(
    *,
    model: Any,
    tools: list[Any],
    system_prompt: str,
    subagents: list[Any] | None = None,
) -> Any:
    """Create a harness only for experiments with an explicit model and tool set."""

    try:
        from deepagents import create_deep_agent  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("experimental 未安装；运行 `uv sync --extra experimental`") from exc

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        subagents=subagents or [],
    )
