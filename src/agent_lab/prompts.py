"""Versioned prompt resources for optional model runtimes."""

from importlib.resources import files

TRUSTED_RESEARCH_PROMPT_VERSION = "trusted-research-system-v1"


def trusted_research_system_prompt() -> str:
    return (
        files("agent_lab.data")
        .joinpath("prompts/trusted_research_system_v1.txt")
        .read_text(encoding="utf-8")
        .strip()
    )
