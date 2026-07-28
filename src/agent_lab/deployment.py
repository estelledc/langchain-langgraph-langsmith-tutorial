"""Agent Server graph blueprint.

Agent Server injects its own persistence. Local application code may compile the same
runtime with an explicit checkpointer instead.
"""

from agent_lab.capabilities.retrieval import FixtureSearchAdapter
from agent_lab.runtimes.graph import LangGraphResearchRuntime

graph = LangGraphResearchRuntime(FixtureSearchAdapter(), checkpointer=None).graph
