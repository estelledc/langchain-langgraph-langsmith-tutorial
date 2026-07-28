from __future__ import annotations

from agent_lab.application.policies import safe_evidence
from agent_lab.capabilities.retrieval import FixtureSearchAdapter, SearchQuery, SearchStatus
from agent_lab.capabilities.tools.contracts import ToolErrorCode
from agent_lab.domain.models import SourceType, TrustLevel


def test_fixture_search_is_explicitly_offline(search: FixtureSearchAdapter) -> None:
    result = search.search(SearchQuery(query="LangSmith eval"), call_id="call-1")
    assert result.status is SearchStatus.FOUND
    assert result.fixture_version == "trusted-research-fixture-v1"
    assert result.evidence
    assert all(item.source_type is SourceType.FIXTURE for item in result.evidence)
    assert all(item.trust_level is TrustLevel.SYNTHETIC for item in result.evidence)
    assert all(item.tool_call_id == "call-1" for item in result.evidence)


def test_unknown_query_returns_not_found(search: FixtureSearchAdapter) -> None:
    result = search.search(SearchQuery(query="量子香蕉编译器 2039"), call_id="call-2")
    assert result.status is SearchStatus.NOT_FOUND
    assert result.error_code is ToolErrorCode.NOT_FOUND
    assert result.evidence == ()


def test_keyword_hit_does_not_mix_incidental_documents(search: FixtureSearchAdapter) -> None:
    result = search.search(SearchQuery(query="injection-test"), call_id="call-3")
    assert [item.evidence_id for item in result.evidence] == ["fx-injection-adversarial-v1"]
    assert safe_evidence(result.evidence) == ()
