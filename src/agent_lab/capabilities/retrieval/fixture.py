"""Deterministic retrieval adapter backed by package fixtures."""

from __future__ import annotations

import json
import re
from datetime import datetime
from importlib.resources import files
from typing import Any

from agent_lab.capabilities.retrieval.models import (
    FIXTURE_SEARCH_SPEC,
    SearchQuery,
    SearchResult,
    SearchStatus,
)
from agent_lab.capabilities.tools.contracts import ToolErrorCode, ToolSpec
from agent_lab.domain.models import Evidence, SourceType, TrustLevel


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9_-]{2,}", lowered))
    cjk = "".join(re.findall(r"[\u4e00-\u9fff]", lowered))
    cjk_tokens = {cjk[index : index + 2] for index in range(max(0, len(cjk) - 1))}
    return ascii_tokens | cjk_tokens


class FixtureSearchAdapter:
    """Search a small, versioned corpus without network access."""

    spec: ToolSpec = FIXTURE_SEARCH_SPEC

    def __init__(self, fixture_name: str = "trusted_research.json") -> None:
        fixture_path = files("agent_lab.data").joinpath(fixture_name)
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.version = str(payload["fixture_version"])
        self.documents: tuple[dict[str, Any], ...] = tuple(payload["documents"])

    def search(self, request: SearchQuery, *, call_id: str) -> SearchResult:
        query_tokens = _tokens(request.query)
        ranked: list[tuple[int, int, dict[str, Any]]] = []
        query_lower = request.query.lower()

        for document in self.documents:
            keywords = [str(item).lower() for item in document.get("keywords", [])]
            keyword_score = sum(4 for keyword in keywords if keyword in query_lower)
            token_score = len(query_tokens & _tokens(f"{document['title']} {document['content']}"))
            score = keyword_score + token_score
            if keyword_score > 0 or token_score >= 2:
                ranked.append((score, keyword_score, document))

        if any(keyword_score > 0 for _, keyword_score, _ in ranked):
            ranked = [item for item in ranked if item[1] > 0]
        ranked.sort(key=lambda item: (-item[0], str(item[2]["id"])))
        selected = ranked[: request.limit]
        if not selected:
            return SearchResult(
                status=SearchStatus.NOT_FOUND,
                error_code=ToolErrorCode.NOT_FOUND,
                message="离线 fixture 中没有支持该问题的证据",
                fixture_version=self.version,
            )

        evidence = tuple(
            self._to_evidence(document, call_id=call_id) for _, _, document in selected
        )
        return SearchResult(
            status=SearchStatus.FOUND,
            evidence=evidence,
            message=f"从离线 fixture 找到 {len(evidence)} 条候选证据",
            fixture_version=self.version,
        )

    def _to_evidence(self, document: dict[str, Any], *, call_id: str) -> Evidence:
        return Evidence.from_content(
            evidence_id=str(document["id"]),
            source_type=SourceType.FIXTURE,
            source_uri=str(document["source_uri"]),
            title=str(document["title"]),
            content=str(document["content"]),
            observed_at=datetime.fromisoformat(str(document["observed_at"])),
            valid_at=(
                datetime.fromisoformat(str(document["valid_at"]))
                if document.get("valid_at")
                else None
            ),
            tool_call_id=call_id,
            trust_level=TrustLevel.SYNTHETIC,
            untrusted=True,
            metadata={
                "fixture_version": self.version,
                "contains_injection": str(bool(document.get("contains_injection", False))).lower(),
            },
        )
