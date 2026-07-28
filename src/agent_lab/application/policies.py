"""Deterministic permission, evidence and approval policies."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass

from agent_lab.capabilities.tools.contracts import SideEffect, ToolSpec
from agent_lab.domain.models import Citation, Evidence, RunRequest

INJECTION_PATTERNS = (
    re.compile(r"ignore (all |any )?(previous|prior) (instructions|rules)", re.IGNORECASE),
    re.compile(r"忽略(之前|以上|所有).{0,12}(指令|规则)"),
    re.compile(r"(reveal|泄露).{0,12}(system prompt|系统提示)", re.IGNORECASE),
    re.compile(r"<\|(?:system|assistant|developer)\|>", re.IGNORECASE),
)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    allowed: bool
    reason: str
    approval_required: bool = False


class ToolPolicy:
    def authorize(
        self,
        *,
        request: RunRequest,
        spec: ToolSpec,
        permissions: Collection[str],
    ) -> PolicyDecision:
        if spec.capability not in request.allowed_capabilities:
            return PolicyDecision(False, f"请求未允许 capability: {spec.capability}")
        missing = spec.required_permissions - frozenset(permissions)
        if missing:
            return PolicyDecision(False, f"缺少权限: {', '.join(sorted(missing))}")
        if spec.side_effect in {SideEffect.WRITE, SideEffect.EXTERNAL_WRITE}:
            return PolicyDecision(False, "副作用工具需要人工审批", approval_required=True)
        return PolicyDecision(True, "policy allow")


def contains_prompt_injection(evidence: Evidence) -> bool:
    if evidence.metadata.get("contains_injection") == "true":
        return True
    return any(pattern.search(evidence.content) for pattern in INJECTION_PATTERNS)


def safe_evidence(evidence: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
    """Quarantine obvious instruction-shaped fixture content.

    This filter is an executable teaching contract, not a complete injection defense.
    """

    return tuple(item for item in evidence if not contains_prompt_injection(item))


def validate_citations(citations: tuple[Citation, ...], evidence: tuple[Evidence, ...]) -> bool:
    known = {item.evidence_id for item in evidence}
    return bool(citations) and all(set(item.evidence_ids) <= known for item in citations)
