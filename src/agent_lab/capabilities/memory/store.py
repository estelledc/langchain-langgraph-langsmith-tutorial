"""A small memory port that does not confuse chat history with durable knowledge."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import Field

from agent_lab.domain.models import FrozenModel, utc_now


class MemoryKind(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"


class MemoryWriteStatus(StrEnum):
    STORED = "stored"
    APPROVAL_REQUIRED = "approval_required"
    REJECTED = "rejected"


class MemoryRecord(FrozenModel):
    memory_id: str = Field(default_factory=lambda: str(uuid4()))
    namespace: tuple[str, ...]
    key: str = Field(min_length=1)
    kind: MemoryKind
    value: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    version: int = Field(default=1, ge=1)
    created_at: datetime = Field(default_factory=utc_now)
    approved_by: str | None = None


class MemoryWrite(FrozenModel):
    namespace: tuple[str, ...]
    key: str = Field(min_length=1)
    kind: MemoryKind
    value: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = ()
    approved_by: str | None = None


class MemoryWriteResult(FrozenModel):
    status: MemoryWriteStatus
    message: str
    record: MemoryRecord | None = None


class MemoryStore(Protocol):
    def put(self, request: MemoryWrite) -> MemoryWriteResult: ...

    def get(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None: ...

    def search(self, namespace: tuple[str, ...], query: str) -> tuple[MemoryRecord, ...]: ...


class InMemoryStore:
    """Deterministic teaching store; process-local and not a production database."""

    def __init__(self, *, require_approval: bool = True) -> None:
        self.require_approval = require_approval
        self._records: dict[tuple[tuple[str, ...], str], MemoryRecord] = {}

    def put(self, request: MemoryWrite) -> MemoryWriteResult:
        if self.require_approval and not request.approved_by:
            return MemoryWriteResult(
                status=MemoryWriteStatus.APPROVAL_REQUIRED,
                message="长期记忆写入需要 approved_by",
            )
        identity = (request.namespace, request.key)
        previous = self._records.get(identity)
        record = MemoryRecord(
            namespace=request.namespace,
            key=request.key,
            kind=request.kind,
            value=request.value,
            evidence_ids=request.evidence_ids,
            version=(previous.version + 1 if previous else 1),
            approved_by=request.approved_by,
        )
        self._records[identity] = record
        return MemoryWriteResult(
            status=MemoryWriteStatus.STORED,
            message="记忆已按 namespace/key 存储",
            record=record,
        )

    def get(self, namespace: tuple[str, ...], key: str) -> MemoryRecord | None:
        return self._records.get((namespace, key))

    def search(self, namespace: tuple[str, ...], query: str) -> tuple[MemoryRecord, ...]:
        needle = query.casefold()
        return tuple(
            record
            for (record_namespace, _), record in sorted(
                self._records.items(), key=lambda item: item[0][1]
            )
            if record_namespace == namespace
            and (needle in record.key.casefold() or needle in record.value.casefold())
        )
