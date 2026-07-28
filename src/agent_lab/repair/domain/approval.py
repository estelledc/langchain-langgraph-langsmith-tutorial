"""Approval receipts that bind one authorization to one immutable patch payload."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Self
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from agent_lab.domain.models import FrozenModel, utc_now
from agent_lab.repair.domain.task import scope_contains, validate_relative_path


class ApprovalAction(StrEnum):
    APPLY_PATCH = "git.apply_patch"
    CREATE_COMMIT = "git.create_commit"


def payload_sha256(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class ApprovalReceipt(FrozenModel):
    schema_version: str = "approval-receipt-v1"
    approval_id: str = Field(default_factory=lambda: str(uuid4()), min_length=1)
    actor_id: str = Field(min_length=1)
    action: ApprovalAction
    payload_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    scope: tuple[str, ...] = Field(min_length=1)
    issued_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime
    idempotency_key: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    consumed_at: datetime | None = None

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(validate_relative_path(item, label="approval scope") for item in value)
        if len(set(normalized)) != len(normalized):
            raise ValueError("approval scope must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_lifetime(self) -> Self:
        if self.issued_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("approval timestamps must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise ValueError("approval must expire after it is issued")
        if self.consumed_at is not None:
            if self.consumed_at.tzinfo is None:
                raise ValueError("consumed_at must be timezone-aware")
            if self.consumed_at < self.issued_at:
                raise ValueError("approval cannot be consumed before issuance")
        return self

    @classmethod
    def issue(
        cls,
        *,
        actor_id: str,
        action: ApprovalAction,
        payload: bytes,
        scope: tuple[str, ...],
        ttl: timedelta = timedelta(minutes=15),
        now: datetime | None = None,
    ) -> Self:
        issued_at = now or utc_now()
        payload_hash = payload_sha256(payload)
        normalized_scope = tuple(sorted(scope))
        material = "\0".join((action, payload_hash, *normalized_scope)).encode("utf-8")
        return cls(
            actor_id=actor_id,
            action=action,
            payload_hash=payload_hash,
            scope=normalized_scope,
            issued_at=issued_at,
            expires_at=issued_at + ttl,
            idempotency_key=payload_sha256(material),
        )

    def authorize(
        self,
        *,
        action: ApprovalAction,
        payload: bytes,
        paths: tuple[str, ...],
        now: datetime | None = None,
    ) -> None:
        observed_at = now or utc_now()
        if self.consumed_at is not None:
            raise ValueError("approval receipt has already been consumed")
        if observed_at >= self.expires_at:
            raise ValueError("approval receipt has expired")
        if action != self.action:
            raise ValueError("approval action does not match")
        if payload_sha256(payload) != self.payload_hash:
            raise ValueError("approval payload hash does not match")
        for path in paths:
            normalized = validate_relative_path(path, label="approved path")
            if not any(scope_contains(pattern, normalized) for pattern in self.scope):
                raise ValueError(f"path is outside approval scope: {normalized}")

    def consume(self, *, now: datetime | None = None) -> Self:
        if self.consumed_at is not None:
            raise ValueError("approval receipt has already been consumed")
        consumed_at = now or utc_now()
        if consumed_at >= self.expires_at:
            raise ValueError("approval receipt has expired")
        return self.model_copy(update={"consumed_at": consumed_at})
