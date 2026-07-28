from __future__ import annotations

from agent_lab.capabilities.memory import (
    InMemoryStore,
    MemoryKind,
    MemoryWrite,
    MemoryWriteStatus,
)


def test_memory_write_requires_approval_and_versions_updates() -> None:
    store = InMemoryStore(require_approval=True)
    request = MemoryWrite(
        namespace=("users", "u-1"),
        key="explanation-style",
        kind=MemoryKind.SEMANTIC,
        value="先讲直觉，再讲定义",
        evidence_ids=("user-statement-1",),
    )
    pending = store.put(request)
    assert pending.status is MemoryWriteStatus.APPROVAL_REQUIRED
    assert store.get(request.namespace, request.key) is None

    first = store.put(request.model_copy(update={"approved_by": "u-1"}))
    second = store.put(
        request.model_copy(update={"value": "先讲例子，再讲定义", "approved_by": "u-1"})
    )
    assert first.status is second.status is MemoryWriteStatus.STORED
    assert first.record and first.record.version == 1
    assert second.record and second.record.version == 2
    assert store.search(("users", "u-1"), "例子") == (second.record,)
    assert store.search(("users", "u-2"), "例子") == ()
