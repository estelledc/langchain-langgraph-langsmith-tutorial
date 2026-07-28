"""Framework-neutral contracts for reproducible iOS bug repair."""

from agent_lab.repair.domain.approval import ApprovalAction, ApprovalReceipt
from agent_lab.repair.domain.proof import (
    AgentIdentity,
    PatchPassport,
    ProofArtifact,
    build_patch_passport,
    task_contract_hash,
)
from agent_lab.repair.domain.replay import ReplaySpec, RuntimeExpectation, RuntimeRecord
from agent_lab.repair.domain.result import (
    PatchRecord,
    RepairResult,
    RepairStage,
    RepairStatus,
    VerificationRecord,
)
from agent_lab.repair.domain.task import (
    IOSExecutionEnvironment,
    PermissionSpec,
    ProofArtifactKind,
    RepairTask,
    RepositoryKind,
    RepositorySpec,
    ReproductionSpec,
    TaskOrigin,
    VerificationCheck,
    VerificationSpec,
)

__all__ = [
    "AgentIdentity",
    "ApprovalAction",
    "ApprovalReceipt",
    "IOSExecutionEnvironment",
    "PatchPassport",
    "PatchRecord",
    "PermissionSpec",
    "ProofArtifact",
    "ProofArtifactKind",
    "RepairResult",
    "RepairStage",
    "RepairStatus",
    "RepairTask",
    "ReplaySpec",
    "RepositoryKind",
    "RepositorySpec",
    "ReproductionSpec",
    "RuntimeExpectation",
    "RuntimeRecord",
    "TaskOrigin",
    "VerificationCheck",
    "VerificationRecord",
    "VerificationSpec",
    "build_patch_passport",
    "task_contract_hash",
]
