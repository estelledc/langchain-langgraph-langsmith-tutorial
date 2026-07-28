# XcodeFixBench

面向可公开重放的 iOS Bug 的证据优先修复基准。

当前状态：`executable-dev-slice`。仓库已有一个可运行的 synthetic Keyboard task 和真实 iOS Simulator replay；还没有多 Agent 排行榜、真机第三方输入法证明或线上修复能力。

已落地：

- [`RFC 0002`](../../docs/rfcs/0002-xcode-fix-bench.md)：产品范围、状态机、威胁模型和晋级门禁；
- [`RepairTask`](../../src/agent_lab/repair/domain/task.py)：固定源码、环境、复现、权限、预算与验证合同；
- [`RepairResult`](../../src/agent_lab/repair/domain/result.py)：区分修复失败、策略阻断、未知和基础设施错误；
- [`PatchPassport`](../../src/agent_lab/repair/domain/proof.py)：只从 delivered result 构建的机器可读成功证据。
- [`keyboard-layout-001`](../../benchmarks/ios-repair/dev/keyboard-layout-001/)：固定 Git 快照、Gold/Negative Patch、XCTest Oracle 与 Simulator runtime oracle。

合同与执行层测试：

```bash
uv run pytest tests/unit/test_repair_contracts.py tests/unit/test_repair_execution.py
```

目标流程：

```text
固定 Task + base commit
→ 复现
→ 根因证据
→ 审批绑定
→ 隔离 worktree 修改
→ Build / Tests / Simulator Oracle
→ Patch Passport
```

完整 Gold 路径：

```bash
uv run xcodefix run --task keyboard-layout-001 --candidate gold --approve-patch
```

省略 `--approve-patch` 时，流程必须停在 `approval_required`，不会写入隔离 workspace。Negative Patch 用于确认 Harness 会拒绝只适配展示数值、无法泛化到其他 guide 高度的修复。

证据边界：当前只证明一个公开 synthetic task 的本机 Xcode/Simulator 全链路；尚不能宣传为成熟公开 Benchmark。
