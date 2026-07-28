---
layout: default
title: RFC 0002 · XcodeFixBench
---

# RFC 0002：XcodeFixBench

状态：Implemented as executable dev slice

## 决定

在 Agent Engineering Lab 内孵化 XcodeFixBench：一个面向可公开重放的 iOS Bug、用于比较 Coding Agent 修复行为的证据优先基准。

它不和通用 Coding Agent 比“谁更会写代码”。它回答一个更窄的问题：

> 给定固定源码提交、固定 Xcode/Simulator 环境和明确权限，Agent 能否复现、诊断、修改并证明 Bug 已修复？

当前已经具备一个 `synthetic-seeded` Keyboard task、真实 iOS Simulator replay、隔离 Git workspace、审批绑定、Gold/Negative Patch、XCTest/runtime oracle 和 Patch Passport。它仍未通过 20 次稳定性晋级门禁，也没有 held-out 集、多 Agent adapter 或排行榜，因此不能称为成熟 iOS 修复 Benchmark。

## 为什么是独立纵向切片

Trusted Research 已验证只读检索、Evidence、Citation 和离线 Eval，但没有覆盖真实源码写入、构建、Simulator 状态和回滚。XcodeFixBench 使用现有稳定核心，补上这条副作用链：

```text
RepairTask
→ 固定 base commit 与环境
→ 复现证据
→ 根因证据
→ 审批绑定
→ 限定范围 Patch
→ Build / Tests / Runtime Oracle
→ PatchPassport
```

XcodeFixBench 是应用和数据集；`RunRequest / RunResult`、Policy、Evidence、Trace 与 Eval 仍属于 Agent Engineering Lab。领域合同不得依赖 LangGraph、MCP、某个 Agent CLI 或模型提供商。

## 产品边界

首个公开版本需要同时提供：

- 可公开重放的 Bug task；
- 确定性 Gold Patch，用于证明 Harness 自身可用；
- 至少一个必然失败的 Negative Patch；
- 统一 Agent adapter 合同；
- Build、测试、Simulator 与越权检查；
- 机器可读 Patch Passport；
- `PASS / FAIL / UNKNOWN / ERROR` 四态结果。

单个演示视频、一次模型成功、编译通过或手工填写的“测试通过”都不能升级任务状态。

## `RepairTask` 合同

每个任务固定以下输入：

| 分组 | 必填信息 | 目的 |
|---|---|---|
| 来源 | URL、完整 base commit、SPDX license、任务来源类型 | 可追溯且不把合成 Bug 冒充生产事实 |
| 环境 | macOS、Xcode、iOS Runtime、Device、可选 Swift 版本 | 解释环境漂移 |
| 问题 | Bug 报告、可判定 invariant | 把自然语言问题转成验收合同 |
| 复现 | scheme、结构化步骤、所需证据 | 先证明 Bug 存在 |
| 权限 | read/write/forbidden path 与 capability | 限定副作用 |
| 预算 | 模型、工具、步骤、token、成本、deadline | 保持 Agent 对比公平 |
| 验证 | build scheme、测试计划、oracle ID、replay、强制检查 | 决定成功而非解释成功 |

任务中的路径必须是仓库相对 POSIX 路径。任务合同不接受任意 shell 命令；适配器必须将 scheme、test plan 和 replay spec 作为结构化参数传递，不能拼接到 shell 字符串。

## 状态机

```text
PREPARED
→ REPRODUCING
→ DIAGNOSING
→ PLAN_READY
→ APPROVAL_REQUIRED
→ PATCHING
→ BUILDING
→ TESTING
→ VERIFYING_RUNTIME
→ DELIVERED
```

允许的终止状态：

- `approval_required`：尚未获得与当前写动作绑定的审批；
- `policy_blocked`：路径、能力、预算或 payload 不符合合同；
- `build_failed` / `test_failed` / `runtime_verification_failed`：Agent 或 Patch 未满足门禁；
- `unknown`：证据不足，不能判断；
- `infra_error`：Xcode、Simulator、Runner 或 evaluator 自身失败；
- `rolled_back`：已执行副作用被可验证地撤销；
- `delivered`：所有任务要求的强制检查均为 `PASS`。

基础设施失败不得记为 Agent `FAIL`，也不得通过重试后只保留成功 trial 来隐藏。

## 强制验证

每个任务至少要求：

1. 原问题已被复现并留下证据；
2. 根因结论引用源码、日志或结构化运行证据；
3. Patch 从任务的 base commit 生成；
4. 修改文件没有超出 writable scope；
5. 工程构建通过；
6. 现有测试没有被删除、跳过或弱化；
7. 任务专用 Oracle 通过；
8. forbidden path、Oracle、Runner 和基准配置未被修改；
9. 回归检查通过。

UI、布局、生命周期和交互任务还必须执行 Simulator replay。截图只作为证据之一；几何值、Accessibility、日志或 XCTest 等结构化 Oracle 优先。

## Patch Passport

Patch Passport 只代表成功结果，不是所有 trial 的通用报告。它由 `RepairTask + delivered RepairResult` 构建，至少记录：

- task contract hash、base commit 和 run ID；
- Agent、模型、prompt 和工具版本；
- macOS、Xcode、Runtime 与 Device；
- approval receipt ID；
- Patch hash、修改文件和 diff artifact；
- 每个强制检查及其 Evidence/Artifact；
- trace ID、预算消耗和明确限制。

Passport 不接受独立的 `test_status=PASS` 参数。后续 Verification Engine 必须从真实进程退出码、`xcresult`、Simulator replay 和策略收据生成检查记录。

## Threat Model

第一版至少阻断：

| 攻击或错误 | 必须产生的结果 |
|---|---|
| `../`、绝对路径、symlink 逃逸 | `policy_blocked` |
| 修改 writable allowlist 之外文件 | `policy_blocked` |
| 修改 Oracle、golden、Runner 或原测试 | `FAIL` 或 `policy_blocked` |
| shell 参数注入 | 执行前拒绝，不启动子进程 |
| secret 进入日志、Patch 或 Trace | redaction 失败，禁止交付 |
| 审批后替换 diff/payload | 原审批失效 |
| 重放已消费审批 | `policy_blocked` |
| 同一请求重复产生写入 | 幂等门禁失败 |
| source commit 漂移 | `UNKNOWN` 或 `policy_blocked`，不得继续应用旧 Patch |
| 测试失败后宣布成功 | `test_failed`，不得生成 Patch Passport |
| Xcode/Simulator 故障 | `infra_error`，不得折算为 Agent 质量 |

任务本身不得包含凭证、公司代码、内部 URL、真实用户数据或未经筛选的生产 Trace。

## 数据集可信度

任务来源必须标记为：

- `synthetic-seeded`；
- `open-source-history`；
- `real-world-anonymized`。

开发集公开 Task、Gold Patch 与 Oracle，帮助外部作者接入。正式榜单的任务说明公开，但至少保留一层未暴露的 task oracle，并由受控 Runner 执行。两者不能混用同一个分数。

任何任务进入正式集合前都要满足：

- 固定环境重复运行；
- Gold Patch 稳定通过；
- Broken/Negative Patch 稳定失败；
- 基准作弊检查有效；
- 基础设施错误率低于独立阈值；
- license 和来源可公开审计。

## 分阶段交付

### PR 1：Flagship Contract

- 本 RFC；
- `RepairTask`、`RepairResult`、`PatchPassport`；
- 路径、审批、commit、检查和 proof bundle 不变量测试；
- `apps/xcode_fixbench/` 孵化入口。

### PR 2：Deterministic Xcode Verification

- 隔离 worktree；
- Native `xcodebuild` adapter；
- `xcresult` 解析；
- Patch scope、build、test grader；
- 一个 Gold Patch 和一个 Negative Patch。

### PR 3：Simulator Proof

- Device port；
- 可重复 replay；
- Screenshot、Accessibility、日志与结构化 runtime oracle；
- 完整 Patch Passport 产物。

上述三个阶段已在同一开发分支形成第一条可执行纵向切片。进入多 Agent、排行榜、云 Mac、Team Memory、A2A 或自动 Draft PR 前，仍须先满足下述第一任务晋级条件。

## 第一任务晋级条件

首个 Keyboard Layout task 只有同时满足以下条件，才从 fixture 候选晋级为 executable：

- 在固定环境连续重放至少 20 次，原 Bug 可稳定触发；
- Gold Patch 20/20 通过；
- 至少两个看似合理但错误的 Patch 20/20 被拒绝；
- 无审批不能写，修改 payload 后旧审批失效；
- build/test/replay 任一失败都不能生成 Patch Passport；
- Runner 故障被记为 `ERROR`，不记为 Agent `FAIL`；
- 所有 proof artifact 均有 hash，且能从 base commit 重建。

重复次数是任务稳定性门禁，不是对概率 Agent 成功率的承诺。

## 非目标

- 不开发新 IDE 或基础模型；
- 不重新实现完整 XcodeBuildMCP、Appium 或设备控制框架；
- 不默认允许 push、merge、生产发布或宿主机无限制执行；
- 不同时扩展 Android、Web 和 Desktop；
- 不用总分掩盖任何强制门禁失败；
- 不把 synthetic task、Gold Patch 或本地 fixture 描述成真实生产修复。

## 回滚与兼容

repair package 通过独立 `xcodefix` CLI 接入，不改变 Trusted Research 的 `RunRequest / RunResult` 或默认 runtime。删除该 CLI 和 task corpus 不应改变 Agent Lab 的离线研究行为。`RepairTask` 与 `PatchPassport` 已采用版本化 schema，后续迁移不得静默修改既有 task 或 passport 的语义。
