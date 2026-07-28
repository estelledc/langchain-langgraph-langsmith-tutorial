---
layout: default
title: ADR 0002 · Domain contracts
---

# ADR 0002：领域合同

状态：Accepted

## 决定

运行边界使用 `RunRequest → RunResult`。中间和最终工件必须通过以下对象表达：

- `Budget`：模型、工具、步骤、时间和证据上限。
- `Evidence`：来源类型、内容哈希、观察时间、可信等级和工具调用 ID。
- `Citation`：可核验 claim 与 Evidence ID 的绑定。
- `Artifact`：答案以外的可交付物及其证据依赖。
- `TraceEvent`：动作和状态变化，不记录隐藏推理。
- `AgentError`：错误码、归属层和是否可重试。

## 不变量

- `completed` 必须有答案和终止原因。
- 每个 citation 必须至少引用一条存在的 evidence。
- 内容哈希由规范化 UTF-8 内容计算，不接受调用方伪造。
- `not_found`、`unauthorized`、`timeout` 与系统异常不得混成自然语言成功结果。
- secret、连接对象和 provider client 只存在于 runtime context，不进入 state、trace 或 artifact。
