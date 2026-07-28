---
layout: default
title: ADR 0001 · Framework boundaries
---

# ADR 0001：框架边界

状态：Accepted

## 决定

依赖方向固定为：

```text
domain ← application ← capabilities ← runtimes/adapters
```

- `domain` 只使用标准库与 Pydantic。
- `application` 编排领域合同，不感知模型提供商。
- `capabilities` 定义工具、检索、记忆和审批端口。
- `runtimes` 适配 Workflow、LangChain、LangGraph 与 Deep Agents。
- `evaluation` 读取统一 `RunResult`，不依赖某个 tracing 平台。

## 原因

框架 API 会变化，任务合同、失败样例和回归数据不应随之报废。LangChain 用于标准工具循环；LangGraph 用于显式状态、恢复、分支和 HITL；Deep Agents 仅在长任务、上下文隔离或文件系统能力有可测收益时进入实验轨。

## 约束

- 核心包不得 import `deepagents` 或具体 provider。
- optional adapter 必须延迟 import，并在缺依赖时给出可操作错误。
- 同一任务的不同 runtime 必须接受同一个 `RunRequest` 并返回 `RunResult`。
