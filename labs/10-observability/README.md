---
layout: default
title: Observability 与 Trace
description: 记录节点、工具、状态和终止原因，不暴露隐藏推理。
---

# 10-observability · Observability 与 Trace

> 状态：`executable`。离线实现与自动化验收均已接入。

记录节点、工具、状态和终止原因，不暴露隐藏推理。

- 前置：[09-memory-engineering](../09-memory-engineering/)
- 产物：`TraceRecorder and RunMetrics`
- 能力：`structured-trace`, `termination-reason`, `operational-metrics`

## 1. Frame

先回答：**看到一条失败 Trace 后，如何定位是工具、路由还是证据问题？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
request_received → tool_started → tool_finished → run_completed
```

## 3. Build

完成或审查 `TraceRecorder and RunMetrics`。实现入口：

- [`src/agent_lab/observability/tracing.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/observability/tracing.py)
- [`src/agent_lab/domain/models.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/domain/models.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `missing-trace-id`
- `hidden-reasoning-leak`
- `ambiguous-termination`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `actions-not-thoughts`
- `ordered-events`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
