---
layout: default
title: Production Trace → Dataset → Regression
description: 只把经筛选和脱敏的真实失败 Trace 提升为离线回归样例。
---

# 14-production-loop · Production Trace → Dataset → Regression

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

只把经筛选和脱敏的真实失败 Trace 提升为离线回归样例。

- 前置：[13-deployment](../13-deployment/)
- 产物：`production promotion contract`
- 能力：`trace-backtest`, `promotion-review`, `privacy-filter`

## 1. Frame

先回答：**生产流量里的异常如何安全、可追溯地回流？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
observe → triage → redact → review → promote → regress
```

## 3. Build

完成或审查 `production promotion contract`。实现入口：

- [`docs/verification.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/verification.md)
- [`docs/adrs/0003-evaluation-semantics.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0003-evaluation-semantics.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `raw-trace-committed`
- `fixture-promoted-as-production`
- `missing-owner-review`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `promotion-is-reviewed`
- `source-tier-preserved`

关联 suite：`production-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
