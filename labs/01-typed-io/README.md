---
layout: default
title: Typed I/O 与结构化结果
description: 用 RunRequest、Budget、Evidence 和 RunResult 把成功与失败说清楚。
---

# 01-typed-io · Typed I/O 与结构化结果

> 状态：`executable`。离线实现与自动化验收均已接入。

用 RunRequest、Budget、Evidence 和 RunResult 把成功与失败说清楚。

- 前置：[00-no-agent-baseline](../00-no-agent-baseline/)
- 产物：`framework-neutral domain contracts`
- 能力：`typed-request`, `typed-outcome`, `evidence-hash`

## 1. Frame

先回答：**哪些状态应该由代码拒绝，而不是让模型猜？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
validate
```

## 3. Build

完成或审查 `framework-neutral domain contracts`。实现入口：

- [`src/agent_lab/domain/models.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/domain/models.py)
- [`tests/unit/test_domain.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/unit/test_domain.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `missing-answer`
- `missing-evidence-id`
- `duplicate-evidence-id`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `invalid-contract-fails`
- `completed-requires-answer`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
