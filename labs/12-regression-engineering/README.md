---
layout: default
title: Regression Engineering
description: 把 unknown、注入、越权和预算失败固化为版本化数据集。
---

# 12-regression-engineering · Regression Engineering

> 状态：`executable`。离线实现与自动化验收均已接入。

把 unknown、注入、越权和预算失败固化为版本化数据集。

- 前置：[11-agent-evals](../11-agent-evals/)
- 产物：`capability and regression datasets`
- 能力：`capability-dataset`, `regression-dataset`, `strict-gate`

## 1. Frame

先回答：**一次修复如何变成以后永远会被检查的资产？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
case → runtime → trial → graders → gate
```

## 3. Build

完成或审查 `capability and regression datasets`。实现入口：

- [`datasets/capability/trusted-research-v1.jsonl`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/datasets/capability/trusted-research-v1.jsonl)
- [`datasets/regression/trusted-research-v1.jsonl`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/datasets/regression/trusted-research-v1.jsonl)
- [`evals/suites/fast.yaml`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/evals/suites/fast.yaml)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `duplicate-case-id`
- `dataset-drift`
- `evaluator-error`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `workflow-and-graph-pass`
- `error-rate-zero`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
