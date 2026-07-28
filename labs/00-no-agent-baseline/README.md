---
layout: default
title: Agent、Workflow 与普通程序
description: 先做确定性基线，再判断任务是否需要模型和工具循环。
---

# 00-no-agent-baseline · Agent、Workflow 与普通程序

> 状态：`executable`。离线实现与自动化验收均已接入。

先做确定性基线，再判断任务是否需要模型和工具循环。

- 前置：无
- 产物：`TrustedResearchWorkflow`
- 能力：`deterministic-baseline`, `bounded-execution`

## 1. Frame

先回答：**这个问题是否根本不需要 Agent？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
validate → authorize → retrieve → validate_evidence → synthesize → verify → finalize
```

## 3. Build

完成或审查 `TrustedResearchWorkflow`。实现入口：

- [`src/agent_lab/runtimes/workflow.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/runtimes/workflow.py)
- [`tests/integration/test_workflow.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/integration/test_workflow.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `evidence-not-found`
- `permission-denied`
- `budget-exhausted`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `offline-without-api-key`
- `unknown-abstains`
- `citations-reference-evidence`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
