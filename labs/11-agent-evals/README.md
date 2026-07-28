# 11-agent-evals · Final、Step 与 Trajectory Eval

> 状态：`executable`。离线实现与自动化验收均已接入。

先用确定性 grader 检查状态、证据、工具和轨迹，再考虑语义 Judge。

- 前置：[10-observability](../10-observability/)
- 产物：`deterministic graders`
- 能力：`final-eval`, `step-eval`, `trajectory-eval`, `attribution-eval`

## 1. Frame

先回答：**哪些失败可以被代码百分之百判定？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
run-case → grade → aggregate
```

## 3. Build

完成或审查 `deterministic graders`。实现入口：

- [`src/agent_lab/evaluation/graders.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/evaluation/graders.py)
- [`tests/integration/test_evaluation.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/integration/test_evaluation.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `grader-error`
- `missing-expected-trace`
- `false-average`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `four-state-semantics`
- `unknown-error-unscored`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
