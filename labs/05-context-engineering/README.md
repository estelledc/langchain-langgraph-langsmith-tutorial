# 05-context-engineering · Context Engineering

> 状态：`executable`。离线实现与自动化验收均已接入。

区分模型上下文、工具上下文、runtime context、state 和 store。

- 前置：[04-middleware-budget](../04-middleware-budget/)
- 产物：`RunContext and evidence boundary`
- 能力：`runtime-context`, `state-boundary`, `secret-isolation`

## 1. Frame

先回答：**什么应该进入模型，什么只应该留在工具或运行时？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
validate → authorize
```

## 3. Build

完成或审查 `RunContext and evidence boundary`。实现入口：

- [`src/agent_lab/application/context.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/application/context.py)
- [`docs/adrs/0004-memory-model.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0004-memory-model.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `secret-in-state`
- `untrusted-tool-instruction`
- `context-bloat`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `secrets-not-serialized`
- `permissions-from-context`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
