# f08-a2a · A2A

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

只在跨团队、跨服务和远程任务生命周期确实存在时采用 A2A。

- 前置：[f07-mcp](../f07-mcp/)
- 产物：`remote-agent protocol experiment`
- 能力：`a2a`, `remote-task-lifecycle`

## 1. Frame

先回答：**这是远程 Agent 协作，还是一个进程内的普通函数调用？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
submit → stream → artifact → complete
```

## 3. Build

完成或审查 `remote-agent protocol experiment`。实现入口：

- [`docs/adrs/0001-framework-boundaries.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0001-framework-boundaries.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `lost-correlation`
- `duplicate-task`
- `trust-mismatch`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `remote-boundary-real`
- `trace-correlated`

关联 suite：`frontier-a2a-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
