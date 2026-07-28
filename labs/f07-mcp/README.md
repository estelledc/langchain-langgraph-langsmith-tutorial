# f07-mcp · MCP Adapter

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

把 MCP 放在内部 Tool Contract 之外的边界适配层。

- 前置：[f06-sandboxed-code](../f06-sandboxed-code/)
- 产物：`MCP boundary adapter`
- 能力：`mcp-adapter`, `external-tools`

## 1. Frame

先回答：**协议便利性如何不反向污染领域模型？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
discover → authorize → invoke → normalize
```

## 3. Build

完成或审查 `MCP boundary adapter`。实现入口：

- [`docs/adrs/0001-framework-boundaries.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0001-framework-boundaries.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `schema-drift`
- `untrusted-server`
- `permission-escalation`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `internal-contract-stable`
- `server-trust-explicit`

关联 suite：`frontier-mcp-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
