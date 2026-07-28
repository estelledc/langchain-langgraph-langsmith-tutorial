# f01-filesystem-context · Filesystem Context

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

研究何时用文件系统卸载上下文，并测试权限和信息召回。

- 前置：[05-context-engineering](../05-context-engineering/)
- 产物：`experiment contract`
- 能力：`filesystem-context`

## 1. Frame

先回答：**文件系统比长 Prompt 多带来了什么可测收益？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
write-note → read-note → synthesize
```

## 3. Build

完成或审查 `experiment contract`。实现入口：

- [`docs/adrs/0001-framework-boundaries.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0001-framework-boundaries.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `path-escape`
- `stale-note`
- `secret-write`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `sandbox-only`
- `recall-measured`

关联 suite：`frontier-filesystem-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
