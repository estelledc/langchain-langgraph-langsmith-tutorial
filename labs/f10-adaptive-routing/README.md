# f10-adaptive-routing · Adaptive Routing

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

用质量、成本和延迟数据决定模型或架构路由，不凭感觉升级复杂度。

- 前置：[f09-online-evals](../f09-online-evals/)
- 产物：`routing benchmark`
- 能力：`adaptive-routing`, `cost-quality-tradeoff`

## 1. Frame

先回答：**路由策略在哪些分布变化下会失效？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
classify → route → run → compare
```

## 3. Build

完成或审查 `routing benchmark`。实现入口：

- [`docs/adrs/0003-evaluation-semantics.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0003-evaluation-semantics.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `distribution-shift`
- `cost-spike`
- `silent-quality-drop`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `baseline-comparison`
- `rollback-threshold`

关联 suite：`frontier-routing-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
