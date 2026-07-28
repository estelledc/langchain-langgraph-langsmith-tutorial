# f09-online-evals · Online Evals

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

对真实流量设置采样、成本和隐私策略，并把失败回流离线数据集。

- 前置：[14-production-loop](../14-production-loop/)
- 产物：`online evaluation policy`
- 能力：`online-eval`, `sampling`, `alerting`

## 1. Frame

先回答：**哪些线上信号足以触发告警，哪些需要人工复核？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
sample → evaluate → alert → promote
```

## 3. Build

完成或审查 `online evaluation policy`。实现入口：

- [`docs/adrs/0003-evaluation-semantics.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0003-evaluation-semantics.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `biased-sample`
- `privacy-leak`
- `judge-outage`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `sampling-explicit`
- `errors-separated`

关联 suite：`frontier-online-eval-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
