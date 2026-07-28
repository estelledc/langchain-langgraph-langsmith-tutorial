# 06-langgraph-runtime · LangGraph Runtime

> 状态：`executable`。离线实现与自动化验收均已接入。

把同一任务实现为显式状态图，比较额外复杂度带来的能力。

- 前置：[05-context-engineering](../05-context-engineering/)
- 产物：`LangGraphResearchRuntime`
- 能力：`typed-stategraph`, `conditional-routing`, `runtime-parity`

## 1. Frame

先回答：**哪些状态、分支和恢复需求值得引入自定义图？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
authorize → retrieve → validate_evidence → synthesize → verify → finalize
```

## 3. Build

完成或审查 `LangGraphResearchRuntime`。实现入口：

- [`src/agent_lab/runtimes/graph.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/runtimes/graph.py)
- [`tests/graph/test_graph_runtime.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/graph/test_graph_runtime.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `wrong-route`
- `missing-terminal`
- `citation-contract-failed`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `same-domain-contract`
- `expected-node-sequence`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
