# 08-evidence-first-rag · Evidence-first Retrieval

> 状态：`executable`。离线实现与自动化验收均已接入。

搜索结果先成为结构化 Evidence，再由 writer 生成带 ID 的结论。

- 前置：[07-persistence-hitl](../07-persistence-hitl/)
- 产物：`versioned fixture search`
- 能力：`typed-evidence`, `citation-contract`, `abstention`

## 1. Frame

先回答：**如何证明答案真由检索证据支持，而不是模型顺手编的？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
retrieve → validate_evidence → synthesize → verify
```

## 3. Build

完成或审查 `versioned fixture search`。实现入口：

- [`src/agent_lab/capabilities/retrieval/fixture.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/capabilities/retrieval/fixture.py)
- [`src/agent_lab/data/trusted_research.json`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/data/trusted_research.json)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `not-found`
- `stale-evidence`
- `unsupported-citation`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `fixture-not-web`
- `no-evidence-no-fact`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
