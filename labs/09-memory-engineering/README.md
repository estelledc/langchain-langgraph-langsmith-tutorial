# 09-memory-engineering · Memory Engineering

> 状态：`executable`。离线实现与自动化验收均已接入。

把 thread state 与 semantic、episodic、procedural memory 分开。

- 前置：[08-evidence-first-rag](../08-evidence-first-rag/)
- 产物：`approval-aware memory store`
- 能力：`semantic-memory`, `episodic-memory`, `procedural-memory`, `versioned-write`

## 1. Frame

先回答：**哪些聊天内容有资格升级为长期记忆？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
approval → memory-write
```

## 3. Build

完成或审查 `approval-aware memory store`。实现入口：

- [`src/agent_lab/capabilities/memory/store.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/capabilities/memory/store.py)
- [`tests/unit/test_memory.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/unit/test_memory.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `unapproved-write`
- `namespace-leak`
- `stale-version`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `approval-required`
- `namespace-isolated`
- `updates-versioned`

关联 suite：`memory-contract-tests`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
