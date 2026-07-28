# 07-persistence-hitl · Persistence、Interrupt 与 Resume

> 状态：`executable`。离线实现与自动化验收均已接入。

注入 checkpointer，按 thread 保存状态，并通过 interrupt 暂停审批。

- 前置：[06-langgraph-runtime](../06-langgraph-runtime/)
- 产物：`approval graph and injected checkpointer`
- 能力：`checkpoint`, `interrupt`, `resume`, `hitl`

## 1. Frame

先回答：**相同 thread_id 在什么条件下才真的能续接？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
request_approval → interrupt → apply_decision
```

## 3. Build

完成或审查 `approval graph and injected checkpointer`。实现入口：

- [`src/agent_lab/runtimes/approval.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/runtimes/approval.py)
- [`tests/graph/test_graph_runtime.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/graph/test_graph_runtime.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `new-checkpointer-loses-state`
- `wrong-thread-resume`
- `non-idempotent-side-effect`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `same-thread-resumes`
- `approval-precedes-side-effect`

关联 suite：`graph-contract-tests`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
