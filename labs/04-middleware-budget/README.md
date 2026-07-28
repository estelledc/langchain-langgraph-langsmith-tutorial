# 04-middleware-budget · Middleware、Policy 与预算

> 状态：`executable`。离线实现与自动化验收均已接入。

用代码层 middleware 和 policy 限制模型、工具、步骤与权限。

- 前置：[03-langchain-agent](../03-langchain-agent/)
- 产物：`ToolPolicy and call-limit middleware`
- 能力：`policy-shell`, `model-budget`, `tool-budget`

## 1. Frame

先回答：**哪些约束放在 Prompt 里仍然不够？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
authorize → agent
```

## 3. Build

完成或审查 `ToolPolicy and call-limit middleware`。实现入口：

- [`src/agent_lab/application/policies.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/application/policies.py)
- [`src/agent_lab/runtimes/langchain_agent.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/runtimes/langchain_agent.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `missing-permission`
- `tool-budget-zero`
- `capability-not-allowed`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `denied-before-tool-call`
- `budgets-are-deterministic`

关联 suite：`trusted-research-fast-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
