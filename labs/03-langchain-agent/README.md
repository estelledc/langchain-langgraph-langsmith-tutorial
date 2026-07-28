# 03-langchain-agent · LangChain create_agent

> 状态：`implemented`。实现已存在；涉及外部模型的行为仍需单独 live 验证。

在确定性权限外壳中接入标准模型与工具循环，并要求结构化引用。

- 前置：[02-tool-contracts](../02-tool-contracts/)
- 产物：`LangChainResearchRuntime`
- 能力：`create-agent`, `structured-output`, `tool-loop`

## 1. Frame

先回答：**简单工具循环是否已足够，还是确实需要自定义图？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
validate → agent → verify
```

## 3. Build

完成或审查 `LangChainResearchRuntime`。实现入口：

- [`src/agent_lab/runtimes/langchain_agent.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/runtimes/langchain_agent.py)
- [`tests/security/test_security_boundaries.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/tests/security/test_security_boundaries.py)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `model-budget-zero`
- `provider-error`
- `invalid-citation`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `model-is-injected`
- `provider-error-is-not-score`
- `tool-output-untrusted`

关联 suite：`live-provider-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
