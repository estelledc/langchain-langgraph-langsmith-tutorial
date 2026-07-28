# 02-tool-contracts · 工具合同与错误语义

> 状态：`executable`。离线实现与自动化验收均已接入。

把工具当正式 API，显式声明参数、权限、副作用、幂等与错误。

- 前置：[01-typed-io](../01-typed-io/)
- 产物：`fixture search and safe calculator`
- 能力：`typed-tools`, `error-taxonomy`, `safe-arithmetic`

## 1. Frame

先回答：**NOT_FOUND、TIMEOUT 和系统异常为什么不能返回同一句自然语言？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
authorize → tool-call
```

## 3. Build

完成或审查 `fixture search and safe calculator`。实现入口：

- [`src/agent_lab/capabilities/tools/contracts.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/capabilities/tools/contracts.py)
- [`src/agent_lab/capabilities/tools/calculator.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/capabilities/tools/calculator.py)
- [`datasets/contracts/tool-contract-v1.jsonl`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/datasets/contracts/tool-contract-v1.jsonl)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `malicious-expression`
- `unknown-query`
- `unauthorized-tool`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `no-eval-exec`
- `structured-not-found`
- `contract-dataset-matches-code`

关联 suite：`tool-input-adversarial-v1`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
