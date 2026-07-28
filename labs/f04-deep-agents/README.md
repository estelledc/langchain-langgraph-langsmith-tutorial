# f04-deep-agents · Deep Agents Harness

> 状态：`implemented`。实现已存在；涉及外部模型的行为仍需单独 live 验证。

用可选适配器比较规划、文件系统和 subagents 是否改善长任务。

- 前置：[f03-agent-skills](../f03-agent-skills/)
- 产物：`lazy Deep Agents adapter`
- 能力：`deep-agent`, `planning`, `context-offload`

## 1. Frame

先回答：**Harness 的额外复杂度在哪些 case 上真的换来收益？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
plan → delegate → synthesize
```

## 3. Build

完成或审查 `lazy Deep Agents adapter`。实现入口：

- [`src/agent_lab/adapters/deepagents.py`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/src/agent_lab/adapters/deepagents.py)
- [`docs/adrs/0006-provider-strategy.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0006-provider-strategy.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `dependency-drift`
- `context-bloat`
- `needless-delegation`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `optional-dependency`
- `same-domain-outcome`

关联 suite：`frontier-deepagents-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
