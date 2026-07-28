# f03-agent-skills · Agent Skills

> 状态：`scaffold`。当前只有实验合同，不代表能力已实现或验证。

将程序性知识按需加载，同时审计来源、版本和不可信指令。

- 前置：[f02-context-compaction](../f02-context-compaction/)
- 产物：`skill trust contract`
- 能力：`progressive-disclosure`, `skill-versioning`

## 1. Frame

先回答：**Skill 是知识模块，还是新的供应链攻击面？**

不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。

## 2. Predict

在运行前预测可观察轨迹：

```text
discover → validate → load → execute
```

## 3. Build

完成或审查 `skill trust contract`。实现入口：

- [`docs/adrs/0005-tool-security.md`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/adrs/0005-tool-security.md)

## 4. Break

主动制造这些失败，不要只跑 happy path：

- `untrusted-skill`
- `version-drift`
- `overbroad-permission`

## 5. Trace

只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。

## 6. Evaluate

验收合同：

- `source-recorded`
- `permissions-bounded`

关联 suite：`frontier-skills-candidate`

## 7. Reflect

解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。

## 8. Promote

把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。

[返回实验目录](../)
