---
layout: default
title: RFC 0001 · Agent Engineering Lab
---

# RFC 0001：产品范围

## 决定

仓库从“按 LangChain API 排课的中文教程”重构为“按 Agent 工程能力组织的可执行实验室”。

核心问题不再是“会不会调用某个框架”，而是：

1. 任务成功的合同是什么？
2. Agent 使用了哪些工具、证据与预算？
3. 失败发生在最终答案、单步决策还是完整轨迹？
4. 修复能否成为永久回归资产？

## 稳定核心

- Typed request、budget、evidence、artifact 与 outcome。
- 工具权限、副作用、幂等和错误语义。
- Context、thread state、cross-thread memory 的生命周期边界。
- Trace、dataset、grader、regression 与 verification passport。
- 无 API Key 可运行的 fixture 和 deterministic workflow。

## 可替换实现层

- LangChain `create_agent`。
- 自定义 LangGraph。
- LangSmith tracing / evaluation / deployment。
- Deep Agents、MCP、A2A 与真实模型提供商。

这些实现可以变化，但不得反向污染领域合同和离线数据集。

## 教学主线

每个实验遵循：

```text
Frame → Predict → Build → Break → Trace → Evaluate → Reflect → Promote
```

第一条垂直切片是 Trusted Research：fixture 检索、结构化证据、引用校验、无证据拒答、轨迹评测和验证护照。

## 非目标

- 不做 LangChain 全 API 百科。
- 不默认引入多 Agent、自动规划或宿主机代码执行。
- 不开发自有生产控制台。
- 不把 message history 等同于长期记忆。
- 不用 LLM Judge 代替 schema、权限、预算和引用完整性检查。
- 不把离线 fixture、模型实验或 CI 绿色描述成线上真实成功。

## V2 验收

- fresh clone 后 `uv sync --frozen` 成功。
- `uv run agent-lab verify` 在没有 API Key 时通过。
- Workflow 与 LangGraph 对同一 fixture 数据集给出一致的可追溯结果。
- 每个关键结论都引用 Evidence ID。
- 工具错误、未知答案、注入文本和预算耗尽都有回归样例。
- `UNKNOWN` 与 `ERROR` 不产生质量分。
- Pages 构建前执行 Python、eval、课程元数据和站点门禁。
- 发布产物包含 commit、lock、dataset 和 gate 摘要。
