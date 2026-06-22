---
title: Week 2 · Tools & Agent
nav_order: 2
---

# Week 2 · Tools & Agent

> 让 LLM 不只输出文本，还能调外部工具、产出结构化结果、稳健流式响应。

## 章节

1. [tools agent](./01_tools_agent.md) — tool calling + agent 循环
2. [structured output](./02_structured_output.md) — Pydantic schema 强约束 LLM 输出
3. [streaming and resilience](./03_streaming_and_resilience.md) — 流式 + 重试 + fallback

## 学完应能

- 给 LLM 注册自定义 tool 并跑出可用 agent
- 用 `with_structured_output` 拿到类型安全的 JSON
- 处理 rate limit / 超时 / 网络抖动

[← 回教程总览](../)
