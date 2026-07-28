---
layout: default
title: Agent Engineering Labs
description: 15 个核心实验与 10 个前沿实验的可执行学习路径。
---

# Agent Engineering Labs

课程不是按框架 API 排序，而是按任务合同、工具、上下文、状态、证据和评测能力推进。状态字段是证据边界，不是进度装饰。

## 核心路径

从无 Agent 基线到证据闭环与部署合同。

| 实验 | 状态 | 产物 | 核心问题 |
|---|---|---|---|
| [00-no-agent-baseline · Agent、Workflow 与普通程序](00-no-agent-baseline/) | `executable` | TrustedResearchWorkflow | 这个问题是否根本不需要 Agent？ |
| [01-typed-io · Typed I/O 与结构化结果](01-typed-io/) | `executable` | framework-neutral domain contracts | 哪些状态应该由代码拒绝，而不是让模型猜？ |
| [02-tool-contracts · 工具合同与错误语义](02-tool-contracts/) | `executable` | fixture search and safe calculator | NOT_FOUND、TIMEOUT 和系统异常为什么不能返回同一句自然语言？ |
| [03-langchain-agent · LangChain create_agent](03-langchain-agent/) | `implemented` | LangChainResearchRuntime | 简单工具循环是否已足够，还是确实需要自定义图？ |
| [04-middleware-budget · Middleware、Policy 与预算](04-middleware-budget/) | `executable` | ToolPolicy and call-limit middleware | 哪些约束放在 Prompt 里仍然不够？ |
| [05-context-engineering · Context Engineering](05-context-engineering/) | `executable` | RunContext and evidence boundary | 什么应该进入模型，什么只应该留在工具或运行时？ |
| [06-langgraph-runtime · LangGraph Runtime](06-langgraph-runtime/) | `executable` | LangGraphResearchRuntime | 哪些状态、分支和恢复需求值得引入自定义图？ |
| [07-persistence-hitl · Persistence、Interrupt 与 Resume](07-persistence-hitl/) | `executable` | approval graph and injected checkpointer | 相同 thread_id 在什么条件下才真的能续接？ |
| [08-evidence-first-rag · Evidence-first Retrieval](08-evidence-first-rag/) | `executable` | versioned fixture search | 如何证明答案真由检索证据支持，而不是模型顺手编的？ |
| [09-memory-engineering · Memory Engineering](09-memory-engineering/) | `executable` | approval-aware memory store | 哪些聊天内容有资格升级为长期记忆？ |
| [10-observability · Observability 与 Trace](10-observability/) | `executable` | TraceRecorder and RunMetrics | 看到一条失败 Trace 后，如何定位是工具、路由还是证据问题？ |
| [11-agent-evals · Final、Step 与 Trajectory Eval](11-agent-evals/) | `executable` | deterministic graders | 哪些失败可以被代码百分之百判定？ |
| [12-regression-engineering · Regression Engineering](12-regression-engineering/) | `executable` | capability and regression datasets | 一次修复如何变成以后永远会被检查的资产？ |
| [13-deployment · Agent Server 与发布合同](13-deployment/) | `configured` | Agent Server blueprint and GitHub Pages workflow | 构建成功、部署请求和线上可用分别需要什么证据？ |
| [14-production-loop · Production Trace → Dataset → Regression](14-production-loop/) | `scaffold` | production promotion contract | 生产流量里的异常如何安全、可追溯地回流？ |

## 前沿实验

只有测出收益后才进入核心架构的可选能力。

| 实验 | 状态 | 产物 | 核心问题 |
|---|---|---|---|
| [f01-filesystem-context · Filesystem Context](f01-filesystem-context/) | `scaffold` | experiment contract | 文件系统比长 Prompt 多带来了什么可测收益？ |
| [f02-context-compaction · Context Compaction](f02-context-compaction/) | `scaffold` | compaction comparison | 压缩省下的 token 是否抵得过事实丢失？ |
| [f03-agent-skills · Agent Skills](f03-agent-skills/) | `scaffold` | skill trust contract | Skill 是知识模块，还是新的供应链攻击面？ |
| [f04-deep-agents · Deep Agents Harness](f04-deep-agents/) | `implemented` | lazy Deep Agents adapter | Harness 的额外复杂度在哪些 case 上真的换来收益？ |
| [f05-subagents · Subagents](f05-subagents/) | `scaffold` | architecture comparison | 并行收益是否超过协调成本和错误面？ |
| [f06-sandboxed-code · Sandboxed Code Execution](f06-sandboxed-code/) | `scaffold` | sandbox threat model | 执行能力需要哪些宿主机之外的安全合同？ |
| [f07-mcp · MCP Adapter](f07-mcp/) | `scaffold` | MCP boundary adapter | 协议便利性如何不反向污染领域模型？ |
| [f08-a2a · A2A](f08-a2a/) | `scaffold` | remote-agent protocol experiment | 这是远程 Agent 协作，还是一个进程内的普通函数调用？ |
| [f09-online-evals · Online Evals](f09-online-evals/) | `scaffold` | online evaluation policy | 哪些线上信号足以触发告警，哪些需要人工复核？ |
| [f10-adaptive-routing · Adaptive Routing](f10-adaptive-routing/) | `scaffold` | routing benchmark | 路由策略在哪些分布变化下会失效？ |

## 状态语义

- `executable`：离线实现与自动化验收均已接入。
- `implemented`：代码已存在，但 live provider 或外部系统行为未在默认门禁中验证。
- `configured`：集成与部署配置已存在，不能替代真实部署 receipt。
- `scaffold`：只有实验合同，不能描述为已完成能力。

V1 教程保存在 [`v1-legacy`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/tree/v1-legacy)。
