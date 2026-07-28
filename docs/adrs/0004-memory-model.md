---
layout: default
title: ADR 0004 · Memory model
---

# ADR 0004：记忆模型

状态：Accepted

## 决定

明确区分五类状态：

| 类型 | 生命周期 | 例子 |
|---|---|---|
| Runtime context | 单次调用且不可序列化 | 身份、权限、客户端、密钥 |
| Thread state | 同一 thread | 当前任务、步骤、临时证据、中断点 |
| Semantic memory | 跨 thread | 经审批的稳定偏好和事实 |
| Episodic memory | 跨 thread | 经筛选的成功或失败执行摘要 |
| Procedural memory | 版本化 | policy、prompt、skill、workflow |

LangGraph checkpointer 只负责 thread state；Store 负责跨 thread 数据。聊天记录不会自动升级为长期记忆。

## 写入规则

- 长期写入必须有来源、去重键、有效期或版本。
- 从不可信工具输出提取的内容默认不可写入。
- 涉及身份、权限、个人偏好或团队决策时，写入前需要显式审批。
- 相同 thread 是否可续接由复用的 checkpointer 决定，不能只看相同 `thread_id` 字符串。
