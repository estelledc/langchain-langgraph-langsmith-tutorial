---
layout: default
title: ADR 0005 · Tool security
---

# ADR 0005：工具安全

状态：Accepted

## 决定

每个工具声明 typed input/output、side effect、幂等性、权限、超时、重试和错误集合。工具输出默认不可信。

## 强制规则

- 读写能力分离；副作用工具必须有幂等键。
- 高风险或外部写操作必须经过代码层 policy，不依赖 prompt 自律。
- 搜索 fixture 必须明确标记 `fixture`，不得声称联网或最新。
- 模型可控表达式不得进入 `eval`、`exec` 或宿主 shell。
- 计算器只允许白名单 AST 节点，并限制数字、指数、深度和结果大小。
- 工具返回 `NOT_FOUND` 时 Agent 必须拒答或承认未知，不得伪造成功。
- Evidence 中的提示注入文本可以保存用于审计，但不能改变系统指令、policy 或工具权限。
