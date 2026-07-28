---
layout: default
title: Compatibility
description: Python、LangChain、LangGraph、LangSmith 与可选适配器的版本边界。
---

# Compatibility

V2 使用兼容范围表达支持面，`uv.lock` 固定本次验证环境。

| 组件 | `pyproject.toml` | 当前锁定基线 | 默认安装 |
|---|---|---|---|
| Python | `>=3.11,<3.14` | CI 矩阵 3.11 / 3.12 / 3.13 | 是 |
| LangChain | `>=1.3,<2` | 1.3.14 | 是 |
| LangGraph | `>=1.2,<2` | 1.2.9 | 是 |
| LangSmith | `>=0.10,<1` | 0.10.10 | LangChain 传递依赖；cloud 行为不默认启用 |
| Pydantic | `>=2.13,<3` | 2.13.4 | 是 |
| langchain-openai | `>=1.4,<2` | 由 lock 固定 | 可选 |
| SQLite checkpointer | `>=3.1,<4` | 由 lock 固定 | 可选 |
| Deep Agents | `>=0.6,<1` | 由 lock 固定 | 可选实验 |
| LangGraph CLI | `>=0.4.31,<1` | 由 lock 固定 | 可选本地 server |

升级流程：

1. 新分支更新兼容范围或执行 `uv lock --upgrade-package <name>`。
2. `uv sync --frozen` 验证 fresh resolution。
3. 运行 tests、fast/security/contracts eval、课程和站点门禁。
4. 涉及模型或外部服务时单独运行 live suite，记录 UNKNOWN 和 ERROR。
5. 更新本页锁定基线和 release passport。

V1 的 `langchain-community==1.0.4` 无可解析版本，因此只保留历史，不再作为 V2 安装合同。
