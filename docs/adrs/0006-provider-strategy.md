---
layout: default
title: ADR 0006 · Provider strategy
---

# ADR 0006：Provider 策略

状态：Accepted

## 决定

默认安装与 CI 不依赖模型 API。Provider 通过 optional dependency 和注入的 `BaseChatModel` 接入。

依赖组：

- `core`：Pydantic、LangChain、LangGraph 与领域代码。
- `provider-openai`：OpenAI 兼容模型适配。
- `eval-cloud`：LangSmith 客户端。
- `persistence`：SQLite checkpointer。
- `experimental`：Deep Agents。
- `dev`：测试、lint、类型和站点合同工具。

## 版本策略

- `pyproject.toml` 声明兼容范围。
- `uv.lock` 固定当前验证环境。
- CI 使用 `uv sync --frozen`，不在流水线临时解析新版本。
- optional 路径只有在凭证和预算明确时运行 live smoke；离线测试不读取 `.env`。
