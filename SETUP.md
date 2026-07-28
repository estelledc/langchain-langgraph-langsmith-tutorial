---
layout: default
title: Setup
description: Agent Engineering Lab 的离线安装、可选 provider 与验证入口。
---

# Setup

## 1. 默认离线路径

需要 Python 3.11–3.13 和 [uv](https://docs.astral.sh/uv/)。仓库通过 `.python-version` 默认选择 Python 3.13。

```bash
git clone https://github.com/estelledc/langchain-langgraph-langsmith-tutorial.git
cd langchain-langgraph-langsmith-tutorial
uv sync --frozen
uv run agent-lab run --goal "LangSmith 的评测闭环是什么？"
uv run agent-lab eval --suite fast
```

这条路径不读取 `.env`，不调用模型 API，也不上传 Trace。

## 2. 完整本地门禁

```bash
uv run agent-lab verify
```

它依次运行格式检查、lint、mypy、pytest + coverage、课程合同、站点源合同、`git diff --check` 和 fast eval。

站点渲染另需 Ruby/Bundler：

```bash
bundle install
bundle exec jekyll build
uv run python scripts/check_site.py --built _site
```

## 3. 可选依赖

```bash
uv sync --extra provider-openai  # OpenAI 兼容模型
uv sync --extra eval-cloud       # 显式 LangSmith 实验
uv sync --extra persistence      # SQLite checkpointer
uv sync --extra experimental     # Deep Agents
uv sync --extra server           # 本地 Agent Server CLI
uv sync --all-extras             # 全部可选适配器
```

可选依赖不改变离线门禁，也不会自动启用外部调用。

## 4. Live provider

复制 `.env.example` 为 `.env`，只填写自己被授权使用的凭证。不要提交 `.env`。

Live 运行至少要显式记录：

- provider 与 model configuration。
- prompt、dataset 和 grader 版本。
- trials、pass rate、unknown rate 和 evaluator error rate。
- 外部错误、成本和延迟。

单次 smoke 只证明一次请求发生过，不证明稳定能力或生产状态。

## 5. Agent Server

`langgraph.json` 已注册 `trusted_research` 图。安装锁定的可选 CLI 后可本地启动：

```bash
uv sync --extra server
uv run langgraph dev --no-browser
```

云端或自托管部署需要单独的认证、数据库、队列和 deployment receipt。本仓库的 Pages 部署不等于 Agent Server 已部署。
