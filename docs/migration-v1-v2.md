---
layout: default
title: V1 → V2 Migration
description: 从框架教程迁移到 Agent Engineering Lab 的目录、行为和证据变化。
---

# V1 → V2 Migration

## 保留什么

- V1 原始提交由 `v1-legacy` 标签固定。
- V1 课程、参考代码、测试记录和学习约束保存在 `legacy/v1/`。
- 原有 Git 历史和公开 Issue 不删除。

## 替换什么

| V1 | V2 |
|---|---|
| `requirements.txt` | `pyproject.toml` + `uv.lock` |
| `tutorial/ + final/` | `curriculum/ + labs/ + src/ + tests/ + datasets/ + evals/` |
| mock `web_search` | 明确的 `FixtureSearchAdapter` |
| 模型输入进入 `eval()` | `SafeCalculator` 白名单 AST |
| messages 为主要状态 | typed Evidence、Citation、Artifact 与 RunResult |
| 少量最终答案分数 | Final、Step、Trajectory、Attribution、Policy grader |
| grader 异常给 0.5 | `ERROR`，无 score |
| 语法编译 CI | fresh install + full offline gate + rendered Pages |

V1 页面链接若不再存在，请使用 GitHub 标签查看；V2 不承诺旧路径继续作为新课程入口。
