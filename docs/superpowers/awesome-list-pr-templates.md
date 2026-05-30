# Awesome list PR 提交模板

> 基于 2026-05-30 调研：扫了 7 个候选 list，2 个推荐 + 3 个 fallback。
> 主仓库 [README.md](../../README.md) Phase 2 配套。

## 推荐目标 Top 2

### A. Awesome-Chinese-LLM（22.6k⭐，强推）

- **URL**: https://github.com/HqWu-HITCS/Awesome-Chinese-LLM
- **目标章节**: Tutorials → LLM 应用教程
- **PR 难度**: easy（追加一行表格）
- **维护节奏**: 2026-05 仍有 PR 合并，月度级响应

**操作步骤**：

1. fork 到 `estelledc/Awesome-Chinese-LLM`
2. 找到 README 中 "LLM 应用教程" 表格段（搜索 "Tutorials" 或 "教程"）
3. 在表格末尾追加：

```markdown
| [langchain-langgraph-langsmith-tutorial](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial) | 给编程零基础选手的 LangChain 1.3.x 中文教程，4 周 14 篇 learning-by-doing，AI 辅助学习元教程 + 任务卡结构 |
```

4. 提 PR：

**PR 标题**:
```
docs: add langchain-langgraph-langsmith-tutorial to LLM应用教程
```

**PR body**:
```markdown
## 新增条目

- 项目：[langchain-langgraph-langsmith-tutorial](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial)
- 章节：Tutorials → LLM应用教程

## 简介

面向编程零基础学习者的 LangChain 1.3.x 中文教程，包含 4 周共 14 篇 learning-by-doing 实战章节。覆盖 LangChain / LangGraph / LangSmith 三件套从环境搭建到 Agent 编排的完整链路。

## 与已收录项目的差异

- 与现有 LangChain 教程类条目相比，本仓库**面向零基础**而非有 Python 经验的开发者，每篇含「日常类比 → 概念 → 代码 → 踩坑」四段任务卡结构
- 基于 **LangChain 1.3.x 实测**（而非已迁移弃用的 0.x API），所有代码可直接跑通
- 同时是「AI 辅助学习元教程」——记录用 Claude/Cursor 辅助学新框架的 7 条 prompt 心法与排查路径，对其他 AI 时代的学习者有迁移价值

## 现状声明

- 当前 14 ⭐，仍在持续更新中
- 已配置 pre-commit lint，所有内部链接、来源字段均自动校验
- 后续会持续维护到 LangChain 2.x 升级周期

## Checklist

- [x] 链接可访问
- [x] 描述简洁（< 80 字）
- [x] 中文项目，匹配本 list 定位
- [x] 章节归属合理（应用教程类）
```

---

### B. awesome-LLM-resourses（8.4k⭐，推荐）

- **URL**: https://github.com/WangRongsheng/awesome-LLM-resourses
- **目标章节**: 教程 (Tutorial)
- **PR 难度**: easy（追加一行 bullet）
- **维护节奏**: 2026-04 最近合并 PR #88，活跃

**操作步骤**：

1. fork 到 `estelledc/awesome-LLM-resourses`
2. 找到 README 中 "教程 (Tutorial)" 章节
3. 在列表末尾追加：

```markdown
- [langchain-langgraph-langsmith-tutorial](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial) - 给编程零基础选手的 LangChain 1.3.x 中文教程，4 周 14 篇 learning-by-doing；AI 辅助学习元教程 + 1.3.x 实测 + 任务卡结构。
```

4. 提 PR：

**PR 标题**:
```
Add langchain-langgraph-langsmith-tutorial to 教程 section
```

**PR body**:
```markdown
## 新增条目

- 项目：[langchain-langgraph-langsmith-tutorial](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial)
- 章节：教程 (Tutorial)

## 简介

面向编程零基础学习者的 LangChain 1.3.x 中文教程，4 周共 14 篇 learning-by-doing 实战章节，覆盖 LangChain / LangGraph / LangSmith 全链路。每篇采用「日常类比 → 概念 → 代码 → 踩坑」四段任务卡结构。

## 与已收录项目的差异化

本 list 教程章节已有的条目多为「面向有 Python 经验开发者的概念讲解」或「官方文档翻译」。本仓库的差异点：

1. **零基础定位**：每个新概念先用日常生活类比，再进入技术定义，不假设懂任何术语
2. **LangChain 1.3.x 实测**：当前许多中文 LangChain 教程仍停留在 0.x API（已 deprecated），本教程基于 1.3.x 全新 API 实测，代码即跑即通
3. **AI 辅助学习元教程**：附带「我是怎么用 Claude/Cursor 辅助学这个概念的」7 条 prompt 心法与排查路径，对所有 AI 时代的自学者有迁移价值

## 仓库现状

- Stars: 14（小而新，但定位差异化清晰）
- 维护节奏：4 周计划进行中
- 质量保障：pre-commit hook 强制校验链接、来源字段
- 维护承诺：后续持续跟进 LangChain 2.x 升级

## Checklist

- [x] 链接可访问且为公开仓库
- [x] 中文资源，匹配本 list 定位
- [x] 一句话描述，控制在 80 字内
- [x] 章节归属合理（教程类）
```

---

## Fallback（top 2 被 reject 时再投）

| 名字 | URL | 章节 | 风险 |
|---|---|---|---|
| build-your-own-x | https://github.com/codecrafters-io/build-your-own-x | Build your own AI Model | 65% 合并率 + 批处理节奏 + 偏英文 |
| awesome-chatgpt-zh | https://github.com/yzfly/awesome-chatgpt-zh | 应用开发指南 / LLM RAG 开发指南 | 维护近停滞，PR 可能石沉 |
| Awesome-AITools | https://github.com/ikaijua/Awesome-AITools | GPT LLMs Applications | 章节语义不匹配（产品 vs 教程），易被 stale |

## 不推荐目标

- **awesome-langchain (kyrolabs)**：拒绝中文内容，硬规则
- **awesome-compression (datawhalechina)**：主题不匹配（模型压缩 vs LangChain）
- **FindTheChatGPTer**：仓库已停更近 3 年

## 投递节奏建议

1. 先投 A（最高曝光），观察 1 周
2. 同时或一周后投 B（备份覆盖）
3. 两周内 A/B 都没动静再考虑 fallback
4. 提 PR 前先在 fork 仓库上跑通 lint（如果该 list 配了 pre-commit）

## 投递后维护

- 标记 A/B 的 PR URL 在 daily 日报或本文末尾
- 被合并后，更新主仓 README 顶部加「被 X⭐ list 收录」徽章
