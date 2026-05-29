# LangChain Tutorial Zero — 给零基础选手的 AI 辅助学习教程

> 不是另一份官方教程，是一份**学习脚手架**：你跟 Claude Code / Cursor 对话产出代码，再回头对照参考答案看自己写得如何。

## 这是什么

- 4 周从零学会 LangChain + LangGraph + LangSmith
- **核心方法**：每个知识点配「给 AI 的 prompt」+ 自检任务，让 AI 陪你拆代码
- **后端**：DashScope（通义千问）+ LangSmith Trace（都有免费层）

## 30 秒上手

```bash
git clone <fork 后你自己的 URL>
cd langchain-tutorial-zero

# 1. 跟着 SETUP.md 配 .env 和依赖
open SETUP.md

# 2. 读完元教程再开始（5 分钟）
open HOW_TO_LEARN_WITH_AI.md

# 3. 走第一篇 tutorial
open tutorial/week-1-langchain/01_hello_llm.md
```

## 仓库结构

```
├── HOW_TO_LEARN_WITH_AI.md   # 必读：怎么用 CC/Cursor 学陌生代码
├── SETUP.md                  # 必读：环境搭建
├── tutorial/                 # 学习剧本（你主要看这里）
│   └── week-1-langchain/  ... week-4-langsmith-and-project/
├── final/                    # 参考答案（任务卡指明何时偷看）
├── _scratch/                 # 你的主战场（自己代码放这）
└── docs/                     # 速查表 / 排错手册
```

## 适合你吗

✅ 适合：编程零基础或只懂语法、第一次碰 LangChain、想用 AI 工具学陌生代码、有 30+ 小时（4 周）
❌ 不适合：想找官方文档替代品、不想写代码只想读、英文资料够你看

## 这个仓库**不**做什么

- 不是另一份官方文档翻译
- 不教你 Python 基础（请先有 Python 入门，至少能看懂 `def foo():`）
- 不保证最新版 LangChain 兼容（pin 在 0.3.x，半年回头一次）

## 反馈与贡献

提 Issue 或 PR 都行。本项目基于 [MIT License](LICENSE)。
