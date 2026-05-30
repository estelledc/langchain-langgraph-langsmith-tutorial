# LangChain Tutorial Zero — 给零基础选手的 AI 辅助学习教程

> 不是另一份官方教程，是一份**学习脚手架**：你跟 Claude Code / Cursor 对话产出代码，再回头对照参考答案看自己写得如何。
>
> **代码实测**：14 个 final/.py 全部经过作者实测（langchain 1.3.2，2026-05）→ [docs/test-runs.md](docs/test-runs.md)

---

## 30 秒上手

```bash
# 1. fork 本仓库到你自己的 GitHub，clone（YOUR-USERNAME 换成你自己）
git clone https://github.com/YOUR-USERNAME/langchain-langgraph-langsmith-tutorial.git
cd langchain-langgraph-langsmith-tutorial

# 2. 跟着 SETUP.md 配 .env 和依赖（约 15 分钟，含申 API Key）
open SETUP.md

# 3. 读完元教程再开始（5 分钟，必读）
open HOW_TO_LEARN_WITH_AI.md

# 4. 走第一篇 tutorial
open tutorial/week-1-langchain/01_hello_llm.md
```

> **教程在线版**：[estelledc.github.io/langchain-langgraph-langsmith-tutorial](https://estelledc.github.io/langchain-langgraph-langsmith-tutorial/) — 不想 clone 也能直接读全部内容
>
> **仓库 vs 项目代号**：仓库名 `langchain-langgraph-langsmith-tutorial` 是历史名（从原 fork 继承）；这套教程的代号叫 **LangChain Tutorial Zero**——以后可能改名，但 git clone 命令以仓库名为准。

---

## VS 官方教程

| 维度 | LangChain 官方文档 | 这套教程 |
|---|---|---|
| **目标读者** | 已懂 LLM / Agent 概念的工程师 | 编程零基础（能看懂 `def foo():` 即可） |
| **教学风格** | reference style — 直给代码 + API 文档 | learning-by-doing — 任务卡 + 自己写 + 自检 |
| **AI 工具维度** | 完全没有 | 每个知识点配「给 AI 的 prompt」让 CC/Cursor 陪你拆代码 |
| **术语处理** | 默认你懂（"embedding"/"agent"/"trace" 直接用） | [docs/concepts.md](docs/concepts.md) 18 个核心术语全用日常类比 |
| **错误处理** | 散落各处，要自己搜 | [docs/debug-recipes.md](docs/debug-recipes.md) 16 个高频报错速查 |
| **prompt 工程** | 几个固定示例 | [docs/prompts-cheatsheet.md](docs/prompts-cheatsheet.md) 21 个高频 prompt 模板 + 反模式表 |
| **语言** | 主英文，中文翻译滞后 | 原生中文，术语首次出现注英文 |
| **进阶练习** | 无 | [docs/challenges.md](docs/challenges.md) 7 个真实小项目挑战 |
| **典型用法** | "这个 API 怎么调" 时翻文档查一下 | "我要从零学会"——4 周 14 篇按节奏走 |

**一句话**：官方教程是**字典**，这套是**家教**。

---

## 这是什么

- 4 周从零学会 LangChain + LangGraph + LangSmith
- **核心方法**：每个知识点配「给 AI 的 prompt」+ 自检任务，让 AI 陪你拆代码
- **后端**：DashScope（通义千问）+ LangSmith Trace（都有免费层）

---

## 仓库结构

```
├── HOW_TO_LEARN_WITH_AI.md      # 必读：怎么用 CC/Cursor 学陌生代码
├── SETUP.md                     # 必读：环境搭建
├── tutorial/                    # 学习剧本（你主要看这里）
│   ├── README.md                # 学习路线索引
│   └── week-1-langchain/  ...   # 按周组织，4 周共 14 篇
├── final/                       # 参考答案（任务卡指明何时偷看）
│   ├── _common.py               # 共享 boilerplate（学习者别改）
│   └── 01_langchain/  ...       # 14 个独立可执行 .py
├── docs/                        # 速查 + 进阶资源
│   ├── concepts.md              # 18 个核心术语小词典
│   ├── prompts-cheatsheet.md    # 21 个高频 prompt 模板
│   ├── debug-recipes.md         # 16 个高频报错速查
│   ├── challenges.md            # capstone 后 7 个真实挑战
│   └── test-runs.md             # 14 个 final 实测档案
└── _scratch/                    # 你的主战场（你写的代码放这；已 gitignore）
    └── journal/                 # 卡点日志
```

---

## 适合你吗

✅ 适合：

- 编程零基础或只懂语法
- 第一次碰 LangChain
- 想用 AI 工具学陌生代码
- 中文学习者
- 有 30+ 小时（4 周）

❌ 不适合：

- 想找官方文档替代品（请去 [docs.langchain.com](https://docs.langchain.com)）
- 不想写代码只想读
- 英文资料够你看

---

## 学习路线

| 周次 | 内容 | 状态 |
|---|---|---|
| **Week 1** | LangChain 核心：LLM、Prompt、LCEL、Memory、RAG | ✅ 5 篇 |
| **Week 2** | Tool & Agent | ✅ 1 篇 |
| **Week 3** | LangGraph：StateGraph、条件边、HITL、多 Agent | ✅ 4 篇 |
| **Week 4** | LangSmith + Capstone：Tracing、Evaluation、Dataset、综合项目 | ✅ 4 篇 |

每篇 tutorial 结构统一：**准备 → 任务卡（4-5 个） → 通关条件 → 卡点日志 → 通往下一站**。

通关后建议：

1. 跑完 capstone → 挑一个 [challenges.md](docs/challenges.md) 做出师作业
2. 你自己沉淀的 prompt → 提 PR 到 [prompts-cheatsheet.md](docs/prompts-cheatsheet.md)
3. 你撞到的新报错 → 提 PR 到 [debug-recipes.md](docs/debug-recipes.md)

---

## 这个仓库**不**做什么

- 不是另一份官方文档翻译——你已经能看英文文档了，本仓库帮不上忙
- 不教 Python 基础——请先看 [Python 基础教程](https://www.runoob.com/python3/python3-tutorial.html) 至少能写循环和函数
- 不保证最新版 LangChain 兼容——pin 在 1.3.x，半年回头一次

---

## 反馈与贡献

提 Issue 或 PR 都行：

- 撞了报错 → PR 到 `debug-recipes.md`
- 觉得某 tutorial 卡点 → 提 issue 描述哪一段不顺
- 用过的 prompt 真有效 → PR 到 `prompts-cheatsheet.md`
- 完成挑战 → PR 你的卡点日志到 `_scratch/journal/challenge-N-<日期>.md`

本项目基于 [MIT License](LICENSE)。

---

> **设计哲学**：零基础不是问题，**不知道怎么问**才是。这套教程的全部价值，是把"你不知道怎么问"的部分预制好，让你只管按 prompt 走。
