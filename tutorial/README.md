# Tutorial — 学习剧本

每篇 tutorial 是一份带任务卡 + 给 AI 的 prompt 的"学习剧本"。

**这不是文档，是脚手架**——你不读完它就懂，你按它跟 AI 对话产出代码，最后回头对比 `final/` 看自己写得怎样。

> **强烈建议先读 [00_big_picture.md](00_big_picture.md)**（10 分钟）——给你 4 周地图，走每一篇时不迷路。这一页官方教程里没有。

---

## 学习路线（4 周 16 篇）

> 时长是「零基础学员跟着 AI 走一遍」的预估（含读 tutorial + 跟 AI 对话写代码 + 跑 final 对照 + 写卡点日志）。熟手通常是这个时长的 1/3。

| 周次 | 目录 | 概念 | 篇数 | 预估时长 |
|------|------|------|------|----------|
| Week 1 | [week-1-langchain/](week-1-langchain/) | LangChain 核心：LLM、Prompt、LCEL、Memory、RAG | 5/5 | ~4 小时 |
| Week 2 | [week-2-tools-and-agent/](week-2-tools-and-agent/) | Tool & Agent / 结构化输出 / 流式 + 容错 | 3/3 | ~3.5 小时 |
| Week 3 | [week-3-langgraph/](week-3-langgraph/) | StateGraph、条件边、HITL、多 Agent | 4/4 | ~5.5 小时 |
| Week 4 | [week-4-langsmith-and-project/](week-4-langsmith-and-project/) | 追踪、评估、综合项目 | 4/4 | ~5 小时 |

---

## 学习路径图（依赖关系）

```mermaid
flowchart TD
    Start([HOW_TO_LEARN_WITH_AI.md<br/>5 分钟元教程]) --> W1_01

    subgraph W1[Week 1 · LangChain 核心]
        W1_01[01 hello_llm<br/>LLM 调用三态]
        W1_02[02 prompt_template<br/>模板 + 占位符]
        W1_03[03 chains<br/>LCEL 管道]
        W1_04[04 memory<br/>多轮对话]
        W1_05[05 rag_basic<br/>检索增强]
        W1_01 --> W1_02 --> W1_03 --> W1_04 --> W1_05
    end

    subgraph W2[Week 2 · Tool & 工程化]
        W2_01[01 tools_agent<br/>@tool + AgentExecutor]
        W2_02[02 structured_output<br/>with_structured_output + Pydantic]
        W2_03[03 streaming_and_resilience<br/>stream / retry / fallback]
        W2_01 --> W2_02 --> W2_03
    end

    subgraph W3[Week 3 · LangGraph]
        W3_01[01 simple_graph<br/>StateGraph 三件套]
        W3_02[02 conditional_edges<br/>ReAct + Checkpointer]
        W3_03[03 human_in_the_loop<br/>interrupt + 审批]
        W3_04[04 multi_agent<br/>Supervisor + 并行]
        W3_01 --> W3_02 --> W3_03 --> W3_04
    end

    subgraph W4[Week 4 · LangSmith + Capstone]
        W4_01[01 tracing<br/>@traceable + RunTree]
        W4_02[02 evaluation<br/>evaluator + LLM-as-Judge]
        W4_03[03 dataset<br/>CRUD + 从 trace 收集]
        W4_04[04 capstone<br/>4 文件研究助手]
        W4_01 --> W4_02 --> W4_03 --> W4_04
    end

    W1_05 --> W2_01
    W2_03 --> W3_01
    W3_04 --> W4_01

    W4_04 --> Done([挑战赛<br/>docs/challenges.md])

    style Start fill:#a1c4fd,color:#000
    style Done fill:#84fab0,color:#000
    style W1 fill:#fff5e6
    style W2 fill:#ffe6e6
    style W3 fill:#e6f7ff
    style W4 fill:#f0e6ff
```

> mermaid 图在 GitHub repo 浏览 / GitHub Pages 都能直接渲染。本地预览用 VS Code 装 "Markdown Preview Mermaid Support" 插件。

---

## 前置概念依赖 + 预估时长

走每一篇前需要懂的"前置概念"——讲不出 = 回去补。时长是零基础学员节奏。

| Tutorial | 必须懂 | 预估时长 |
|---|---|---|
| **week-1/01_hello_llm** | LLM 是补全器 / token 计费 / Chat 三种角色（→ [docs/concepts.md](../docs/concepts.md)） | 35 分钟 |
| **week-1/02_prompt_template** | 01 内容 + Python f-string | 35 分钟 |
| **week-1/03_chains** | 02 内容 + Unix 管道（`cat \| grep`）的概念 | 50 分钟 |
| **week-1/04_memory** | 03 内容 + dict / list 操作 | 60 分钟 |
| **week-1/05_rag_basic** | 04 内容 + Embedding 是什么 | 80 分钟 |
| **week-2/01_tools_agent** | week-1 全部 + Python 装饰器（`@xxx`） | 80 分钟 |
| **week-2/02_structured_output** | 01 + Pydantic BaseModel | 60 分钟 |
| **week-2/03_streaming_and_resilience** | 02 + asyncio 基础（at-least 知道 async / await） | 70 分钟 |
| **week-3/01_simple_graph** | week-2 + TypedDict + 流程图概念 | 70 分钟 |
| **week-3/02_conditional_edges** | 01 + ReAct 循环 | 80 分钟 |
| **week-3/03_human_in_the_loop** | 02 + Checkpointer / thread_id | 80 分钟 |
| **week-3/04_multi_agent** | 03 + Supervisor 模式日常类比（PM + 专家团） | 95 分钟 |
| **week-4/01_tracing** | week-3 + 黑匣子录像类比 | 60 分钟 |
| **week-4/02_evaluation** | 01 + dataset 概念 | 80 分钟 |
| **week-4/03_dataset** | 02 + CRUD 概念 | 60 分钟 |
| **week-4/04_capstone** | 全部 15 篇 + 项目分模块的工程感 | 110 分钟 |

---

## 走法

1. 每周从最小编号开始（01_xxx.md）
2. 按"准备 → 任务卡 → 自检 → 卡点日志"顺序走
3. 卡 5 分钟以上 → 发 AI："给我一个再小一号的练习"
4. 通过自检 → 跳下一篇；不通过 → 留尾巴下次问，先继续

---

## 第一次跑这套？

按顺序走 4 步：

1. [HOW_TO_LEARN_WITH_AI.md](../HOW_TO_LEARN_WITH_AI.md) — 元教程，5 分钟
2. [SETUP.md](../SETUP.md) — 环境配置，~15 分钟
3. [docs/concepts.md](../docs/concepts.md) — 18 个核心术语提前扫一遍（不用记，知道有这页能查就行）
4. 回来这里 → 开始 [week-1-langchain/01_hello_llm.md](week-1-langchain/01_hello_llm.md)

走完 4 周 → 挑一个 [docs/challenges.md](../docs/challenges.md) 做出师作业。
