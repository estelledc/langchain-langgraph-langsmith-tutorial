---
name: LangChain 学习计划
overview: 基于已配置好的 DashScope（阿里云通义千问）+ LangSmith 环境，从零搭建项目结构，分阶段学习 LangChain → LangGraph → LangSmith，最终完成一个完整的 AI Agent 项目。
todos:
  - id: setup
    content: 创建 requirements.txt、README.md 和项目目录骨架
    status: completed
  - id: week1-basic
    content: 01_langchain/：完成 Hello LLM、PromptTemplate、LCEL 链路三个示例文件
    status: completed
  - id: week1-memory
    content: 01_langchain/：完成 Memory 对话历史示例
    status: completed
  - id: week2-rag
    content: 01_langchain/：完成 RAG 基础示例（FAISS + 检索链）
    status: completed
  - id: week2-tool
    content: 01_langchain/：完成 Tool & AgentExecutor 示例，并在 LangSmith 查看 Trace
    status: completed
  - id: week3-graph
    content: 02_langgraph/：完成 StateGraph 基础 + 条件边 + HITL 三个示例
    status: completed
  - id: week4-langsmith
    content: 03_langsmith/：完成追踪、数据集评估示例
    status: completed
  - id: week4-project
    content: 04_project/：实现研究助手 Agent 综合项目
    status: completed
isProject: false
---

# LangChain / LangGraph / LangSmith 系统学习计划

## 环境概况

- **LLM 后端**：DashScope API（通义千问 `qwen-`* 系列）
- **可观测性**：LangSmith（`LANGCHAIN_PROJECT=study` 已配置）
- **节奏**：每周 10+ 小时，约 **4 周**完成全部阶段

---

## 项目结构（最终形态）

```
langSmith/
├── .env
├── requirements.txt
├── README.md
├── 01_langchain/          # 第一阶段练习
│   ├── 01_hello_llm.py
│   ├── 02_prompt_template.py
│   ├── 03_chains.py
│   ├── 04_memory.py
│   └── 05_rag_basic.py
├── 02_langgraph/          # 第二阶段练习
│   ├── 01_simple_graph.py
│   ├── 02_conditional_edges.py
│   ├── 03_human_in_the_loop.py
│   └── 04_multi_agent.py
├── 03_langsmith/          # 第三阶段练习
│   ├── 01_tracing.py
│   ├── 02_evaluation.py
│   └── 03_dataset.py
└── 04_project/            # 最终综合项目：研究助手 Agent
    ├── agent.py
    ├── tools.py
    ├── graph.py
    └── eval.py
```

---

## 架构数据流（最终项目）

```mermaid
flowchart TD
    userInput["用户问题"] --> graph["LangGraph StateGraph"]
    graph --> planner["Planner 节点\nLangChain Chain"]
    planner --> toolNode["Tool 节点\n搜索/计算/代码执行"]
    toolNode --> reflect["Reflect 节点\n判断是否完成"]
    reflect -->|"需要继续"| planner
    reflect -->|"完成"| output["最终回答"]
    graph --> langSmith["LangSmith 全链路追踪"]
    langSmith --> trace["Trace/Eval/Dataset"]
```



---

## 第一阶段：LangChain 核心（第 1-2 周）

**目标**：掌握 LangChain 核心抽象，能独立构建 RAG 链路

### Week 1 — 基础链路

- **Day 1-2**：环境搭建 + Hello LLM
  - 安装 `langchain`, `langchain-community`, `langchain-openai`（兼容 DashScope OpenAI 格式）
  - 用 `ChatOpenAI(base_url=..., api_key=...)` 接通通义千问
  - 核心概念：`BaseChatModel`, `HumanMessage`, `AIMessage`
- **Day 3-4**：Prompt Template & LCEL
  - `ChatPromptTemplate`, `PromptTemplate`
  - LCEL 管道：`prompt | llm | StrOutputParser()`
  - 流式输出 `stream()`
- **Day 5-7**：Memory & 对话历史
  - `ConversationBufferMemory`, `ConversationSummaryMemory`
  - `RunnableWithMessageHistory`（新式 API）

### Week 2 — RAG 与工具

- **Day 8-10**：RAG 基础
  - `RecursiveCharacterTextSplitter` 文档切分
  - `FAISS` 本地向量库（或 `Chroma`）
  - `create_retrieval_chain` 组装检索增强链
- **Day 11-14**：Tool & Agent（旧式，打基础）
  - `@tool` 装饰器定义工具
  - `create_react_agent` + `AgentExecutor`
  - 观察 LangSmith Trace 界面

**关键文件**：`[.env](.env)` 中的 `DASHSCOPE_BASE_URL` 和 `LANGCHAIN_API_KEY` 全程生效

---

## 第二阶段：LangGraph 状态图（第 3 周）

**目标**：理解 StateGraph 控制流，实现带循环和人工审核的 Agent

- **Day 15-16**：StateGraph 基础
  - `StateGraph`, `TypedDict` 定义 State
  - `add_node`, `add_edge`, `compile()`
  - `graph.invoke()` vs `graph.stream()`
- **Day 17-18**：条件边与循环
  - `add_conditional_edges` 实现 ReAct 循环
  - `tools_condition` 内置条件
- **Day 19-21**：高级特性
  - **Checkpointer**（`MemorySaver`）持久化状态
  - **Human-in-the-loop**：`interrupt_before` 暂停等待人工输入
  - **子图（Subgraph）**：模块化多 Agent

---

## 第三阶段：LangSmith 可观测性（第 4 周前半）

**目标**：能用 LangSmith 做追踪、评估、数据集管理

- **追踪**：理解 Run Tree 结构，自定义 `@traceable` 装饰器
- **评估（Evaluation）**：
  - 创建 `Dataset`（问答对）
  - `evaluate()` 跑批量评估
  - 使用 LLM-as-Judge 评分器
- **Feedback**：手动打分与自动化质量监控

**仪表板**：访问 [smith.langchain.com](https://smith.langchain.com) 查看 `study` 项目的所有 Trace

---

## 第四阶段：综合项目——研究助手 Agent（第 4 周后半）

**目标**：整合三个库，完成一个端到端可运行、可观测的 Agent

**功能**：接收研究问题 → 自动拆解子任务 → 调用搜索/代码工具 → 反思迭代 → 输出报告

- 用 **LangGraph** 构建多轮 ReAct 状态图
- 用 **LangChain** 封装工具和 Prompt
- 用 **LangSmith** 追踪每次运行并评估输出质量

---

## 依赖清单（requirements.txt）

```
langchain>=0.3
langchain-openai>=0.2
langchain-community>=0.3
langgraph>=0.2
langsmith>=0.1
faiss-cpu
python-dotenv
```

---

## 学习资源

- LangChain 官方文档：[python.langchain.com](https://python.langchain.com)
- LangGraph 官方文档：[langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph)
- LangSmith 官方文档：[docs.smith.langchain.com](https://docs.smith.langchain.com)
- 通义千问 OpenAI 兼容接口：[dashscope 文档](https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope)

