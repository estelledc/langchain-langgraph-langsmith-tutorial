# LangChain / LangGraph / LangSmith 学习教程

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/estelledc/langchain-langgraph-langsmith-tutorial)

基于 **DashScope（通义千问）** 作为 LLM 后端，配合 **LangSmith** 可观测性平台，系统学习 LangChain、LangGraph、LangSmith 三大核心库。全部示例可独立运行，适合零基础到进阶的渐进式学习。

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/<your-username>/langchain-langgraph-langsmith-tutorial.git
cd langchain-langgraph-langsmith-tutorial

# 2. 复制环境变量模板并填入你自己的 Key
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY 和 LANGCHAIN_API_KEY

# 3. 安装依赖（建议使用虚拟环境）
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 4. 运行第一个示例
python 01_langchain/01_hello_llm.py
```

运行后可在 [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目中查看完整 Trace。

## 申请 API Key

| 服务 | 用途 | 申请地址 |
|------|------|---------|
| DashScope | LLM 调用（必填） | [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com) |
| LangSmith | Trace 可观测性（必填） | [smith.langchain.com](https://smith.langchain.com)（免费层 5K runs/月） |

## 目录结构

```
langchain-langgraph-langsmith-tutorial/
├── .env.example           # 环境变量模板（复制为 .env 后填写真实 Key）
├── requirements.txt       # Python 依赖
├── README.md
├── 01_langchain/          # 第一阶段：LangChain 核心（Week 1-2）
│   ├── 01_hello_llm.py         # LLM 基础调用（invoke / stream / batch）
│   ├── 02_prompt_template.py   # Prompt 模板（ChatPromptTemplate / LCEL）
│   ├── 03_chains.py            # LCEL 链路组合
│   ├── 04_memory.py            # 对话记忆（RunnableWithMessageHistory）
│   ├── 05_rag_basic.py         # RAG 检索增强（FAISS + 检索链）
│   └── 06_tools_agent.py       # Tool & AgentExecutor
├── 02_langgraph/          # 第二阶段：LangGraph 状态图（Week 3）
│   ├── 01_simple_graph.py         # 基础 StateGraph
│   ├── 02_conditional_edges.py    # 条件边与循环
│   ├── 03_human_in_the_loop.py    # 人工审核 HITL
│   └── 04_multi_agent.py          # 多 Agent 子图
├── 03_langsmith/          # 第三阶段：LangSmith 可观测性（Week 4 前半）
│   ├── 01_tracing.py       # 自定义追踪（@traceable）
│   ├── 02_evaluation.py    # 批量评估（LLM-as-Judge）
│   └── 03_dataset.py       # 数据集管理
└── 04_project/            # 第四阶段：综合项目——研究助手 Agent（Week 4 后半）
    ├── agent.py    # Agent 入口
    ├── tools.py    # 工具定义
    ├── graph.py    # LangGraph 状态图
    └── eval.py     # LangSmith 评估
```

## 学习路线

| 阶段 | 内容 | 目录 | 周次 |
|------|------|------|------|
| 1 | LangChain 核心：LLM、Prompt、LCEL、Memory、RAG、Agent | `01_langchain/` | Week 1-2 |
| 2 | LangGraph：StateGraph、条件边、HITL、多 Agent | `02_langgraph/` | Week 3 |
| 3 | LangSmith：追踪、评估、数据集 | `03_langsmith/` | Week 4 前半 |
| 4 | 综合项目：研究助手 Agent | `04_project/` | Week 4 后半 |

## 运行示例

```bash
# 按阶段逐步运行
python 01_langchain/01_hello_llm.py
python 01_langchain/02_prompt_template.py
# ...以此类推
```

每个文件顶部都有详细的知识点说明和注释，可直接阅读源码学习。

## 参考资料

- [LangChain 文档](https://python.langchain.com)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph)
- [LangSmith 文档](https://docs.smith.langchain.com)
- [DashScope OpenAI 兼容接口](https://help.aliyun.com/zh/model-studio/developer-reference/compatibility-of-openai-with-dashscope)

## 贡献 & 许可

欢迎提 Issue 或 PR 补充示例、修正错误。本项目基于 [MIT License](LICENSE) 开源。
