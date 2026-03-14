"""
05_rag_basic.py — RAG（检索增强生成）基础

知识点：
- RecursiveCharacterTextSplitter：文档切分
- DashScopeEmbeddings / OpenAIEmbeddings（兼容模式）：文本向量化
- FAISS：本地向量数据库的存储与检索
- create_retrieval_chain：组装完整的 RAG 链
- with_sources：返回来源文档
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── LLM ──────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0,  # RAG 场景建议设为 0，减少幻觉
)

# ── Embeddings（使用 DashScope 兼容 OpenAI 的 embedding 接口）──────────────
# DashScope 支持 text-embedding-v3 等模型
embeddings = OpenAIEmbeddings(
    model="text-embedding-v3",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
)


# ── 1. 文档准备与切分 ─────────────────────────────────────────────────────────
def build_knowledge_base() -> FAISS:
    """构建本地向量知识库"""
    
    # 模拟知识库文档（实际场景可从文件/数据库加载）
    raw_docs = [
        Document(
            page_content="""
LangChain 是一个用于构建 LLM 应用的开源框架，由 Harrison Chase 于2022年创建。
它提供了一套标准化的接口，将 LLM、提示模板、记忆、工具和链路整合在一起。
LangChain 支持 Python 和 JavaScript 两种语言，是目前最流行的 LLM 应用开发框架之一。
核心组件包括：Model I/O（模型输入输出）、Chains（链）、Agents（代理）、Memory（记忆）、Callbacks（回调）。
            """.strip(),
            metadata={"source": "langchain_intro.txt", "topic": "LangChain"},
        ),
        Document(
            page_content="""
LangGraph 是 LangChain 生态中用于构建有状态、多步骤 Agent 的库。
它基于图（Graph）的概念，将 Agent 的执行流程建模为节点（Node）和边（Edge）的有向图。
LangGraph 的核心特性：
1. StateGraph：有状态的图，每个节点可以读写共享状态
2. 条件边（Conditional Edges）：根据状态动态决定下一步执行哪个节点
3. Checkpointer：支持将状态持久化到数据库，实现长时运行和断点续传
4. Human-in-the-loop：在执行过程中暂停并等待人工介入
LangGraph 特别适合构建需要多轮工具调用、反思迭代的复杂 Agent 系统。
            """.strip(),
            metadata={"source": "langgraph_intro.txt", "topic": "LangGraph"},
        ),
        Document(
            page_content="""
LangSmith 是 LangChain 提供的 LLM 应用可观测性平台。
主要功能：
- Tracing（追踪）：记录每次 LLM 调用的完整输入输出、耗时、Token 消耗
- Evaluation（评估）：对 LLM 应用进行批量测试和质量评估
- Dataset（数据集）：管理测试用例和评估数据集
- Monitoring（监控）：生产环境的实时质量监控和告警
LangSmith 提供免费层，每月 5000 次 Trace，适合个人学习和小型项目。
配置方式：设置 LANGCHAIN_TRACING_V2=true 和 LANGCHAIN_API_KEY 环境变量即可自动启用。
            """.strip(),
            metadata={"source": "langsmith_intro.txt", "topic": "LangSmith"},
        ),
        Document(
            page_content="""
RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与生成模型结合的技术。
RAG 的核心流程：
1. 索引阶段：将文档切分为小块（Chunk），用 Embedding 模型转为向量，存入向量数据库
2. 检索阶段：将用户问题转为向量，在向量数据库中找到最相似的文档块
3. 生成阶段：将检索到的文档块作为上下文，连同用户问题一起传给 LLM 生成答案
RAG 可以有效解决 LLM 的知识截止问题，并减少幻觉（Hallucination）。
常用向量数据库：FAISS（本地）、Chroma（本地）、Pinecone（云端）、Weaviate（云端）。
            """.strip(),
            metadata={"source": "rag_intro.txt", "topic": "RAG"},
        ),
        Document(
            page_content="""
LCEL（LangChain Expression Language）是 LangChain 提供的声明式链路构建语法。
使用 | 操作符将各组件串联，形成一个 Runnable 对象。
LCEL 的优势：
- 统一接口：所有组件都实现 Runnable 协议，支持 invoke/stream/batch
- 流式支持：天然支持流式输出，无需额外配置
- 并行执行：RunnableParallel 自动并行执行独立的链路
- 易于调试：与 LangSmith 深度集成，每个步骤都有独立的 Trace
典型用法：chain = prompt | llm | StrOutputParser()
            """.strip(),
            metadata={"source": "lcel_intro.txt", "topic": "LCEL"},
        ),
    ]
    
    # 文本切分：将长文档切成小块
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,       # 每块最多 300 字符
        chunk_overlap=50,     # 块间重叠 50 字符，避免切断关键信息
        separators=["\n\n", "\n", "。", "，", " ", ""],  # 中文优先按段落/句子切分
    )
    
    chunks = splitter.split_documents(raw_docs)
    print(f"原始文档：{len(raw_docs)} 篇，切分后：{len(chunks)} 个块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"  块 {i}: [{chunk.metadata['topic']}] {chunk.page_content[:50]}...")
    
    # 构建 FAISS 向量索引
    print("\n正在构建向量索引（需要调用 Embedding API）...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("向量索引构建完成！")
    
    return vectorstore


# ── 2. 基础 RAG 链 ────────────────────────────────────────────────────────────
def demo_basic_rag(vectorstore: FAISS):
    print("\n" + "=" * 50)
    print("【基础 RAG 链】")
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",   # 相似度检索
        search_kwargs={"k": 2},     # 返回最相似的 2 个块
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个知识库问答助手。请根据以下检索到的上下文回答用户问题。
如果上下文中没有相关信息，请明确说明"知识库中没有相关信息"，不要编造答案。

上下文：
{context}"""),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(f"[来源：{d.metadata['source']}]\n{d.page_content}" for d in docs)
    
    # LCEL 组装 RAG 链
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    questions = [
        "LangGraph 有哪些核心特性？",
        "LangSmith 的免费层有什么限制？",
        "RAG 的主要流程是什么？",
    ]
    
    for q in questions:
        print(f"\n问：{q}")
        answer = rag_chain.invoke(q)
        print(f"答：{answer}")


# ── 3. 带来源的 RAG（返回检索到的文档）─────────────────────────────────────
def demo_rag_with_sources(vectorstore: FAISS):
    print("\n" + "=" * 50)
    print("【带来源信息的 RAG】")
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据上下文回答问题，保持简洁。\n\n上下文：{context}"),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    # 同时返回答案和来源文档
    rag_chain_with_source = RunnablePassthrough.assign(
        context=lambda x: format_docs(retriever.invoke(x["input"])),
        source_docs=lambda x: retriever.invoke(x["input"]),
    ) | {
        "answer": prompt | llm | StrOutputParser(),
        "source_docs": lambda x: x["source_docs"],
    }
    
    result = rag_chain_with_source.invoke({"input": "LCEL 的主要优势是什么？"})
    
    print(f"答案：{result['answer']}")
    print(f"\n参考来源：")
    for doc in result["source_docs"]:
        print(f"  - {doc.metadata['source']} [{doc.metadata['topic']}]")


# ── 4. 向量相似度搜索演示 ──────────────────────────────────────────────────
def demo_similarity_search(vectorstore: FAISS):
    print("\n" + "=" * 50)
    print("【向量相似度搜索】")
    
    query = "如何监控 LLM 应用的质量？"
    
    # 返回文档及相似度分数
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=3)
    
    print(f"查询：{query}")
    print("最相关的文档块：")
    for doc, score in docs_with_scores:
        # FAISS 返回的是 L2 距离，越小越相似
        print(f"  相似度分数（L2距离）：{score:.4f}")
        print(f"  来源：{doc.metadata['source']}")
        print(f"  内容：{doc.page_content[:80]}...\n")


if __name__ == "__main__":
    print("=" * 50)
    print("【步骤 1：构建知识库】")
    vectorstore = build_knowledge_base()
    
    demo_basic_rag(vectorstore)
    demo_rag_with_sources(vectorstore)
    demo_similarity_search(vectorstore)
    
    print("\n✅ RAG 示例完成！")
    print("   提示：在 LangSmith 可以看到 Retriever 和 LLM 的独立 Trace，")
    print("   方便分析检索质量和生成质量。")
