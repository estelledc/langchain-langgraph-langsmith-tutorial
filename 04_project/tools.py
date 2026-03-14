"""
tools.py — 研究助手 Agent 的工具集

包含：搜索、计算、文本分析、知识库检索等工具
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import math
import json
from datetime import datetime
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """
    搜索互联网获取最新信息。适合查找新闻、事件、技术文章等。
    
    Args:
        query: 搜索关键词或问题
    """
    # 实际场景接入 Tavily / SerpAPI / Bing Search API
    # 这里用模拟数据演示
    knowledge_base = {
        "langchain": """
LangChain 最新动态（2024）：
- v0.3 发布，LCEL 成为核心构建方式
- LangGraph 独立为专门的 Agent 框架
- 与 LangSmith 深度集成，提供完整的 LLMOps 工具链
- 支持 100+ LLM 提供商
""",
        "langgraph": """
LangGraph 技术特性：
- 基于有向图的状态机模型
- 原生支持流式输出和断点续传
- Checkpointer 支持 SQLite/PostgreSQL
- 内置 Human-in-the-loop 机制
- 支持子图和多 Agent 协作
""",
        "人工智能": """
AI 行业 2024 年趋势：
- 大模型推理效率持续提升（MoE、量化技术）
- Agent 和 RAG 成为主流应用范式
- 多模态模型快速发展
- 模型安全和对齐研究受到重视
""",
        "向量数据库": """
主流向量数据库对比：
- FAISS：Facebook 出品，本地部署，高性能
- Chroma：开源，轻量，适合原型开发
- Pinecone：云端全托管，自动扩缩容
- Weaviate：支持混合检索，生产级别
- Milvus：开源分布式，适合大规模场景
""",
    }
    
    result = []
    for key, content in knowledge_base.items():
        if key in query.lower() or query.lower() in key:
            result.append(content.strip())
    
    if result:
        return "\n\n".join(result)
    return f"搜索'{query}'：找到若干相关资料，建议进一步查阅官方文档和最新论文。"


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。支持基本运算和 sqrt、pow、log 等函数。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "sqrt(16)"
    """
    try:
        allowed = {
            "sqrt": math.sqrt, "pow": math.pow, "log": math.log,
            "log2": math.log2, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "pi": math.pi, "e": math.e, "abs": abs, "round": round,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def summarize_text(text: str, max_words: int = 100) -> str:
    """
    对长文本进行摘要提炼。
    
    Args:
        text: 需要摘要的原文
        max_words: 摘要最大字数（默认100字）
    """
    # 简单实现：截取前 max_words 个字符并加省略号
    if len(text) <= max_words:
        return text
    return text[:max_words] + "..."


@tool  
def get_current_date() -> str:
    """获取当前日期和时间。"""
    now = datetime.now()
    return now.strftime("%Y年%m月%d日 %H:%M，星期" + "一二三四五六日"[now.weekday()])


@tool
def structure_report(
    title: str,
    sections: str,
    conclusion: str,
) -> str:
    """
    将研究内容整理成结构化报告格式。
    
    Args:
        title: 报告标题
        sections: 各章节内容（JSON 格式的字符串，如 '[{"heading": "...", "content": "..."}]'）
        conclusion: 总结结论
    """
    try:
        sections_list = json.loads(sections)
    except (json.JSONDecodeError, TypeError):
        # 如果解析失败，把 sections 当作普通字符串处理
        sections_list = [{"heading": "研究内容", "content": str(sections)}]
    
    report_lines = [
        f"# {title}",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    
    for i, section in enumerate(sections_list, 1):
        heading = section.get("heading", f"第{i}部分")
        content = section.get("content", "")
        report_lines.extend([f"## {i}. {heading}", content, ""])
    
    report_lines.extend(["## 结论", conclusion])
    
    return "\n".join(report_lines)


# 所有工具的列表
ALL_TOOLS = [web_search, calculate, summarize_text, get_current_date, structure_report]
