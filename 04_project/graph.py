"""
graph.py — 研究助手 Agent 的 LangGraph 状态图

架构：
用户问题 → Planner（拆解任务）→ ReAct 循环（搜索/计算/分析）→ Writer（生成报告）→ 输出
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.3,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)
tool_node = ToolNode(ALL_TOOLS)


# ── State 定义 ────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    research_plan: str        # Planner 制定的研究计划
    collected_info: str       # ReAct 循环收集的信息
    final_report: str         # 最终报告
    iteration_count: int      # ReAct 循环次数（防止无限循环）


# ── 节点函数 ──────────────────────────────────────────────────────────────────
def planner_node(state: ResearchState) -> dict:
    """
    规划节点：分析用户问题，制定研究计划
    """
    user_question = state["messages"][-1].content
    
    system_prompt = """你是一个研究助手的规划模块。
你的任务是分析用户的研究问题，制定一个清晰的研究计划。

研究计划格式：
1. 核心问题：[一句话概括要研究什么]
2. 子任务：[列出2-4个具体的信息收集任务]
3. 所需工具：[web_search / calculate / summarize_text / get_current_date / structure_report]

请保持计划简洁，每个子任务一行。"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"请为以下问题制定研究计划：\n{user_question}"),
    ])
    
    print(f"\n[Planner] 研究计划：\n{response.content}\n")
    
    return {
        "research_plan": response.content,
        "iteration_count": 0,
        "messages": [
            SystemMessage(content=f"研究计划已制定：\n{response.content}\n\n请按计划逐步收集信息。")
        ],
    }


def researcher_node(state: ResearchState) -> dict:
    """
    研究节点（ReAct 循环的 Reasoning 部分）：
    根据计划调用工具收集信息
    """
    system = SystemMessage(content=f"""你是一个研究助手，正在执行以下研究计划：

{state.get('research_plan', '收集相关信息')}

请使用可用工具收集必要信息。当你认为信息足够充分时，
输出"[信息收集完毕]"并总结已收集的关键信息，不要再调用工具。""")
    
    messages = [system] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    iteration = state.get("iteration_count", 0) + 1
    print(f"\n[Researcher] 第 {iteration} 轮推理")
    
    return {
        "messages": [response],
        "iteration_count": iteration,
    }


def writer_node(state: ResearchState) -> dict:
    """
    写作节点：根据收集的信息生成最终报告
    """
    # 提取所有收集到的信息
    collected = []
    for msg in state["messages"]:
        if hasattr(msg, "content") and msg.content:
            collected.append(msg.content)
    
    all_info = "\n\n".join(collected[-6:])  # 取最近6条消息
    
    user_question = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    system_prompt = """你是一个专业的研究报告撰写专家。
根据收集到的信息，生成一份结构清晰、内容充实的研究报告。

报告要求：
- 标题醒目，反映核心内容
- 分2-3个主要章节，每节有实质内容
- 结论部分给出明确的见解和建议
- 语言专业但易懂，适合技术读者
- 总长度约300-500字"""
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"研究问题：{user_question}\n\n收集的信息：\n{all_info}\n\n请生成研究报告。"),
    ])
    
    report = response.content
    print(f"\n[Writer] 报告已生成（{len(report)} 字）")
    
    return {
        "final_report": report,
        "messages": [response],
    }


# ── 条件函数 ──────────────────────────────────────────────────────────────────
def should_continue_research(state: ResearchState) -> str:
    """
    决定是继续使用工具、完成研究，还是强制退出
    """
    last_message = state["messages"][-1]
    iteration = state.get("iteration_count", 0)
    
    # 超过最大轮次，强制退出
    if iteration >= 4:
        print("[Router] 达到最大迭代次数，进入写作阶段")
        return "write"
    
    # 如果 LLM 决定调用工具
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"[Router] 调用工具：{[tc['name'] for tc in last_message.tool_calls]}")
        return "use_tools"
    
    # 如果 LLM 输出了完成信号
    if "[信息收集完毕]" in (last_message.content or ""):
        print("[Router] 信息收集完毕，进入写作阶段")
        return "write"
    
    # 继续研究
    return "write"


# ── 构建图 ────────────────────────────────────────────────────────────────────
def build_research_graph(with_memory: bool = True):
    """
    构建研究助手的 StateGraph
    
    Args:
        with_memory: 是否启用 Checkpointer（支持多轮对话）
    """
    builder = StateGraph(ResearchState)
    
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("tools", tool_node)
    builder.add_node("writer", writer_node)
    
    # 连接边
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "researcher")
    
    builder.add_conditional_edges(
        "researcher",
        should_continue_research,
        {
            "use_tools": "tools",
            "write": "writer",
        }
    )
    
    builder.add_edge("tools", "researcher")   # 工具执行后返回 researcher
    builder.add_edge("writer", END)
    
    if with_memory:
        memory = MemorySaver()
        return builder.compile(checkpointer=memory)
    
    return builder.compile()
