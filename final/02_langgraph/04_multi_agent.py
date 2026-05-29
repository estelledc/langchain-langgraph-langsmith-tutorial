"""
04_multi_agent.py — 多 Agent 子图（Subgraph）

知识点：
- Subgraph：将一个编译好的图作为另一个图的节点
- 父图与子图的 State 共享规则
- Supervisor 模式：一个 Agent 协调多个专业 Agent
- 并行子图：同时运行多个独立任务
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.3,
)


# ── 专业工具 ─────────────────────────────────────────────────────────────────
@tool
def search_papers(topic: str) -> str:
    """搜索学术论文（模拟）。"""
    papers = {
        "transformer": "Attention is All You Need (Vaswani et al., 2017)\nBERT (Devlin et al., 2018)",
        "rag": "RAG (Lewis et al., 2020)\nFusion-in-Decoder (Izacard & Grave, 2020)",
        "agent": "ReAct (Yao et al., 2022)\nToolFormer (Schick et al., 2023)",
    }
    for key, result in papers.items():
        if key in topic.lower():
            return result
    return f"找到若干关于'{topic}'的论文（模拟数据）"


@tool
def write_code(task: str) -> str:
    """编写代码示例（模拟）。"""
    if "hello" in task.lower():
        return 'print("Hello, World!")'
    return f"# {task}\ndef solution():\n    pass  # 实现逻辑"


@tool
def analyze_data(description: str) -> str:
    """分析数据并给出洞察（模拟）。"""
    return f"数据分析结果：{description}相关的关键指标显示正常，建议关注趋势变化。"


# ── 1. Supervisor 模式：主控 Agent 分配任务 ───────────────────────────────────
def demo_supervisor_pattern():
    print("=" * 50)
    print("【Supervisor 模式 — 主控 Agent 协调专业 Agent】")
    
    # ── 子 Agent：研究员 ──
    class ResearchState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    research_tools = [search_papers]
    research_tool_node = ToolNode(research_tools)
    research_llm = llm.bind_tools(research_tools)
    
    def researcher_agent(state: ResearchState) -> dict:
        system = SystemMessage(content="你是专业的研究员，擅长搜索和总结学术资料。")
        response = research_llm.invoke([system] + state["messages"])
        return {"messages": [response]}
    
    research_builder = StateGraph(ResearchState)
    research_builder.add_node("researcher", researcher_agent)
    research_builder.add_node("tools", research_tool_node)
    research_builder.add_edge(START, "researcher")
    research_builder.add_conditional_edges("researcher", tools_condition)
    research_builder.add_edge("tools", "researcher")
    research_subgraph = research_builder.compile()
    
    # ── 子 Agent：程序员 ──
    class CoderState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    code_tools = [write_code]
    code_tool_node = ToolNode(code_tools)
    code_llm = llm.bind_tools(code_tools)
    
    def coder_agent(state: CoderState) -> dict:
        system = SystemMessage(content="你是专业的程序员，擅长编写清晰、高效的代码。")
        response = code_llm.invoke([system] + state["messages"])
        return {"messages": [response]}
    
    coder_builder = StateGraph(CoderState)
    coder_builder.add_node("coder", coder_agent)
    coder_builder.add_node("tools", code_tool_node)
    coder_builder.add_edge(START, "coder")
    coder_builder.add_conditional_edges("coder", tools_condition)
    coder_builder.add_edge("tools", "coder")
    coder_subgraph = coder_builder.compile()
    
    # ── 父图：Supervisor ──
    class SupervisorState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        next_agent: str    # supervisor 决定下一个执行的子 Agent
        task_result: str   # 收集子 Agent 的结果
    
    members = ["researcher", "coder"]
    
    def supervisor(state: SupervisorState) -> dict:
        """决定下一步交给哪个子 Agent，或直接回答"""
        system_prompt = f"""你是一个任务协调员，管理以下专家团队：
- researcher：负责搜索和研究学术资料
- coder：负责编写代码
- FINISH：所有任务已完成

根据用户需求，决定下一步交给哪位专家，或者回答 FINISH。
只输出一个词：{', '.join(members + ['FINISH'])}"""
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
        ] + state["messages"])
        
        content = response.content.strip()
        next_agent = content if content in members else "FINISH"
        print(f"\n  [Supervisor] 决定：{next_agent}")
        return {"next_agent": next_agent, "messages": [response]}
    
    def run_researcher(state: SupervisorState) -> dict:
        """调用研究员子图"""
        print("  [Researcher] 开始工作...")
        result = research_subgraph.invoke({"messages": state["messages"]})
        last_msg = result["messages"][-1]
        print(f"  [Researcher] 完成：{last_msg.content[:80]}...")
        return {"messages": [last_msg], "task_result": last_msg.content}
    
    def run_coder(state: SupervisorState) -> dict:
        """调用程序员子图"""
        print("  [Coder] 开始工作...")
        result = coder_subgraph.invoke({"messages": state["messages"]})
        last_msg = result["messages"][-1]
        print(f"  [Coder] 完成：{last_msg.content[:80]}...")
        return {"messages": [last_msg], "task_result": last_msg.content}
    
    def route_to_agent(state: SupervisorState) -> str:
        return state.get("next_agent", "FINISH")
    
    supervisor_builder = StateGraph(SupervisorState)
    supervisor_builder.add_node("supervisor", supervisor)
    supervisor_builder.add_node("researcher", run_researcher)
    supervisor_builder.add_node("coder", run_coder)
    
    supervisor_builder.add_edge(START, "supervisor")
    supervisor_builder.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {"researcher": "researcher", "coder": "coder", "FINISH": END},
    )
    supervisor_builder.add_edge("researcher", "supervisor")
    supervisor_builder.add_edge("coder", "supervisor")
    
    supervisor_graph = supervisor_builder.compile()
    
    # 测试：需要多个专家协作的任务
    task = "我想了解 RAG 技术，并且需要一个 Python 示例代码"
    print(f"\n任务：{task}\n")
    
    result = supervisor_graph.invoke({
        "messages": [HumanMessage(content=task)],
        "next_agent": "",
        "task_result": "",
    })
    print(f"\n最终结果（最后一条消息）：\n{result['messages'][-1].content[:200]}...")


# ── 2. 并行子图：同时执行多个独立任务 ────────────────────────────────────────
def demo_parallel_subgraphs():
    print("\n" + "=" * 50)
    print("【并行子图 — 同时执行多个独立任务】")
    
    class ParallelState(TypedDict):
        topic: str
        research_result: str
        analysis_result: str
        summary: str
    
    def research_task(state: ParallelState) -> dict:
        """研究任务（并行执行之一）"""
        response = llm.invoke([
            HumanMessage(content=f"用2句话介绍 {state['topic']} 的技术背景")
        ])
        return {"research_result": response.content}
    
    def analysis_task(state: ParallelState) -> dict:
        """分析任务（并行执行之二）"""
        response = llm.invoke([
            HumanMessage(content=f"列举 {state['topic']} 的3个应用场景，每条一行")
        ])
        return {"analysis_result": response.content}
    
    def synthesize(state: ParallelState) -> dict:
        """汇总两个并行任务的结果"""
        combined = f"""
技术背景：
{state['research_result']}

应用场景：
{state['analysis_result']}
""".strip()
        return {"summary": combined}
    
    builder = StateGraph(ParallelState)
    builder.add_node("research", research_task)
    builder.add_node("analysis", analysis_task)
    builder.add_node("synthesize", synthesize)
    
    # 从 START 同时连接两个节点（并行）
    builder.add_edge(START, "research")
    builder.add_edge(START, "analysis")
    # 两个节点都完成后才执行 synthesize
    builder.add_edge(["research", "analysis"], "synthesize")
    builder.add_edge("synthesize", END)
    
    graph = builder.compile()
    
    print("\n主题：向量数据库")
    result = graph.invoke({"topic": "向量数据库"})
    print(f"\n综合报告：\n{result['summary']}")


if __name__ == "__main__":
    demo_supervisor_pattern()
    demo_parallel_subgraphs()
    
    print("\n✅ 多 Agent 示例完成！")
    print("   两种常见的多 Agent 模式：")
    print("   1. Supervisor 模式：中央协调员动态分配任务给专业子 Agent")
    print("   2. 并行模式：多个独立任务同时执行，最后汇总结果")
