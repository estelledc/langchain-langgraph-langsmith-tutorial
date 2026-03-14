"""
02_conditional_edges.py — 条件边与 ReAct 循环

知识点：
- add_conditional_edges：根据状态动态路由
- tools_condition：内置的工具调用条件
- ToolNode：内置的工具执行节点
- ReAct 模式：Reasoning + Acting 循环
- MemorySaver：Checkpointer 持久化状态（跨轮对话）
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import math
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0,
)


# ── 工具定义 ─────────────────────────────────────────────────────────────────
@tool
def calculate(expression: str) -> str:
    """计算数学表达式，支持加减乘除和 sqrt/pow 等函数。"""
    try:
        allowed = {"sqrt": math.sqrt, "pow": math.pow, "pi": math.pi, "abs": abs}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@tool
def get_weather(city: str) -> str:
    """获取城市天气（模拟数据）。"""
    data = {
        "北京": "晴，22°C", "上海": "多云，28°C",
        "广州": "阵雨，32°C", "深圳": "晴转多云，30°C",
    }
    return data.get(city, f"{city}暂无数据")


@tool
def word_count(text: str) -> str:
    """统计文本的字数和词数。"""
    chars = len(text)
    words = len(text.split())
    return f"字符数：{chars}，词数（空格分隔）：{words}"


tools = [calculate, get_weather, word_count]
tool_node = ToolNode(tools)                   # 内置工具执行节点
llm_with_tools = llm.bind_tools(tools)        # 将工具绑定到 LLM


# ── 1. 标准 ReAct Agent（使用 tools_condition）───────────────────────────────
def demo_react_agent():
    print("=" * 50)
    print("【ReAct Agent — tools_condition 自动路由】")
    
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    def agent_node(state: AgentState) -> dict:
        """LLM 推理节点：决定是调用工具还是直接回答"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "agent")
    
    # tools_condition 是内置条件函数：
    #   - 如果 LLM 输出包含 tool_calls → 路由到 "tools"
    #   - 否则 → 路由到 END
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")   # 工具执行后回到 agent 继续推理
    
    graph = builder.compile()
    
    questions = [
        "sqrt(196) + pow(3, 4) 等于多少？",
        "北京和上海今天天气如何？哪个城市更热？",
        "统计这段文字的字数：LangGraph 是一个强大的 Agent 框架",
    ]
    
    for q in questions:
        print(f"\n问：{q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        print(f"答：{result['messages'][-1].content}")


# ── 2. 带 Checkpointer 的多轮对话（跨请求保持记忆）─────────────────────────
def demo_persistent_agent():
    print("\n" + "=" * 50)
    print("【带 Checkpointer 的多轮对话】")
    
    class AgentState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    # MemorySaver：将每次执行的状态保存在内存中
    # 生产环境可替换为 SqliteSaver 或 PostgresSaver
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    # thread_id 标识一个对话会话，相同 thread_id 共享历史
    config = {"configurable": {"thread_id": "user_001"}}
    
    def chat(message: str):
        result = graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=config,
        )
        return result["messages"][-1].content
    
    # 多轮对话，每次都能记住上文
    print(f"\n第1轮：北京今天天气怎么样？")
    print(f"AI：{chat('北京今天天气怎么样？')}")
    
    print(f"\n第2轮：那上海呢？")
    print(f"AI：{chat('那上海呢？')}")  # Agent 知道在问天气
    
    print(f"\n第3轮：这两个城市哪个温度更高？")
    print(f"AI：{chat('这两个城市哪个温度更高？')}")  # 能引用前面的天气数据


# ── 3. 自定义路由：多分支条件边 ──────────────────────────────────────────────
def demo_custom_routing():
    print("\n" + "=" * 50)
    print("【自定义多分支路由】")
    
    class RouterState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        intent: str   # 用于存储识别出的意图
    
    def classify_intent(state: RouterState) -> dict:
        """分类用户意图"""
        user_msg = state["messages"][-1].content.lower()
        
        if any(kw in user_msg for kw in ["计算", "等于", "+", "-", "*", "sqrt"]):
            intent = "math"
        elif any(kw in user_msg for kw in ["天气", "气温", "下雨"]):
            intent = "weather"
        elif any(kw in user_msg for kw in ["字数", "统计", "多少字"]):
            intent = "count"
        else:
            intent = "general"
        
        print(f"  [路由] 识别意图：{intent}")
        return {"intent": intent}
    
    def math_handler(state: RouterState) -> dict:
        expr = state["messages"][-1].content
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}
    
    def weather_handler(state: RouterState) -> dict:
        result = llm_with_tools.invoke(state["messages"])
        return {"messages": [result]}
    
    def general_handler(state: RouterState) -> dict:
        result = llm.invoke(state["messages"])
        return {"messages": [result]}
    
    def route_by_intent(state: RouterState) -> str:
        """根据 intent 字段决定下一个节点"""
        intent = state.get("intent", "general")
        if intent in ("math", "weather", "count"):
            return intent
        return "general"
    
    builder = StateGraph(RouterState)
    builder.add_node("classify", classify_intent)
    builder.add_node("math", math_handler)
    builder.add_node("weather", weather_handler)
    builder.add_node("count", general_handler)
    builder.add_node("general", general_handler)
    
    # 工具节点（math 和 weather 可能需要工具）
    builder.add_node("tools", tool_node)
    
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_intent,
        {"math": "math", "weather": "weather", "count": "count", "general": "general"},
    )
    
    # math 和 weather 节点可能调用工具
    builder.add_conditional_edges("math", tools_condition)
    builder.add_conditional_edges("weather", tools_condition)
    builder.add_edge("tools", "general")  # 工具执行后直接结束（简化版）
    
    builder.add_edge("count", END)
    builder.add_edge("general", END)
    
    graph = builder.compile()
    
    test_cases = [
        "sqrt(81) 等于多少？",
        "上海今天天气好吗？",
        "你好，你是谁？",
    ]
    
    for q in test_cases:
        print(f"\n问：{q}")
        result = graph.invoke({"messages": [HumanMessage(content=q)]})
        print(f"答：{result['messages'][-1].content}")


if __name__ == "__main__":
    demo_react_agent()
    demo_persistent_agent()
    demo_custom_routing()
    
    print("\n✅ 条件边示例完成！")
    print("   关键理解：")
    print("   - tools_condition：检查 last message 是否有 tool_calls")
    print("   - MemorySaver：每次 invoke 后自动保存完整 State 快照")
    print("   - thread_id：用于区分不同用户/会话的状态隔离")
