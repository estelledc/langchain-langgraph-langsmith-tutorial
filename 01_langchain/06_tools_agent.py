"""
06_tools_agent.py — Tool 定义与 Agent（ReAct 模式）

知识点：
- @tool 装饰器定义工具
- StructuredTool：复杂参数工具
- create_react_agent（langgraph 版，推荐）
- AgentExecutor（旧式，了解原理）
- 工具调用的 LangSmith Trace 分析
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import math
import json
import random
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0,
)


# ── 1. 用 @tool 装饰器定义工具 ───────────────────────────────────────────────
@tool
def get_current_time() -> str:
    """获取当前日期和时间。当用户询问现在几点或今天是什么日期时使用。"""
    now = datetime.now()
    return now.strftime("现在是 %Y年%m月%d日 %H:%M:%S，星期" + "一二三四五六日"[now.weekday()])


@tool
def calculate(expression: str) -> str:
    """
    计算数学表达式。支持加减乘除、幂运算、平方根等。
    
    Args:
        expression: 数学表达式字符串，例如 "2 + 3 * 4" 或 "sqrt(16)"
    """
    try:
        # 安全的数学运算（只允许数学函数）
        allowed_names = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "pi": math.pi,
            "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


@tool
def search_weather(city: str) -> str:
    """
    查询指定城市的天气信息（模拟数据）。
    
    Args:
        city: 城市名称，例如 "北京"、"上海"
    """
    # 实际场景接入天气 API，这里用模拟数据
    weather_data = {
        "北京": {"temp": 22, "condition": "晴", "humidity": 45},
        "上海": {"temp": 28, "condition": "多云", "humidity": 70},
        "广州": {"temp": 32, "condition": "阵雨", "humidity": 85},
        "深圳": {"temp": 30, "condition": "晴转多云", "humidity": 75},
        "成都": {"temp": 25, "condition": "阴", "humidity": 65},
    }
    
    if city in weather_data:
        w = weather_data[city]
        return f"{city}天气：{w['condition']}，气温 {w['temp']}°C，湿度 {w['humidity']}%"
    else:
        return f"暂无{city}的天气数据"


@tool
def search_knowledge(query: str) -> str:
    """
    在知识库中搜索相关信息。当用户询问 LangChain/LangGraph/LangSmith 相关知识时使用。
    
    Args:
        query: 搜索关键词或问题
    """
    knowledge = {
        "langchain": "LangChain 是构建 LLM 应用的框架，提供链路、工具、记忆等核心抽象。",
        "langgraph": "LangGraph 是构建有状态 Agent 的库，基于有向图管理复杂执行流程。",
        "langsmith": "LangSmith 是 LLM 应用的可观测性平台，提供追踪、评估、监控功能。",
        "rag": "RAG（检索增强生成）将向量检索与 LLM 结合，有效减少幻觉并支持知识更新。",
        "lcel": "LCEL（LangChain 表达式语言）用 | 操作符串联组件，支持流式和并行执行。",
    }
    
    query_lower = query.lower()
    results = []
    for key, value in knowledge.items():
        if key in query_lower or query_lower in key:
            results.append(value)
    
    return "\n".join(results) if results else f"未找到关于'{query}'的相关信息"


# ── 2. StructuredTool：复杂参数工具 ──────────────────────────────────────────
class UnitConverterInput(BaseModel):
    value: float = Field(description="要转换的数值")
    from_unit: str = Field(description="原单位，如 km、mile、kg、lb、celsius、fahrenheit")
    to_unit: str = Field(description="目标单位，如 km、mile、kg、lb、celsius、fahrenheit")


def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """单位换算工具"""
    conversions = {
        ("km", "mile"): lambda x: x * 0.621371,
        ("mile", "km"): lambda x: x * 1.60934,
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
    }
    
    key = (from_unit.lower(), to_unit.lower())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"不支持 {from_unit} 到 {to_unit} 的换算"


unit_converter_tool = StructuredTool.from_function(
    func=unit_converter,
    name="unit_converter",
    description="单位换算工具，支持长度（km/mile）、重量（kg/lb）、温度（celsius/fahrenheit）换算",
    args_schema=UnitConverterInput,
)

# 所有工具列表
tools = [get_current_time, calculate, search_weather, search_knowledge, unit_converter_tool]


# ── 3. 新式 Agent（tool_calling，推荐）───────────────────────────────────────
def demo_tool_calling_agent():
    print("=" * 50)
    print("【Tool Calling Agent（推荐方式）】")
    
    # 系统提示 + 工具调用占位
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个智能助手，可以使用工具来帮助用户解决问题。请根据需要调用合适的工具。"),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # create_tool_calling_agent 使用模型原生的 tool_use 能力
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,        # 打印每步的思考过程
        max_iterations=5,    # 防止无限循环
    )
    
    test_questions = [
        "现在几点了？北京今天天气怎么样？",
        "帮我算一下 sqrt(144) + 3^4 等于多少？",
        "100摄氏度等于多少华氏度？",
        "LangSmith 是什么？它有什么主要功能？",
    ]
    
    for question in test_questions:
        print(f"\n{'─' * 40}")
        print(f"问题：{question}")
        result = agent_executor.invoke({"input": question})
        print(f"最终答案：{result['output']}")


# ── 4. 查看工具的元数据 ───────────────────────────────────────────────────────
def demo_tool_metadata():
    print("\n" + "=" * 50)
    print("【工具元数据】")
    
    for t in tools:
        print(f"\n工具名：{t.name}")
        print(f"描述：{t.description}")
        if hasattr(t, 'args_schema') and t.args_schema:
            print(f"参数：{t.args_schema.schema()}")


if __name__ == "__main__":
    demo_tool_metadata()
    demo_tool_calling_agent()
    
    print("\n✅ Tools & Agent 示例完成！")
    print("   重点：前往 LangSmith 查看 Agent 的完整执行树：")
    print("   - 每次工具调用都有独立的 Trace 节点")
    print("   - 可以看到 LLM 的推理过程（Thought -> Action -> Observation）")
    print("   - 可以分析哪个工具被调用了多少次，各步骤耗时多少")
