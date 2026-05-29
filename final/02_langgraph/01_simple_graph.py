"""
01_simple_graph.py — LangGraph 入门：StateGraph 基础

知识点：
- StateGraph：有状态的执行图
- TypedDict 定义共享 State
- add_node / add_edge / set_entry_point / set_finish_point
- compile() 编译图
- invoke() vs stream() 两种执行方式
- 可视化图结构
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.7,
)


# ── 1. 最简 StateGraph：单节点聊天机器人 ─────────────────────────────────────
def demo_simple_chatbot():
    print("=" * 50)
    print("【最简 StateGraph — 单节点聊天机器人】")
    
    # State 定义：图中所有节点共享的状态
    # Annotated[list, add_messages] 表示 messages 字段使用 add_messages reducer
    # add_messages 会追加新消息而非覆盖，这是对话历史的标准做法
    class ChatState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    # 节点函数：接收当前 State，返回要更新的字段
    def chatbot_node(state: ChatState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}  # add_messages 会将此消息追加到列表
    
    # 构建图
    graph_builder = StateGraph(ChatState)
    graph_builder.add_node("chatbot", chatbot_node)
    graph_builder.add_edge(START, "chatbot")   # 入口
    graph_builder.add_edge("chatbot", END)      # 出口
    
    graph = graph_builder.compile()
    
    # invoke：一次性返回最终状态
    result = graph.invoke({
        "messages": [HumanMessage(content="你好！用一句话介绍 LangGraph。")]
    })
    print(f"回答：{result['messages'][-1].content}")
    
    # stream：逐步返回每个节点的输出
    print("\n流式执行（可以看到每个节点的输出）：")
    for step in graph.stream({"messages": [HumanMessage(content="LangGraph 和 LangChain 有什么区别？")]}):
        for node_name, node_output in step.items():
            print(f"  节点 [{node_name}] 输出：{node_output['messages'][-1].content[:60]}...")


# ── 2. 多节点图：带预处理和后处理 ────────────────────────────────────────────
def demo_multi_node_graph():
    print("\n" + "=" * 50)
    print("【多节点图 — 预处理 → LLM → 后处理】")
    
    class PipelineState(TypedDict):
        input: str           # 用户原始输入
        processed_input: str # 预处理后的输入
        llm_output: str      # LLM 原始输出
        final_output: str    # 最终输出
    
    def preprocess(state: PipelineState) -> dict:
        """预处理：清理输入，加前缀"""
        cleaned = state["input"].strip()
        processed = f"请用简洁的中文回答：{cleaned}"
        print(f"  [预处理] '{cleaned}' -> '{processed}'")
        return {"processed_input": processed}
    
    def call_llm(state: PipelineState) -> dict:
        """调用 LLM"""
        response = llm.invoke([HumanMessage(content=state["processed_input"])])
        print(f"  [LLM] 生成回答，共 {len(response.content)} 字")
        return {"llm_output": response.content}
    
    def postprocess(state: PipelineState) -> dict:
        """后处理：添加分隔线"""
        formatted = f"{'─' * 30}\n{state['llm_output']}\n{'─' * 30}"
        return {"final_output": formatted}
    
    # 构建三节点线性图
    builder = StateGraph(PipelineState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("call_llm", call_llm)
    builder.add_node("postprocess", postprocess)
    
    builder.add_edge(START, "preprocess")
    builder.add_edge("preprocess", "call_llm")
    builder.add_edge("call_llm", "postprocess")
    builder.add_edge("postprocess", END)
    
    graph = builder.compile()
    
    result = graph.invoke({"input": "什么是向量数据库？"})
    print(f"\n最终输出：\n{result['final_output']}")


# ── 3. 带计数器的循环图（非 LLM 示例，理解循环原理）────────────────────────
def demo_loop_graph():
    print("\n" + "=" * 50)
    print("【循环图 — 理解条件边与循环】")
    
    class CounterState(TypedDict):
        count: int
        messages: list[str]
    
    def increment(state: CounterState) -> dict:
        new_count = state["count"] + 1
        new_msg = f"第 {new_count} 次执行"
        print(f"  {new_msg}")
        return {
            "count": new_count,
            "messages": state["messages"] + [new_msg],
        }
    
    def should_continue(state: CounterState) -> str:
        """条件函数：返回下一个节点名称"""
        if state["count"] < 3:
            return "continue"   # 继续循环
        else:
            return "stop"       # 结束
    
    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    
    builder.add_edge(START, "increment")
    builder.add_conditional_edges(
        "increment",           # 来源节点
        should_continue,       # 条件函数，返回字符串
        {
            "continue": "increment",  # 返回 "continue" 时跳回 increment
            "stop": END,              # 返回 "stop" 时结束
        }
    )
    
    graph = builder.compile()
    
    result = graph.invoke({"count": 0, "messages": []})
    print(f"\n执行了 {result['count']} 次，消息：{result['messages']}")


# ── 4. 打印图的 ASCII 结构 ────────────────────────────────────────────────────
def demo_graph_visualization():
    print("\n" + "=" * 50)
    print("【图结构可视化】")
    
    class SimpleState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    def node_a(state): return {}
    def node_b(state): return {}
    def node_c(state): return {}
    
    builder = StateGraph(SimpleState)
    builder.add_node("A", node_a)
    builder.add_node("B", node_b)
    builder.add_node("C", node_c)
    builder.add_edge(START, "A")
    builder.add_edge("A", "B")
    builder.add_edge("A", "C")  # A 同时连接 B 和 C（并行）
    builder.add_edge("B", END)
    builder.add_edge("C", END)
    
    graph = builder.compile()
    
    # 打印 ASCII 图（LangGraph 内置功能）
    print(graph.get_graph().draw_ascii())


if __name__ == "__main__":
    demo_simple_chatbot()
    demo_multi_node_graph()
    demo_loop_graph()
    demo_graph_visualization()
    
    print("\n✅ LangGraph 基础示例完成！")
    print("   核心概念总结：")
    print("   - State：所有节点共享的数据字典")
    print("   - Node：纯函数，接收 State 返回更新字典")
    print("   - Edge：节点间的连接（普通边或条件边）")
    print("   - compile()：将图编译为可执行的 Runnable")
