"""
03_human_in_the_loop.py — Human-in-the-Loop（人工审核）

知识点：
- interrupt_before：在执行节点前暂停，等待人工确认
- interrupt_after：在执行节点后暂停
- graph.update_state()：注入人工修改的状态
- Command(resume=...)：恢复执行
- 获取当前状态快照：graph.get_state()
- 实际场景：危险操作前确认、内容审核、人工修改 AI 输出
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from typing import TypedDict, Annotated, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.7,
)


# ── 工具定义（模拟危险操作）────────────────────────────────────────────────
@tool
def delete_file(filename: str) -> str:
    """删除指定文件（危险操作，需要人工确认）。"""
    # 实际场景中这里会真正删除文件
    return f"文件 '{filename}' 已删除"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（需要人工审核内容后才能发送）。"""
    return f"邮件已发送给 {to}，主题：{subject}"


@tool
def search_web(query: str) -> str:
    """搜索网络（安全操作，无需审核）。"""
    return f"搜索结果：关于'{query}'，找到若干相关信息（模拟数据）。"


# ── 1. interrupt_before：节点执行前暂停 ──────────────────────────────────────
def demo_interrupt_before():
    print("=" * 50)
    print("【interrupt_before — 危险工具调用前人工确认】")
    
    tools = [delete_file, send_email, search_web]
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)
    
    class State(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
    
    def agent(state: State) -> dict:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}
    
    builder = StateGraph(State)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    
    memory = MemorySaver()
    # interrupt_before=["tools"]：在执行 tools 节点前暂停，等待人工确认
    graph = builder.compile(checkpointer=memory, interrupt_before=["tools"])
    
    config = {"configurable": {"thread_id": "hitl_demo_1"}}
    
    # 第一步：发送请求，图会在调用工具前暂停
    print("\n用户：帮我删除 temp.txt 文件")
    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我删除 temp.txt 文件")]},
        config=config,
    )
    
    # 查看当前状态（图已暂停）
    current_state = graph.get_state(config)
    print(f"\n[系统] 图已暂停，等待人工确认")
    print(f"[系统] 下一步节点：{current_state.next}")
    
    # 查看 LLM 打算调用什么工具
    last_msg = current_state.values["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tool_call = last_msg.tool_calls[0]
        print(f"[系统] LLM 想调用工具：{tool_call['name']}({tool_call['args']})")
    
    # 模拟人工审核决策
    user_approval = input("\n[人工审核] 是否允许执行此操作？(y/n): ").strip().lower()
    
    if user_approval == "y":
        # 恢复执行
        print("[系统] 已批准，继续执行...")
        final_result = graph.invoke(None, config=config)
        print(f"执行结果：{final_result['messages'][-1].content}")
    else:
        # 注入拒绝消息，修改状态后继续
        print("[系统] 已拒绝，注入拒绝消息...")
        graph.update_state(
            config,
            {"messages": [AIMessage(content="操作已被用户取消，文件未删除。")]},
            as_node="agent",  # 以 agent 节点的身份更新状态
        )
        # 此时图的下一步变为 END
        final_state = graph.get_state(config)
        print(f"当前状态：{final_state.values['messages'][-1].content}")


# ── 2. interrupt()：在节点内部动态暂停（更灵活）─────────────────────────────
def demo_interrupt_in_node():
    print("\n" + "=" * 50)
    print("【interrupt() — 节点内动态暂停，收集人工输入】")
    
    class ReviewState(TypedDict):
        messages: Annotated[list[BaseMessage], add_messages]
        draft: str          # AI 草稿
        human_feedback: str # 人工反馈
        final_content: str  # 最终内容
    
    def generate_draft(state: ReviewState) -> dict:
        """生成初稿"""
        user_request = state["messages"][-1].content
        response = llm.invoke([
            HumanMessage(content=f"请为以下主题写一段 100 字左右的介绍：{user_request}")
        ])
        print(f"\n[AI] 已生成草稿：\n{response.content}\n")
        return {"draft": response.content}
    
    def human_review(state: ReviewState) -> dict:
        """等待人工审核（使用 interrupt）"""
        # interrupt() 会暂停图的执行，并将值传递给外部
        # 当图恢复时，interrupt() 的返回值就是恢复时传入的值
        feedback = interrupt({
            "draft": state["draft"],
            "instruction": "请审核以上草稿，输入修改意见（直接回车接受原稿）：",
        })
        return {"human_feedback": feedback or ""}
    
    def revise_or_finalize(state: ReviewState) -> dict:
        """根据反馈修改或直接采用"""
        if state["human_feedback"].strip():
            # 有反馈，让 LLM 根据反馈修改
            response = llm.invoke([
                HumanMessage(content=f"""
原始草稿：
{state['draft']}

人工反馈：
{state['human_feedback']}

请根据反馈修改草稿。
""".strip())
            ])
            final = response.content
            print(f"\n[AI] 已根据反馈修改。")
        else:
            final = state["draft"]
            print("\n[系统] 直接采用原稿。")
        
        return {"final_content": final}
    
    builder = StateGraph(ReviewState)
    builder.add_node("generate", generate_draft)
    builder.add_node("review", human_review)
    builder.add_node("revise", revise_or_finalize)
    
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "review")
    builder.add_edge("review", "revise")
    builder.add_edge("revise", END)
    
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    
    config = {"configurable": {"thread_id": "review_demo"}}
    
    # 第一次执行：会在 human_review 节点暂停
    topic = "人工智能在医疗领域的应用"
    print(f"主题：{topic}")
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=topic)]},
        config=config,
    )
    
    # 检查图是否暂停
    state = graph.get_state(config)
    if state.next:
        print(f"[系统] 图已暂停，等待节点：{state.next}")
        
        # 获取 interrupt 传出的数据
        interrupt_data = state.tasks[0].interrupts[0].value if state.tasks else {}
        if isinstance(interrupt_data, dict) and "instruction" in interrupt_data:
            feedback = input(f"\n{interrupt_data['instruction']}")
        else:
            feedback = input("\n请输入反馈（直接回车接受）：")
        
        # 用 Command 恢复执行，传入人工反馈
        final_result = graph.invoke(Command(resume=feedback), config=config)
        print(f"\n最终内容：\n{final_result['final_content']}")
    else:
        print(f"\n最终内容：\n{result['final_content']}")


# ── 3. 多步审核工作流 ─────────────────────────────────────────────────────────
def demo_approval_workflow():
    print("\n" + "=" * 50)
    print("【多步审核工作流（自动化演示，不需要交互输入）】")
    
    class WorkflowState(TypedDict):
        task: str
        plan: str
        approved: bool
        result: str
    
    def create_plan(state: WorkflowState) -> dict:
        response = llm.invoke([
            HumanMessage(content=f"为以下任务制定一个简短的执行计划（3步以内）：{state['task']}")
        ])
        print(f"\n[AI] 执行计划：\n{response.content}")
        return {"plan": response.content}
    
    def wait_for_approval(state: WorkflowState) -> dict:
        """模拟自动审核（生产中这里会是真正的人工审核）"""
        # 自动批准（演示用）
        print(f"\n[系统] 自动审核通过（演示模式）")
        return {"approved": True}
    
    def execute_task(state: WorkflowState) -> dict:
        if not state.get("approved"):
            return {"result": "任务未获批准，已取消。"}
        
        response = llm.invoke([
            HumanMessage(content=f"按照以下计划执行任务并给出结果：\n任务：{state['task']}\n计划：{state['plan']}")
        ])
        return {"result": response.content}
    
    def route_after_approval(state: WorkflowState) -> str:
        return "execute" if state.get("approved") else "cancel"
    
    def cancel_task(state: WorkflowState) -> dict:
        return {"result": "任务已取消"}
    
    builder = StateGraph(WorkflowState)
    builder.add_node("plan", create_plan)
    builder.add_node("approve", wait_for_approval)
    builder.add_node("execute", execute_task)
    builder.add_node("cancel", cancel_task)
    
    builder.add_edge(START, "plan")
    builder.add_edge("plan", "approve")
    builder.add_conditional_edges("approve", route_after_approval,
                                  {"execute": "execute", "cancel": "cancel"})
    builder.add_edge("execute", END)
    builder.add_edge("cancel", END)
    
    graph = builder.compile()
    
    result = graph.invoke({"task": "分析 2024 年 AI 行业的三大趋势"})
    print(f"\n最终结果：\n{result['result'][:200]}...")


if __name__ == "__main__":
    # demo_interrupt_before 和 demo_interrupt_in_node 需要交互输入
    # 先运行自动化演示
    demo_approval_workflow()
    
    print("\n" + "=" * 50)
    print("交互式演示（需要人工输入）：")
    print("1. interrupt_before 演示 - 危险操作前确认")
    run_interactive = input("是否运行交互式演示？(y/n): ").strip().lower()
    if run_interactive == "y":
        demo_interrupt_before()
        demo_interrupt_in_node()
    
    print("\n✅ Human-in-the-Loop 示例完成！")
    print("   核心场景：")
    print("   - 危险操作（删除、发邮件）前需人工确认")
    print("   - 内容生成后需人工审核再发布")
    print("   - 复杂决策需人工介入提供判断")
