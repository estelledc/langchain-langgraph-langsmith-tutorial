"""
agent.py — 研究助手 Agent 入口

使用方法：
    python agent.py
    python agent.py --question "LangGraph 的核心特性有哪些？"
    python agent.py --stream  # 流式输出每个步骤
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import sys
import os
from langchain_core.messages import HumanMessage

from graph import build_research_graph, ResearchState


def run_research(question: str, stream: bool = False, thread_id: str = "default") -> str:
    """
    运行研究助手
    
    Args:
        question: 研究问题
        stream: 是否流式输出每个节点的执行过程
        thread_id: 对话线程 ID（相同 ID 可以继续上次的对话）
    
    Returns:
        最终研究报告
    """
    graph = build_research_graph(with_memory=True)
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "research_plan": "",
        "collected_info": "",
        "final_report": "",
        "iteration_count": 0,
    }
    
    print(f"\n{'='*60}")
    print(f"研究问题：{question}")
    print(f"{'='*60}")
    
    if stream:
        # 流式模式：逐步显示每个节点的执行
        final_state = None
        for step in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in step.items():
                print(f"\n[节点: {node_name}]")
                if "messages" in node_output:
                    last_msg = node_output["messages"][-1]
                    if hasattr(last_msg, "content") and last_msg.content:
                        preview = last_msg.content[:150]
                        print(f"  输出预览：{preview}{'...' if len(last_msg.content) > 150 else ''}")
            final_state = step
        
        # 获取最终状态
        final = graph.get_state(config)
        report = final.values.get("final_report", "未生成报告")
    else:
        # 一次性执行
        result = graph.invoke(initial_state, config=config)
        report = result.get("final_report", "未生成报告")
    
    return report


def interactive_mode():
    """交互式多轮对话模式"""
    print("=" * 60)
    print("研究助手 Agent（交互模式）")
    print("输入研究问题，输入 'quit' 退出")
    print("=" * 60)
    
    graph = build_research_graph(with_memory=True)
    thread_id = "interactive_session"
    config = {"configurable": {"thread_id": thread_id}}
    
    while True:
        question = input("\n请输入研究问题 > ").strip()
        
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        
        if not question:
            continue
        
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "research_plan": "",
            "collected_info": "",
            "final_report": "",
            "iteration_count": 0,
        }
        
        try:
            result = graph.invoke(initial_state, config=config)
            report = result.get("final_report", "未生成报告")
            
            print(f"\n{'─'*60}")
            print("研究报告：")
            print(f"{'─'*60}")
            print(report)
            print(f"{'─'*60}")
            
        except Exception as e:
            print(f"错误：{e}")


def main():
    parser = argparse.ArgumentParser(description="研究助手 Agent")
    parser.add_argument("--question", "-q", type=str, help="研究问题")
    parser.add_argument("--stream", "-s", action="store_true", help="流式输出")
    parser.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    parser.add_argument("--thread", "-t", type=str, default="default", help="对话线程ID")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
        return
    
    if args.question:
        report = run_research(args.question, stream=args.stream, thread_id=args.thread)
        
        print(f"\n{'='*60}")
        print("最终研究报告：")
        print(f"{'='*60}")
        print(report)
        print(f"\n✅ 完成！前往 LangSmith 查看完整执行过程：")
        print(f"   https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT', 'study')}")
        return
    
    # 默认：运行演示问题
    demo_questions = [
        "LangChain、LangGraph 和 LangSmith 分别是什么？它们如何协同工作？",
        "RAG 技术的工作原理是什么？有哪些主流的向量数据库？",
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#'*60}")
        print(f"演示问题 {i}/{len(demo_questions)}")
        print(f"{'#'*60}")
        
        report = run_research(question, stream=False, thread_id=f"demo_{i}")
        
        print(f"\n最终报告：")
        print(report)
    
    print(f"\n\n✅ 演示完成！前往 LangSmith 查看所有 Trace：")
    print(f"   https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT', 'study')}")


if __name__ == "__main__":
    main()
