"""
04_memory.py — 对话记忆（Memory & Message History）

知识点：
- ChatMessageHistory：手动维护消息历史
- RunnableWithMessageHistory：自动注入历史的新式 API
- ConversationBufferMemory：完整历史（旧式，了解即可）
- ConversationSummaryMemory：超长对话自动摘要
- session_id：多用户/多会话隔离
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.7,
)


# ── 1. 手动维护消息历史（最底层，理解原理）────────────────────────────────────
def demo_manual_history():
    print("=" * 50)
    print("【手动维护消息历史】")
    
    history = []  # 简单列表保存历史
    
    def chat(user_input: str) -> str:
        history.append(HumanMessage(content=user_input))
        response = llm.invoke(history)
        history.append(response)  # 保存 AI 回复
        return response.content
    
    # 多轮对话
    print(f"用户：我叫小明，今年25岁")
    print(f"AI：{chat('我叫小明，今年25岁')}")
    
    print(f"\n用户：我喜欢打篮球")
    print(f"AI：{chat('我喜欢打篮球')}")
    
    print(f"\n用户：你还记得我叫什么名字吗？")
    print(f"AI：{chat('你还记得我叫什么名字吗？')}")
    
    print(f"\n历史消息共 {len(history)} 条")


# ── 2. RunnableWithMessageHistory：自动注入历史（推荐方式）──────────────────
def demo_runnable_with_history():
    print("\n" + "=" * 50)
    print("【RunnableWithMessageHistory — 自动注入历史】")
    
    # 用 dict 模拟简单的 session 存储（生产环境可替换为 Redis/数据库）
    store: dict[str, BaseChatMessageHistory] = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    
    # Prompt 中用 MessagesPlaceholder 预留历史消息的位置
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个友好的助手，记住用户告诉你的一切。"),
        MessagesPlaceholder(variable_name="history"),  # 历史消息插入此处
        ("human", "{input}"),
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    # 用 RunnableWithMessageHistory 包装，指定 session 来源
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    # session_id = "alice" 的对话
    config_alice = {"configurable": {"session_id": "alice"}}
    
    def chat_alice(msg: str):
        resp = chain_with_history.invoke({"input": msg}, config=config_alice)
        print(f"  Alice > {msg}")
        print(f"  AI    > {resp}\n")
    
    chat_alice("你好！我叫 Alice，我是一名数据科学家。")
    chat_alice("我最近在研究大语言模型。")
    chat_alice("请问你还记得我的名字和职业吗？")
    
    # session_id = "bob" 是完全独立的对话，不知道 alice 说过什么
    config_bob = {"configurable": {"session_id": "bob"}}
    resp = chain_with_history.invoke({"input": "你好，你知道 Alice 是谁吗？"}, config=config_bob)
    print(f"  Bob   > 你好，你知道 Alice 是谁吗？")
    print(f"  AI    > {resp}")
    print("  （Bob 的会话是独立的，不知道 Alice 的存在）")


# ── 3. 带滑动窗口的历史（防止 Token 超限）─────────────────────────────────────
def demo_window_history():
    print("\n" + "=" * 50)
    print("【滑动窗口历史 — 只保留最近 N 轮】")
    
    from langchain_core.chat_history import BaseChatMessageHistory
    
    class WindowChatMessageHistory(BaseChatMessageHistory):
        """只保留最近 window_size 轮对话的历史"""
        
        def __init__(self, window_size: int = 3):
            self.window_size = window_size
            self.messages = []
        
        def add_messages(self, messages):
            self.messages.extend(messages)
            # 每轮是一条 Human + 一条 AI，共 2 条，保留 window_size 轮
            max_messages = self.window_size * 2
            if len(self.messages) > max_messages:
                self.messages = self.messages[-max_messages:]
        
        def clear(self):
            self.messages = []
    
    store = {}
    
    def get_window_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = WindowChatMessageHistory(window_size=2)
        return store[session_id]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个助手。"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain_with_window = RunnableWithMessageHistory(
        prompt | llm | StrOutputParser(),
        get_window_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    cfg = {"configurable": {"session_id": "test"}}
    
    qa_pairs = [
        "我最喜欢的颜色是蓝色",
        "我的爱好是读书",
        "我住在北京",
        "我喜欢喝咖啡",  # 此时最早的"蓝色"已被挤出窗口
        "你还记得我最喜欢什么颜色吗？",  # 应该说不记得了
    ]
    
    for q in qa_pairs:
        resp = chain_with_window.invoke({"input": q}, config=cfg)
        print(f"  用户：{q}")
        print(f"  AI：{resp[:60]}...\n" if len(resp) > 60 else f"  AI：{resp}\n")


if __name__ == "__main__":
    demo_manual_history()
    demo_runnable_with_history()
    demo_window_history()
    
    print("\n✅ Memory 示例完成！")
    print("   提示：LangSmith 会记录每次对话的完整消息历史，方便调试。")
