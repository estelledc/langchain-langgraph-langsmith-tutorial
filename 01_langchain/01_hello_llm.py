"""
01_hello_llm.py — LangChain 入门：直接调用 LLM

知识点：
- ChatOpenAI 对接 DashScope（OpenAI 兼容接口）
- HumanMessage / AIMessage / SystemMessage
- invoke() / stream() 两种调用方式
- LangSmith 自动追踪（.env 中 LANGCHAIN_TRACING_V2=true 即可）
"""

from dotenv import load_dotenv
load_dotenv(override=True)  # override=True 确保 .env 覆盖 shell 环境变量

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── 1. 创建 LLM 实例 ─────────────────────────────────────────────────────────
#   DashScope 兼容 OpenAI 格式，只需修改 base_url 和 api_key
llm = ChatOpenAI(
    model="qwen3.5-plus",                          # 通义千问 qwen3.5-plus 模型
    base_url=os.environ["DASHSCOPE_BASE_URL"],  # 从 .env 读取
    api_key=os.environ["DASHSCOPE_API_KEY"],    # 从 .env 读取
    temperature=0.7,
)

def demo_invoke():
    """最简单的一次性调用"""
    print("=" * 50)
    print("【invoke 调用】")
    
    messages = [
        SystemMessage(content="你是一个简洁的 AI 助手，回答不超过 50 字。"),
        HumanMessage(content="什么是 LangChain？"),
    ]
    
    response: AIMessage = llm.invoke(messages)
    
    # response.content 是文本内容
    print(f"回答：{response.content}")
    # response.response_metadata 包含 token 用量等元数据
    print(f"Token 用量：{response.response_metadata.get('token_usage', {})}")


def demo_stream():
    """流式输出——适合长文本场景"""
    print("\n" + "=" * 50)
    print("【stream 流式输出】")
    
    messages = [
        HumanMessage(content="用三句话介绍一下 Python 语言的特点。"),
    ]
    
    print("回答：", end="", flush=True)
    for chunk in llm.stream(messages):
        # chunk 是 AIMessageChunk，content 是本次片段文本
        print(chunk.content, end="", flush=True)
    print()  # 换行


def demo_batch():
    """批量调用——一次发送多条消息"""
    print("\n" + "=" * 50)
    print("【batch 批量调用】")
    
    questions = [
        [HumanMessage(content="1+1=?")],
        [HumanMessage(content="太阳系有几颗行星？")],
        [HumanMessage(content="Python 之父是谁？")],
    ]
    
    responses = llm.batch(questions)
    for i, resp in enumerate(responses):
        print(f"Q{i+1}: {questions[i][0].content}")
        print(f"A{i+1}: {resp.content}\n")


if __name__ == "__main__":
    demo_invoke()
    demo_stream()
    demo_batch()
    
    print("\n✅ 运行完毕！请前往 https://smith.langchain.com 查看 'study' 项目的 Trace 记录。")
