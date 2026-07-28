"""
final/_common.py — 所有 final 脚本共享的 boilerplate
学习者别改这里；改它会影响所有 17 个示例。
"""

import os
from dotenv import load_dotenv

# override=True 确保 .env 覆盖 shell 环境变量
load_dotenv(override=True)

# DashScope（通义千问）实际可用的模型 ID
# 默认 qwen-plus（DashScope 公网常用）；可在 .env 设 DASHSCOPE_MODEL 覆盖
DEFAULT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-plus")
DASHSCOPE_BASE_URL = os.environ["DASHSCOPE_BASE_URL"]
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]


def make_llm(model: str = DEFAULT_MODEL, temperature: float = 0.7, **kwargs):
    """统一构造 ChatOpenAI 实例，兼容 DashScope。"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
        temperature=temperature,
        **kwargs,
    )
