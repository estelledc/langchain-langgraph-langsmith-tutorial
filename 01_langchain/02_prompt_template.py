"""
02_prompt_template.py — Prompt 模板

知识点：
- PromptTemplate（字符串模板）
- ChatPromptTemplate（对话模板，支持 role）
- FewShotChatMessagePromptTemplate（少样本提示）
- 模板变量插值与格式化
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.7,
)

# ── 1. PromptTemplate：纯文本模板 ─────────────────────────────────────────────
def demo_prompt_template():
    print("=" * 50)
    print("【PromptTemplate — 字符串模板】")
    
    # 用 {变量名} 占位
    template = PromptTemplate.from_template(
        "请用{language}写一个计算{n}的阶乘的函数，只返回代码，不要解释。"
    )
    
    # format() 将模板渲染为字符串
    prompt_str = template.format(language="Python", n=5)
    print(f"渲染后的 Prompt:\n{prompt_str}\n")
    
    # 也可以直接 invoke LLM
    chain = template | llm | StrOutputParser()
    result = chain.invoke({"language": "Python", "n": 5})
    print(f"LLM 回答:\n{result}")


# ── 2. ChatPromptTemplate：多角色对话模板 ────────────────────────────────────
def demo_chat_prompt_template():
    print("\n" + "=" * 50)
    print("【ChatPromptTemplate — 对话模板】")
    
    # 支持 system / human / ai 三种角色
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{domain}领域专家，回答要精准简洁。"),
        ("human", "请解释一下：{concept}"),
    ])
    
    # format_messages() 渲染为消息列表
    messages = chat_prompt.format_messages(domain="机器学习", concept="梯度下降")
    for msg in messages:
        print(f"[{msg.__class__.__name__}] {msg.content}")
    
    # 组成链并调用
    chain = chat_prompt | llm | StrOutputParser()
    result = chain.invoke({"domain": "机器学习", "concept": "过拟合"})
    print(f"\n回答: {result}")


# ── 3. FewShotChatMessagePromptTemplate：少样本提示 ──────────────────────────
def demo_few_shot():
    print("\n" + "=" * 50)
    print("【FewShotChatMessagePromptTemplate — 少样本提示】")
    
    # 准备示例对
    examples = [
        {"input": "2+2", "output": "4"},
        {"input": "3*5", "output": "15"},
        {"input": "10/2", "output": "5"},
    ]
    
    # 定义单个示例的格式
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}"),
    ])
    
    # 构建 Few-Shot 模板
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )
    
    # 组合完整的对话模板
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个计算器，只输出数字结果。"),
        few_shot_prompt,
        ("human", "{input}"),
    ])
    
    chain = final_prompt | llm | StrOutputParser()
    result = chain.invoke({"input": "7*8"})
    print(f"7*8 = {result}")
    
    result2 = chain.invoke({"input": "100-37"})
    print(f"100-37 = {result2}")


# ── 4. 模板的部分填充（partial）────────────────────────────────────────────────
def demo_partial():
    print("\n" + "=" * 50)
    print("【Partial — 预先填充部分变量】")
    
    template = ChatPromptTemplate.from_messages([
        ("system", "你是{language}语言专家。"),
        ("human", "请解释 {concept} 概念。"),
    ])
    
    # 预先固定 language，留 concept 动态填充
    python_expert = template.partial(language="Python")
    
    chain = python_expert | llm | StrOutputParser()
    
    for concept in ["装饰器", "生成器", "上下文管理器"]:
        result = chain.invoke({"concept": concept})
        print(f"\n[{concept}] {result[:80]}...")


if __name__ == "__main__":
    demo_prompt_template()
    demo_chat_prompt_template()
    demo_few_shot()
    demo_partial()
    
    print("\n✅ 完成！前往 LangSmith 查看各次调用的 Prompt 详情。")
