"""
03_chains.py — LCEL（LangChain Expression Language）链路

知识点：
- LCEL 管道操作符 |
- StrOutputParser / JsonOutputParser
- RunnableParallel：并行执行多条链
- RunnablePassthrough：透传输入
- RunnableLambda：将普通函数包装成 Runnable
- 链的调试：verbose 和 LangSmith Trace
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.7,
)


# ── 1. 基础链：prompt | llm | parser ──────────────────────────────────────────
def demo_basic_chain():
    print("=" * 50)
    print("【基础 LCEL 链】")
    
    prompt = ChatPromptTemplate.from_template("用一句话解释：{topic}")
    
    # | 操作符将各步骤串联
    chain = prompt | llm | StrOutputParser()
    
    result = chain.invoke({"topic": "量子纠缠"})
    print(f"量子纠缠：{result}")
    
    # 也可以直接传字符串（当模板只有一个变量时）
    result2 = chain.invoke({"topic": "黑洞"})
    print(f"黑洞：{result2}")


# ── 2. JSON 输出解析 ───────────────────────────────────────────────────────────
def demo_json_output():
    print("\n" + "=" * 50)
    print("【JsonOutputParser — 结构化输出】")
    
    # 定义输出结构
    class MovieReview(BaseModel):
        title: str = Field(description="电影标题")
        score: int = Field(description="评分 1-10")
        summary: str = Field(description="一句话评价")
    
    parser = JsonOutputParser(pydantic_object=MovieReview)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是专业影评人。"),
        ("human", "请评价电影《{movie}》，按以下格式输出：\n{format_instructions}"),
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser
    
    result = chain.invoke({"movie": "星际穿越"})
    print(f"电影评价：{result}")
    print(f"类型：{type(result)}")  # dict


# ── 3. RunnableParallel：并行执行多条链 ───────────────────────────────────────
def demo_parallel():
    print("\n" + "=" * 50)
    print("【RunnableParallel — 并行链】")
    
    # 两条独立的链并行执行，共享同一输入
    pros_chain = (
        ChatPromptTemplate.from_template("列举 {thing} 的3个优点，每点一行")
        | llm
        | StrOutputParser()
    )
    
    cons_chain = (
        ChatPromptTemplate.from_template("列举 {thing} 的3个缺点，每点一行")
        | llm
        | StrOutputParser()
    )
    
    parallel_chain = RunnableParallel(
        pros=pros_chain,
        cons=cons_chain,
    )
    
    result = parallel_chain.invoke({"thing": "远程办公"})
    print(f"优点：\n{result['pros']}")
    print(f"\n缺点：\n{result['cons']}")


# ── 4. RunnablePassthrough：透传与数据增强 ────────────────────────────────────
def demo_passthrough():
    print("\n" + "=" * 50)
    print("【RunnablePassthrough — 透传输入】")
    
    # 把原始输入和 LLM 输出一起传给下一步
    prompt = ChatPromptTemplate.from_template("将这段英文翻译成中文：{text}")
    
    chain = RunnableParallel(
        original=RunnablePassthrough(),       # 原样透传输入
        translated=prompt | llm | StrOutputParser(),  # 翻译结果
    )
    
    result = chain.invoke({"text": "LangChain makes it easy to build LLM applications."})
    print(f"原文：{result['original']}")
    print(f"译文：{result['translated']}")


# ── 5. RunnableLambda：自定义函数步骤 ────────────────────────────────────────
def demo_lambda():
    print("\n" + "=" * 50)
    print("【RunnableLambda — 自定义函数】")
    
    def word_count(text: str) -> str:
        count = len(text.split())
        return f"（共 {count} 个词）\n{text}"
    
    prompt = ChatPromptTemplate.from_template("写一首关于{topic}的短诗（4行）")
    
    chain = (
        prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(word_count)  # 在输出后追加词数统计
    )
    
    result = chain.invoke({"topic": "秋天"})
    print(result)


# ── 6. 链的组合与复用 ─────────────────────────────────────────────────────────
def demo_chain_composition():
    print("\n" + "=" * 50)
    print("【链的组合复用】")
    
    # 子链1：生成关键词
    keywords_chain = (
        ChatPromptTemplate.from_template("从以下文本提取3个关键词，逗号分隔：{text}")
        | llm
        | StrOutputParser()
    )
    
    # 子链2：根据关键词生成摘要（接受上一步输出）
    summary_chain = (
        ChatPromptTemplate.from_template("用这些关键词写一个两句话的摘要：{keywords}")
        | llm
        | StrOutputParser()
    )
    
    # 串联两条子链
    full_chain = (
        {"keywords": keywords_chain}  # 等价于 RunnableParallel(keywords=...)
        | RunnablePassthrough.assign(keywords=lambda x: x["keywords"])
        | summary_chain
    )
    
    text = "深度学习通过多层神经网络学习数据的层次特征，在图像识别、自然语言处理等领域取得了突破性进展。"
    result = full_chain.invoke({"text": text})
    print(f"摘要：{result}")


if __name__ == "__main__":
    demo_basic_chain()
    demo_json_output()
    demo_parallel()
    demo_passthrough()
    demo_lambda()
    demo_chain_composition()
    
    print("\n✅ LCEL 示例完成！在 LangSmith 可以看到每条链的完整调用树。")
