"""
01_tracing.py — LangSmith 追踪详解

知识点：
- 自动追踪：设置环境变量后 LangChain 所有调用自动上报
- @traceable：手动追踪普通 Python 函数
- 自定义 Run 名称、标签、元数据
- 嵌套 Trace：父子关系的 Run Tree
- RunTree：完全手动控制 Trace 的底层 API
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable, Client
from langsmith.run_trees import RunTree

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0.7,
)

# LangSmith 客户端（用于查询 Trace 数据）
ls_client = Client()


# ── 1. 自动追踪（LCEL 链自动上报，无需额外代码）───────────────────────────
def demo_auto_tracing():
    print("=" * 50)
    print("【自动追踪 — LCEL 链自动上报到 LangSmith】")
    print("（只要设置了 LANGCHAIN_TRACING_V2=true，以下代码会自动被追踪）\n")
    
    chain = (
        ChatPromptTemplate.from_template("用一句话解释：{concept}")
        | llm
        | StrOutputParser()
    )
    
    # 这次调用会自动出现在 LangSmith 的 study 项目中
    result = chain.invoke(
        {"concept": "向量嵌入"},
        config={
            "run_name": "explain_concept",          # 自定义 Run 名称
            "tags": ["demo", "auto-tracing"],        # 标签，便于过滤
            "metadata": {"version": "1.0", "env": "dev"},  # 自定义元数据
        }
    )
    print(f"结果：{result}")
    print("\n前往 LangSmith 查看这次调用：")
    print(f"  https://smith.langchain.com/projects/{os.environ.get('LANGCHAIN_PROJECT', 'default')}")


# ── 2. @traceable：追踪普通 Python 函数 ──────────────────────────────────────
def demo_traceable_decorator():
    print("\n" + "=" * 50)
    print("【@traceable — 追踪普通 Python 函数】")
    
    @traceable(
        name="数据预处理",  # 自定义显示名称
        tags=["preprocessing"],
        metadata={"step": 1},
    )
    def preprocess_text(text: str) -> str:
        """清理和标准化文本"""
        # 模拟处理步骤
        processed = text.strip().lower()
        processed = " ".join(processed.split())  # 规范化空格
        return processed
    
    @traceable(name="文本增强", tags=["augmentation"])
    def augment_text(text: str, language: str = "zh") -> str:
        """调用 LLM 增强文本"""
        response = llm.invoke([
            HumanMessage(content=f"请将以下文本改写得更专业：{text}")
        ])
        return response.content
    
    @traceable(
        name="完整文本处理流水线",
        tags=["pipeline"],
        metadata={"pipeline_version": "2.0"},
    )
    def text_pipeline(raw_text: str) -> dict:
        """完整流水线：预处理 → 增强（嵌套 Trace）"""
        preprocessed = preprocess_text(raw_text)
        enhanced = augment_text(preprocessed)
        return {
            "original": raw_text,
            "preprocessed": preprocessed,
            "enhanced": enhanced,
        }
    
    result = text_pipeline("  langchain 是一个非常强大的LLM框架   ")
    print(f"原文：{result['original']}")
    print(f"预处理：{result['preprocessed']}")
    print(f"增强：{result['enhanced'][:100]}...")
    print("\n在 LangSmith 中，text_pipeline 是父 Run，其中包含两个子 Run。")


# ── 3. 为 Trace 添加 Feedback（用户评分）─────────────────────────────────────
def demo_feedback():
    print("\n" + "=" * 50)
    print("【Feedback — 为 Trace 打分】")
    
    # 先运行一次，获取 run_id
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
    
    @traceable(name="QA_with_feedback")
    def answer_question(question: str) -> str:
        response = llm.invoke([HumanMessage(content=question)])
        return response.content
    
    # 捕获 run_id 的方式：使用 langsmith_extra
    run_id = None
    
    @traceable(name="tracked_answer")
    def tracked_answer(question: str) -> tuple[str, str]:
        rt = get_current_run_tree()
        answer = llm.invoke([HumanMessage(content=question)]).content
        return answer, str(rt.id) if rt else ""
    
    question = "LangSmith 能做什么？"
    answer, run_id = tracked_answer(question)
    
    print(f"问：{question}")
    print(f"答：{answer[:100]}...")
    
    if run_id:
        # 模拟用户对这次回答打分
        try:
            ls_client.create_feedback(
                run_id=run_id,
                key="user_rating",
                score=0.9,            # 0-1 之间的分数
                comment="回答准确且简洁",
            )
            print(f"\n已为 Run {run_id[:8]}... 提交评分：0.9")
        except Exception as e:
            print(f"\n提交评分时出错（可能需要等待 Trace 同步）：{e}")
    
    print("\n在 LangSmith UI 中可以看到每次运行的用户评分，用于质量监控。")


# ── 4. RunTree：手动完全控制 Trace ───────────────────────────────────────────
def demo_run_tree():
    print("\n" + "=" * 50)
    print("【RunTree — 手动控制 Trace（了解底层原理）】")
    
    # 创建父 Run
    root_run = RunTree(
        name="手动追踪示例",
        run_type="chain",
        inputs={"query": "什么是 RAG？"},
        tags=["manual", "demo"],
    )
    root_run.post()  # 上报到 LangSmith
    
    try:
        # 模拟步骤1：检索
        retrieval_run = root_run.create_child(
            name="向量检索",
            run_type="retriever",
            inputs={"query": "什么是 RAG？"},
        )
        retrieval_run.post()
        
        time.sleep(0.1)  # 模拟检索耗时
        docs = ["RAG 是检索增强生成技术...", "RAG 结合了检索和生成模型..."]
        retrieval_run.end(outputs={"documents": docs})
        retrieval_run.patch()  # 更新到 LangSmith
        
        # 模拟步骤2：生成
        generation_run = root_run.create_child(
            name="LLM 生成",
            run_type="llm",
            inputs={"prompt": f"基于以下文档回答：{docs}"},
        )
        generation_run.post()
        
        response = llm.invoke([HumanMessage(content="用一句话解释 RAG")])
        generation_run.end(outputs={"text": response.content})
        generation_run.patch()
        
        # 结束父 Run
        root_run.end(outputs={"answer": response.content})
        root_run.patch()
        
        print(f"手动 Trace 已上报，Run ID：{root_run.id}")
        print(f"回答：{response.content}")
        
    except Exception as e:
        root_run.end(error=str(e))
        root_run.patch()
        print(f"发生错误：{e}")


if __name__ == "__main__":
    demo_auto_tracing()
    demo_traceable_decorator()
    demo_feedback()
    demo_run_tree()
    
    print("\n✅ LangSmith 追踪示例完成！")
    print("   前往 https://smith.langchain.com 查看所有 Trace：")
    print(f"   项目：{os.environ.get('LANGCHAIN_PROJECT', 'default')}")
    print("\n   追踪层级：")
    print("   LangChain 链 → 自动追踪所有步骤")
    print("   @traceable → 追踪任意 Python 函数")
    print("   RunTree → 完全手动控制，适合非 LangChain 代码")
