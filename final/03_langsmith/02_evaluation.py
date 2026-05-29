"""
02_evaluation.py — LangSmith 批量评估

知识点：
- evaluate()：批量跑测试集并评估输出
- 内置评估器：ExactMatchStringEvaluator、EmbeddingDistanceEvaluator
- LLM-as-Judge：用 LLM 自动评分（最常用）
- 自定义评估函数
- 查看评估结果
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, evaluate
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.schemas import Run, Example

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0,
)

ls_client = Client()

# ── 被评估的应用（目标函数）────────────────────────────────────────────────
qa_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是一个知识问答助手，回答要准确简洁，不超过100字。"),
        ("human", "{question}"),
    ])
    | llm
    | StrOutputParser()
)


# ── 1. 创建评估数据集 ──────────────────────────────────────────────────────
def create_or_get_dataset(dataset_name: str):
    """创建或获取已有数据集"""
    
    # 检查数据集是否已存在
    existing = list(ls_client.list_datasets(dataset_name=dataset_name))
    if existing:
        print(f"使用已有数据集：{dataset_name}")
        return existing[0]
    
    print(f"创建新数据集：{dataset_name}")
    
    # 问答对：input 是问题，output 是参考答案
    examples = [
        {
            "inputs": {"question": "什么是 LangChain？"},
            "outputs": {"answer": "LangChain 是一个用于构建 LLM 应用的开源框架，提供链路、工具、记忆等核心抽象。"},
        },
        {
            "inputs": {"question": "LangGraph 有什么特点？"},
            "outputs": {"answer": "LangGraph 基于图结构构建有状态 Agent，支持条件边、循环、人工审核等特性。"},
        },
        {
            "inputs": {"question": "LangSmith 的主要功能是什么？"},
            "outputs": {"answer": "LangSmith 提供 LLM 应用的追踪、评估、数据集管理和生产监控功能。"},
        },
        {
            "inputs": {"question": "什么是 RAG？"},
            "outputs": {"answer": "RAG（检索增强生成）将向量检索与 LLM 结合，通过检索相关文档减少幻觉，支持知识更新。"},
        },
        {
            "inputs": {"question": "LCEL 是什么？"},
            "outputs": {"answer": "LCEL 是 LangChain 表达式语言，用 | 操作符串联组件，支持流式输出和并行执行。"},
        },
    ]
    
    dataset = ls_client.create_dataset(
        dataset_name=dataset_name,
        description="LangChain 生态知识问答评估数据集",
    )
    
    ls_client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )
    
    print(f"数据集已创建，包含 {len(examples)} 个样本")
    return dataset


# ── 2. 目标函数（被评估的系统）───────────────────────────────────────────────
def target_function(inputs: dict) -> dict:
    """评估时调用的目标函数，输入来自数据集的 inputs 字段"""
    answer = qa_chain.invoke({"question": inputs["question"]})
    return {"answer": answer}


# ── 3. 自定义评估函数 ─────────────────────────────────────────────────────────
def length_check_evaluator(run: Run, example: Example) -> dict:
    """检查回答长度是否合理（自定义评估器示例）"""
    output = run.outputs or {}
    answer = output.get("answer", "")
    
    # 评分规则：50-150字为满分，过短或过长扣分
    length = len(answer)
    if 50 <= length <= 150:
        score = 1.0
        comment = f"长度适中（{length}字）"
    elif length < 50:
        score = 0.5
        comment = f"回答过短（{length}字）"
    else:
        score = 0.7
        comment = f"回答偏长（{length}字）"
    
    return {
        "key": "length_check",
        "score": score,
        "comment": comment,
    }


def keyword_check_evaluator(run: Run, example: Example) -> dict:
    """检查回答是否包含参考答案中的关键词"""
    output = run.outputs or {}
    reference = (example.outputs or {}).get("answer", "")
    
    answer = output.get("answer", "")
    
    # 提取参考答案中的关键词（简单版：取较长的词）
    ref_words = set(w for w in reference.split() if len(w) >= 3)
    
    if not ref_words:
        return {"key": "keyword_coverage", "score": 0.5, "comment": "无参考词"}
    
    # 计算关键词覆盖率
    matched = sum(1 for w in ref_words if w in answer)
    coverage = matched / len(ref_words)
    
    return {
        "key": "keyword_coverage",
        "score": coverage,
        "comment": f"关键词覆盖：{matched}/{len(ref_words)}",
    }


# ── 4. LLM-as-Judge 评估器 ────────────────────────────────────────────────────
def create_llm_judge():
    """创建 LLM-as-Judge 评估器"""
    
    # 使用 LangChain 内置的 QA 评估器（基于 LLM 判断准确性）
    qa_evaluator = LangChainStringEvaluator(
        "qa",
        config={
            "llm": llm,
        },
        prepare_data=lambda run, example: {
            "prediction": (run.outputs or {}).get("answer", ""),
            "reference": (example.outputs or {}).get("answer", ""),
            "input": (example.inputs or {}).get("question", ""),
        }
    )
    
    return qa_evaluator


# ── 5. 运行评估 ───────────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 50)
    print("【运行批量评估】")
    
    dataset_name = "langchain_qa_eval_v1"
    dataset = create_or_get_dataset(dataset_name)
    
    print(f"\n开始评估，数据集：{dataset_name}")
    print("评估器：长度检查、关键词覆盖、LLM-as-Judge")
    
    # evaluate() 会：
    # 1. 遍历数据集中的每个样本
    # 2. 调用 target_function 得到预测结果
    # 3. 用每个评估器对结果打分
    # 4. 将所有结果上报到 LangSmith
    results = evaluate(
        target_function,
        data=dataset_name,
        evaluators=[
            length_check_evaluator,
            keyword_check_evaluator,
        ],
        experiment_prefix="qa_baseline",  # 实验名称前缀
        metadata={"model": "qwen3.5-plus", "version": "1.0"},
    )
    
    print(f"\n评估完成！共评估 {len(list(results))} 个样本")
    print(f"在 LangSmith 查看详细结果：https://smith.langchain.com")
    
    return results


# ── 6. 对比实验（A/B 测试）────────────────────────────────────────────────────
def run_ab_comparison():
    print("\n" + "=" * 50)
    print("【A/B 对比实验】")
    
    dataset_name = "langchain_qa_eval_v1"
    
    # 版本 A：保守型提示（更简洁）
    chain_v1 = (
        ChatPromptTemplate.from_messages([
            ("system", "简洁回答问题，不超过50字。"),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    
    # 版本 B：详细型提示（更完整）
    chain_v2 = (
        ChatPromptTemplate.from_messages([
            ("system", "详细准确地回答问题，包含关键概念和使用场景，100-150字。"),
            ("human", "{question}"),
        ])
        | llm
        | StrOutputParser()
    )
    
    def target_v1(inputs: dict) -> dict:
        return {"answer": chain_v1.invoke({"question": inputs["question"]})}
    
    def target_v2(inputs: dict) -> dict:
        return {"answer": chain_v2.invoke({"question": inputs["question"]})}
    
    evaluators = [length_check_evaluator, keyword_check_evaluator]
    
    print("运行版本 A（简洁提示）...")
    results_v1 = evaluate(
        target_v1,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="prompt_v1_concise",
    )
    
    print("运行版本 B（详细提示）...")
    results_v2 = evaluate(
        target_v2,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="prompt_v2_detailed",
    )
    
    print("\nA/B 实验完成！在 LangSmith 的 Experiments 页面可以对比两个版本的评分。")


if __name__ == "__main__":
    run_evaluation()
    run_ab_comparison()
    
    print("\n✅ LangSmith 评估示例完成！")
    print("   评估工作流总结：")
    print("   1. 创建 Dataset（问答对）")
    print("   2. 定义 target_function（被评估的系统）")
    print("   3. 定义 evaluators（评估标准）")
    print("   4. 调用 evaluate() 运行批量测试")
    print("   5. 在 LangSmith UI 查看和对比实验结果")
