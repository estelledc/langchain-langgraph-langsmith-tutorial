"""
eval.py — 研究助手 Agent 的 LangSmith 评估

评估维度：
1. 报告完整性：是否涵盖了问题的主要方面
2. 报告准确性：内容是否准确（LLM-as-Judge）
3. 报告长度：是否在合理范围内
4. 工具使用率：是否合理调用了工具
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langsmith import Client, evaluate
from langsmith.schemas import Run, Example

from agent import run_research

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0,
)

ls_client = Client()
DATASET_NAME = "research_agent_eval_v1"


# ── 创建评估数据集 ─────────────────────────────────────────────────────────
def create_eval_dataset():
    """创建研究助手的评估数据集"""
    
    existing = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"使用已有数据集：{DATASET_NAME}")
        return existing[0]
    
    examples = [
        {
            "inputs": {
                "question": "LangChain 是什么？它有哪些核心组件？",
            },
            "outputs": {
                "expected_keywords": ["框架", "LLM", "链", "工具", "记忆"],
                "min_length": 100,
                "max_length": 600,
            },
        },
        {
            "inputs": {
                "question": "什么是 RAG？请介绍其工作原理和常用向量数据库。",
            },
            "outputs": {
                "expected_keywords": ["检索", "生成", "向量", "嵌入", "幻觉"],
                "min_length": 150,
                "max_length": 600,
            },
        },
        {
            "inputs": {
                "question": "LangGraph 和传统 LangChain Agent 有什么区别？",
            },
            "outputs": {
                "expected_keywords": ["状态", "图", "循环", "条件", "节点"],
                "min_length": 100,
                "max_length": 600,
            },
        },
    ]
    
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="研究助手 Agent 评估数据集",
    )
    
    ls_client.create_examples(
        inputs=[e["inputs"] for e in examples],
        outputs=[e["outputs"] for e in examples],
        dataset_id=dataset.id,
    )
    
    print(f"创建数据集：{DATASET_NAME}，{len(examples)} 个样本")
    return dataset


# ── 目标函数 ───────────────────────────────────────────────────────────────
def research_agent_target(inputs: dict) -> dict:
    """被评估的研究助手"""
    question = inputs["question"]
    report = run_research(question, stream=False, thread_id=f"eval_{hash(question) % 10000}")
    return {"report": report}


# ── 评估函数 ───────────────────────────────────────────────────────────────
def evaluate_length(run: Run, example: Example) -> dict:
    """评估报告长度是否合理"""
    report = (run.outputs or {}).get("report", "")
    expected = example.outputs or {}
    
    min_len = expected.get("min_length", 100)
    max_len = expected.get("max_length", 600)
    actual_len = len(report)
    
    if min_len <= actual_len <= max_len:
        score = 1.0
        comment = f"长度合适（{actual_len}字，期望 {min_len}-{max_len}字）"
    elif actual_len < min_len:
        score = actual_len / min_len
        comment = f"报告过短（{actual_len}字，最少 {min_len}字）"
    else:
        # 超出上限但内容充实，给部分分
        score = 0.7
        comment = f"报告偏长（{actual_len}字，建议不超过 {max_len}字）"
    
    return {"key": "report_length", "score": score, "comment": comment}


def evaluate_keywords(run: Run, example: Example) -> dict:
    """评估报告是否包含预期关键词"""
    report = (run.outputs or {}).get("report", "").lower()
    expected_keywords = (example.outputs or {}).get("expected_keywords", [])
    
    if not expected_keywords:
        return {"key": "keyword_coverage", "score": 0.5, "comment": "无预期关键词"}
    
    matched = [kw for kw in expected_keywords if kw in report]
    coverage = len(matched) / len(expected_keywords)
    
    return {
        "key": "keyword_coverage",
        "score": coverage,
        "comment": f"关键词覆盖 {len(matched)}/{len(expected_keywords)}：{matched}",
    }


def evaluate_quality_llm(run: Run, example: Example) -> dict:
    """LLM-as-Judge：评估报告质量"""
    report = (run.outputs or {}).get("report", "")
    question = (example.inputs or {}).get("question", "")
    
    if not report:
        return {"key": "llm_quality", "score": 0.0, "comment": "报告为空"}
    
    judge_prompt = f"""请评估以下研究报告的质量，给出 0-1 之间的分数。

评分标准：
- 1.0：内容准确、结构清晰、覆盖全面、有实质价值
- 0.7：内容基本准确，结构合理，但有遗漏或不够深入
- 0.4：内容部分准确，结构散乱，价值有限
- 0.1：内容不准确或几乎没有实质内容

研究问题：{question}

报告内容：
{report[:600]}

请只输出一个数字（0到1之间），不要其他内容。"""
    
    try:
        response = llm.invoke([HumanMessage(content=judge_prompt)])
        score_str = response.content.strip()
        score = float(score_str)
        score = max(0.0, min(1.0, score))  # 确保在 0-1 范围内
    except (ValueError, Exception):
        score = 0.5  # 解析失败时给中等分
    
    return {
        "key": "llm_quality_judge",
        "score": score,
        "comment": f"LLM-as-Judge 评分：{score:.2f}",
    }


def evaluate_has_structure(run: Run, example: Example) -> dict:
    """评估报告是否有清晰的结构（标题、章节等）"""
    report = (run.outputs or {}).get("report", "")
    
    structure_indicators = ["#", "##", "**", "1.", "2.", "3.", "一、", "二、", "三、"]
    has_structure = any(indicator in report for indicator in structure_indicators)
    
    score = 1.0 if has_structure else 0.3
    comment = "有结构化格式" if has_structure else "缺少结构化格式（建议添加标题和章节）"
    
    return {"key": "has_structure", "score": score, "comment": comment}


# ── 运行评估 ──────────────────────────────────────────────────────────────
def run_evaluation(experiment_name: str = "research_agent_v1"):
    """运行完整的评估流程"""
    
    print("=" * 60)
    print("研究助手 Agent 评估")
    print("=" * 60)
    
    # 1. 准备数据集
    dataset = create_eval_dataset()
    
    # 2. 运行评估
    print(f"\n开始评估实验：{experiment_name}")
    print("评估器：长度检查、关键词覆盖、LLM-as-Judge、结构检查\n")
    
    results = evaluate(
        research_agent_target,
        data=DATASET_NAME,
        evaluators=[
            evaluate_length,
            evaluate_keywords,
            evaluate_quality_llm,
            evaluate_has_structure,
        ],
        experiment_prefix=experiment_name,
        metadata={
            "model": "qwen3.5-plus",
            "agent_version": "1.0",
        },
        max_concurrency=1,  # 顺序执行，避免 rate limit
    )
    
    result_list = list(results)
    print(f"\n评估完成！共 {len(result_list)} 个样本")
    print(f"\n查看详细结果：https://smith.langchain.com")
    print(f"项目：{os.environ.get('LANGCHAIN_PROJECT', 'study')}")
    print(f"实验名称前缀：{experiment_name}")
    
    return result_list


if __name__ == "__main__":
    import sys
    
    experiment = sys.argv[1] if len(sys.argv) > 1 else "research_agent_v1"
    run_evaluation(experiment_name=experiment)
