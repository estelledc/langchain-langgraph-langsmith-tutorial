"""
03_dataset.py — LangSmith 数据集管理

知识点：
- 创建/更新/删除数据集
- 从 Trace 中收集样本到数据集（最实用的功能）
- 数据集版本管理
- 导出数据集
- 使用数据集做回归测试
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

llm = ChatOpenAI(
    model="qwen3.5-plus",
    base_url=os.environ["DASHSCOPE_BASE_URL"],
    api_key=os.environ["DASHSCOPE_API_KEY"],
    temperature=0,
)

ls_client = Client()


# ── 1. 数据集 CRUD 操作 ───────────────────────────────────────────────────────
def demo_dataset_crud():
    print("=" * 50)
    print("【数据集 CRUD 操作】")
    
    dataset_name = f"crud_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 创建数据集
    dataset = ls_client.create_dataset(
        dataset_name=dataset_name,
        description="CRUD 演示数据集",
        data_type="kv",  # key-value 类型
    )
    print(f"\n1. 创建数据集：{dataset_name}（ID: {dataset.id}）")
    
    # 批量添加样本
    examples_data = [
        {
            "inputs": {"text": "今天天气真好！"},
            "outputs": {"sentiment": "positive", "score": 0.9},
        },
        {
            "inputs": {"text": "这个产品质量很差。"},
            "outputs": {"sentiment": "negative", "score": 0.1},
        },
        {
            "inputs": {"text": "还行吧，一般般。"},
            "outputs": {"sentiment": "neutral", "score": 0.5},
        },
    ]
    
    ls_client.create_examples(
        inputs=[e["inputs"] for e in examples_data],
        outputs=[e["outputs"] for e in examples_data],
        dataset_id=dataset.id,
    )
    print(f"2. 添加了 {len(examples_data)} 个样本")
    
    # 查询数据集样本
    examples = list(ls_client.list_examples(dataset_id=dataset.id))
    print(f"3. 查询样本数：{len(examples)}")
    for ex in examples:
        print(f"   - 输入：{ex.inputs}  输出：{ex.outputs}")
    
    # 更新某个样本
    if examples:
        first_example = examples[0]
        ls_client.update_example(
            example_id=first_example.id,
            inputs=first_example.inputs,
            outputs={"sentiment": "very_positive", "score": 0.95},  # 修正标注
        )
        print(f"4. 已更新样本 {first_example.id}")
    
    # 删除数据集（演示结束后清理）
    ls_client.delete_dataset(dataset_id=dataset.id)
    print(f"5. 已删除演示数据集")
    
    return dataset.id


# ── 2. 从 Trace 中收集样本（生产环境最常用）──────────────────────────────────
def demo_collect_from_traces():
    print("\n" + "=" * 50)
    print("【从 Trace 收集样本到数据集（生产环境核心工作流）】")
    
    # 先运行一些会被追踪的调用
    @traceable(name="sentiment_analysis", tags=["sentiment", "production"])
    def analyze_sentiment(text: str) -> dict:
        response = llm.invoke([
            HumanMessage(content=f"判断以下文本的情感倾向（正面/负面/中性），只输出一个词：{text}")
        ])
        sentiment = response.content.strip()
        return {"text": text, "sentiment": sentiment}
    
    test_texts = [
        "LangChain 真的很好用，大大提高了开发效率！",
        "这个 API 文档写得太差了，完全看不懂。",
        "今天学习了 LangGraph，还在消化中。",
    ]
    
    print("\n运行一些会被追踪的调用...")
    results = []
    for text in test_texts:
        result = analyze_sentiment(text)
        results.append(result)
        print(f"  [{result['sentiment']}] {text[:30]}...")
    
    print("\n这些调用已上报到 LangSmith。")
    print("在 LangSmith UI 中，你可以：")
    print("  1. 找到这些 Trace")
    print("  2. 对感兴趣的 Trace 点击 'Add to Dataset'")
    print("  3. 这些 Trace 就变成了评估数据集的样本")
    print("\n这是构建高质量数据集最自然的方式——从真实生产流量中筛选。")


# ── 3. 多版本数据集管理 ───────────────────────────────────────────────────────
def demo_dataset_versioning():
    print("\n" + "=" * 50)
    print("【数据集版本管理】")
    
    dataset_name = "versioning_demo"
    
    # 清理旧数据集（如果存在）
    existing = list(ls_client.list_datasets(dataset_name=dataset_name))
    if existing:
        ls_client.delete_dataset(dataset_id=existing[0].id)
    
    # 创建初始数据集（v1）
    dataset = ls_client.create_dataset(dataset_name=dataset_name)
    
    v1_examples = [
        {"inputs": {"q": "什么是 LLM？"}, "outputs": {"a": "大语言模型"}},
        {"inputs": {"q": "什么是 RAG？"}, "outputs": {"a": "检索增强生成"}},
    ]
    ls_client.create_examples(
        inputs=[e["inputs"] for e in v1_examples],
        outputs=[e["outputs"] for e in v1_examples],
        dataset_id=dataset.id,
    )
    
    v1_count = len(list(ls_client.list_examples(dataset_id=dataset.id)))
    print(f"\nv1 数据集：{v1_count} 个样本")
    
    # 扩充数据集（v2：添加更多样本）
    v2_new_examples = [
        {"inputs": {"q": "什么是 Agent？"}, "outputs": {"a": "能自主使用工具完成任务的 AI 系统"}},
        {"inputs": {"q": "什么是 Vector DB？"}, "outputs": {"a": "存储和检索向量嵌入的专用数据库"}},
        {"inputs": {"q": "什么是 Embedding？"}, "outputs": {"a": "将文本转换为稠密向量的技术"}},
    ]
    ls_client.create_examples(
        inputs=[e["inputs"] for e in v2_new_examples],
        outputs=[e["outputs"] for e in v2_new_examples],
        dataset_id=dataset.id,
    )
    
    v2_count = len(list(ls_client.list_examples(dataset_id=dataset.id)))
    print(f"v2 数据集：{v2_count} 个样本（新增 {v2_count - v1_count} 个）")
    
    # LangSmith 会自动追踪数据集的修改历史
    print("\n在 LangSmith UI 中，可以查看数据集的完整修改历史，")
    print("并可以针对特定版本的数据集重跑评估，实现回归测试。")
    
    # 清理
    ls_client.delete_dataset(dataset_id=dataset.id)
    print("\n演示数据集已清理。")


# ── 4. 导出数据集用于本地分析 ────────────────────────────────────────────────
def demo_export_dataset():
    print("\n" + "=" * 50)
    print("【导出数据集】")
    
    # 查看所有可用数据集
    datasets = list(ls_client.list_datasets())
    print(f"\n当前项目共有 {len(datasets)} 个数据集：")
    for ds in datasets[:5]:  # 只显示前5个
        example_count = len(list(ls_client.list_examples(dataset_id=ds.id, limit=1)))
        print(f"  - {ds.name}（创建时间：{ds.created_at.strftime('%Y-%m-%d')}）")
    
    if not datasets:
        print("  （暂无数据集，请先运行其他演示）")
        return
    
    # 导出第一个数据集为本地 JSON
    target_dataset = datasets[0]
    examples = list(ls_client.list_examples(dataset_id=target_dataset.id))
    
    export_data = [
        {
            "id": str(ex.id),
            "inputs": ex.inputs,
            "outputs": ex.outputs,
            "created_at": ex.created_at.isoformat() if ex.created_at else None,
        }
        for ex in examples
    ]
    
    export_path = f"/tmp/dataset_export_{target_dataset.name}.json"
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n已导出 '{target_dataset.name}' 数据集到：{export_path}")
    print(f"样本数：{len(export_data)}")
    
    if export_data:
        print(f"第一条样本预览：{json.dumps(export_data[0], ensure_ascii=False)[:200]}")


# ── 5. 数据集驱动的回归测试 ──────────────────────────────────────────────────
def demo_regression_test():
    print("\n" + "=" * 50)
    print("【数据集驱动的回归测试】")
    print("（展示如何利用数据集防止模型退化）\n")
    
    from langsmith import evaluate
    
    # 创建回归测试数据集
    dataset_name = "regression_test_demo"
    existing = list(ls_client.list_datasets(dataset_name=dataset_name))
    if existing:
        ls_client.delete_dataset(dataset_id=existing[0].id)
    
    dataset = ls_client.create_dataset(dataset_name=dataset_name)
    ls_client.create_examples(
        inputs=[
            {"question": "1+1等于多少？"},
            {"question": "中国的首都是哪里？"},
            {"question": "Python 是什么编程语言？"},
        ],
        outputs=[
            {"answer": "2"},
            {"answer": "北京"},
            {"answer": "Python 是一种高级、通用、解释型编程语言。"},
        ],
        dataset_id=dataset.id,
    )
    
    def simple_qa(inputs: dict) -> dict:
        response = llm.invoke([HumanMessage(content=inputs["question"])])
        return {"answer": response.content}
    
    def exact_or_contains_evaluator(run, example):
        """检查答案是否包含参考答案中的关键内容"""
        prediction = (run.outputs or {}).get("answer", "").strip()
        reference = (example.outputs or {}).get("answer", "").strip()
        
        # 简单检查：参考答案是否是预测的子集
        score = 1.0 if reference.lower() in prediction.lower() else 0.0
        return {
            "key": "contains_answer",
            "score": score,
            "comment": f"预测：'{prediction[:50]}' | 参考：'{reference}'",
        }
    
    print("运行回归测试...")
    results = evaluate(
        simple_qa,
        data=dataset_name,
        evaluators=[exact_or_contains_evaluator],
        experiment_prefix="regression_v1",
    )
    
    result_list = list(results)
    passed = sum(1 for r in result_list
                 if any(fb.score == 1.0 for fb in (r.get("feedback") or [])))
    
    print(f"测试完成：{len(result_list)} 个用例")
    print("\n每次代码/模型变更后运行此脚本，即可检测是否有功能退化。")
    
    # 清理
    ls_client.delete_dataset(dataset_id=dataset.id)


if __name__ == "__main__":
    demo_dataset_crud()
    demo_collect_from_traces()
    demo_dataset_versioning()
    demo_export_dataset()
    demo_regression_test()
    
    print("\n✅ LangSmith 数据集管理示例完成！")
    print("   最佳实践：")
    print("   1. 从生产 Trace 中持续收集高质量样本")
    print("   2. 人工标注后加入数据集")
    print("   3. 每次迭代后运行回归测试")
    print("   4. 用 A/B 实验对比不同版本的效果")
