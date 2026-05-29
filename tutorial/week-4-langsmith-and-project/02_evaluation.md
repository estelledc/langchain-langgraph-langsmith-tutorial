# 02 · Evaluation — 怎么知道 AI 答得好不好

> **本步带回家的概念**：评估 = 测试集 + 多个评估器 + 跑批输出可对比的分数表。LangSmith 把这套工程化了：写一个 target 函数 + 几个 evaluator 函数 + 一个 dataset 名字，`evaluate()` 一调就跑批 + 上报。
> **配套代码**：[`final/03_langsmith/02_evaluation.py`](../../final/03_langsmith/02_evaluation.py)
> **预计耗时**：45-60 分钟（涉及 dataset/evaluator/A-B 三大概念，密度回升）

---

## 准备 (5 分钟)

- [ ] [01_tracing.md](01_tracing.md) 通关，能读 LangSmith UI 上的 trace
- [ ] 先**只跑** `run_evaluation`（不跑 A/B），耗时也要 ~10 分钟，因为是顺序跑：
  ```bash
  python -c "from final.langsmith_02 import run_evaluation; run_evaluation()" 
  ```
  实在嫌慢就 `python final/03_langsmith/02_evaluation.py` 一并跑（但 A/B 又会 ×2）
- [ ] 跑完去 [smith.langchain.com](https://smith.langchain.com) 的 **Experiments** 页面看结果
- [ ] `_scratch/my_ls02_eval.py` 已建好

---

## 任务卡

### 任务 1 · 跑评估 + 在 UI 看结果（10 分钟）

**做什么**：

```bash
python final/03_langsmith/02_evaluation.py
```

跑完去 LangSmith：
1. 左侧 **Datasets** → 找到 `langchain_qa_eval_v1`，点进去看 5 条 example
2. 左侧 **Experiments** → 看刚才跑的实验（`qa_baseline_*` / `prompt_v1_concise_*` / `prompt_v2_detailed_*`）
3. 点开任一实验 → 看每条样本的 3 个 evaluator 分数（length / keyword / llm_judge）

**核心理解**：评估不是给"对/错"分类，是给**多维分数**——不同 evaluator 测不同维度，UI 让你横向对比"prompt v1 vs v2 哪个综合分高"。

**为什么 final 用了自定义 LLM-as-Judge 而不是 langsmith 内置**：

> 见 [docs/test-runs.md 1.3 节](../../docs/test-runs.md)：langsmith 0.8 移除了 `LangChainStringEvaluator("qa")`。**生产环境推荐自定义 evaluator** 因为内置会随版本变。

**给 AI 的 prompt**：

```
我刚跑了 final/03_langsmith/02_evaluation.py 的 run_evaluation 和 run_ab_comparison。
LangSmith Experiments 页面有 3 个实验：qa_baseline、prompt_v1_concise、prompt_v2_detailed。

请用日常类比帮我搞清 4 件事：

1. Dataset / target_function / evaluators 三件套，
   能不能比喻成"考卷 / 学生答题 / 阅卷标准"？
2. 为什么需要多个 evaluator（length、keyword、llm_judge）而不是一个？
   像不像考试分语文阅卷有"字数符合""关键词覆盖""理解准确"几项？
3. LLM-as-Judge 是什么——让另一个 LLM 当"阅卷老师"评第一个 LLM 的答案？
   这事可靠吗？什么场景能用？
4. A/B 对比实验（prompt_v1 vs prompt_v2）在产品迭代里的角色——
   是不是"上线新版前的回归测试"？

回答 350 字内，每问独立段落。
```

**自检**：能讲清「为什么自定义 evaluator 比内置更可靠」（提示：版本不变内置就不变；自定义函数你自己控）。

---

### 任务 2 · 写一个最简评估流程（15 分钟）

**做什么**：在 `_scratch/my_ls02_eval.py` 里**完全自己写**一遍评估三件套：

1. **Dataset**（3-5 条简单 QA，比如"中国首都是哪？"→"北京"）
2. **target_function**：拿一个最简 LLM chain 答问题
3. **一个 evaluator**：检查回答里**是否包含**参考答案的关键词（最简实现：`reference in prediction`）

数据集名字用你自己的（不要复用 final 的，避免污染）。

**给 AI 的 prompt**：

```
我要在 _scratch/my_ls02_eval.py 里写最简的 LangSmith 评估三件套。

请引导我（不直接给代码）：
1. ls_client.create_dataset / create_examples 这两个调用顺序
   和参数关系是什么？为什么要先 create_dataset 再 create_examples，
   不能一步到位？
2. target_function(inputs: dict) -> dict 的 inputs / outputs 字段
   必须和 dataset 的 inputs / outputs 一致吗？
   如果 dataset inputs 是 {"question": ...} 而 target_function
   接收 {"q": ...} 会怎样？
3. evaluator 函数的签名 (run: Run, example: Example) -> dict——
   返回 dict 必须包含哪个 key 才能被 LangSmith 识别？

每次只问我一个问题。
```

写完跑：

```bash
python _scratch/my_ls02_eval.py
```

去 LangSmith Experiments 看你的实验。

**自检**：能讲清「dataset 是评估的"考卷"，每条 example 包含 inputs（题目）和 outputs（参考答案）」。

---

### 任务 3 · 加 LLM-as-Judge（10 分钟）

**做什么**：在 `my_ls02_eval.py` 里加一个 `llm_judge` evaluator：让 LLM 输出 0/0.5/1 评判预测和参考的语义一致性。

**给 AI 的 prompt**：

```
我要给 my_ls02_eval.py 加一个 LLM-as-Judge evaluator。
让 LLM 看（问题 / 参考答案 / 预测答案）输出 0、0.5 或 1。

请引导我（不直接给代码）：
1. judge_prompt 的 SystemMessage 应该约束 LLM 输出什么格式？
   只输出数字 vs 输出"分数 + 理由" 哪种好？为什么？
2. LLM 输出 "1.0" / "1" / "1\n理由" 三种情况，
   解析时怎么写最容错？（提示：strip + split + float + 异常兜底）
3. evaluator 的 score 范围必须 [0, 1] 吗？
   超出范围会怎样？（提示：UI 会显示但语义混乱；最好 clamp）

每次只问我一个问题。
```

写完跑，去 UI 对比 keyword 和 llm_judge 哪个分数更"合理"。

**自检**：能讲清「LLM-as-Judge 的两个常见坑」（提示：1. 自洽性差，让 LLM 跑 3 遍取众数能稳；2. 受 prompt 偏置影响——"严格"和"宽松"的 system prompt 给的分会差很多）。

---

### 任务 4 · A/B 对比实验（10 分钟）

**做什么**：在 `my_ls02_eval.py` 里做一个最简 A/B：

- v1：system = "简短回答，10 字内"
- v2：system = "详细回答，必须含原因和例子"

跑同一个 dataset、同一组 evaluators（含 llm_judge），看哪个 v 综合分高。

**给 AI 的 prompt**：

```
我要做 A/B 对比实验：v1 简短回答 / v2 详细回答，
跑同一个 dataset 和 evaluators。

请引导我（不直接给代码）：
1. evaluate() 的 experiment_prefix 参数有什么用？
   两个实验用同一个 prefix 会发生什么？
2. 怎么在 LangSmith UI 把两个实验对比看？
   （提示：Experiments 页面勾选两个能并排）
3. 如果 length evaluator 给 v1 高分（短答案命中区间）
   但 llm_judge 给 v2 高分（更准确）——这种"评估器矛盾"怎么处理？
   产品决策时是看综合还是某一项？

每次只问我一个问题。
```

跑完真的去 UI 横向对比。

**自检**：能讲清「评估器之间会冲突，需要根据业务目标定权重」（提示：用户体验首要 → llm_judge 权重高；token 成本控制首要 → length 权重高）。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_ls02_eval.py：
[贴代码]

参考答案 final/03_langsmith/02_evaluation.py：
[让 AI 自己读]

请帮我对比：
1. dataset 创建的 idempotency（final 用 list_datasets 检查是否已存在再决定 create——我有这个吗？没有会导致每次跑都创建新 dataset，污染数据）
2. evaluator 函数返回 dict 的字段（key / score / comment）我都齐了吗？
3. LLM-as-Judge 的 prompt 和容错解析我跟 final 的差别在哪？
4. A/B 实验有没有用 metadata 标注模型版本？

哪些是风格差异，哪些是真问题？真问题告诉我"为什么 final 这样写更好"，
不要直接给修改后代码——让我自己改。
```

---

## 通关条件

- [ ] `python _scratch/my_ls02_eval.py` 跑通
- [ ] 至少 1 个 dataset + 1 个 target_function + 2 个 evaluator（含 llm_judge）
- [ ] 至少跑过 1 次 A/B 对比
- [ ] 在 LangSmith Experiments 能看到自己的实验结果
- [ ] 能用一句话讲清「LLM-as-Judge 不是银弹，要配合人工抽查」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week4-02.md`：

```markdown
# Week 4 · 02_evaluation — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → 跳 [03_dataset.md](03_dataset.md)（dataset 的更深入用法）
- 想多练 → 把任务 2 的 dataset 扩到 20 条，跑同一个 v1/v2，看分数差是不是更稳定
- LLM-as-Judge 想做更准 → 让 judge LLM 跑 3 次取众数（self-consistency 思路）
