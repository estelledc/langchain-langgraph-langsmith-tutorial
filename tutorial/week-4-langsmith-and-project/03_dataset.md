# 03 · Dataset — 把生产数据"养"成评估金矿

> **本步带回家的概念**：数据集不是一次写完的考卷——它是**从生产 trace 里持续筛**养出来的。每次产品迭代，跑同一份 dataset 做回归测试，防止"改完一个 prompt 解了 A 但悄悄破了 B"。
> **配套代码**：[`final/03_langsmith/03_dataset.py`](../../final/03_langsmith/03_dataset.py)
> **预计耗时**：30-40 分钟（dataset 操作概念清晰，密度比 02 低）

---

## 准备 (5 分钟)

- [ ] [02_evaluation.md](02_evaluation.md) 通关，知道 dataset / target / evaluator 关系
- [ ] 跑过 `python final/03_langsmith/03_dataset.py`（5 段 demo，每段都自己清理 dataset，不会污染你账号）
- [ ] `_scratch/my_ls03_dataset.py` 已建好

---

## 任务卡

### 任务 1 · 跑起来感受 dataset 全流程（10 分钟）

**做什么**：

```bash
python final/03_langsmith/03_dataset.py
```

5 段 demo：
1. **CRUD**：创建 / 添加 / 查询 / 更新 / 删除一条龙
2. **从 Trace 收集**：跑了一些 `@traceable` 调用，提示去 UI 手工"Add to Dataset"
3. **版本管理**：v1 → v2 增量加样本
4. **导出**：把 dataset 拉到本地 JSON
5. **回归测试**：固定 dataset + 固定 evaluator，每次跑都能看分数变化

**关键观察**：第 2 段有个动作要你**手工**去 UI 做。打开 LangSmith → 找到 `sentiment_analysis` 标签的 trace → 选一条 → 右上 "Add to Dataset" 试一次。这个动作是**生产数据集养成**的核心。

**给 AI 的 prompt**：

```
我刚跑了 final/03_langsmith/03_dataset.py 的 5 段 demo。

请用日常类比帮我搞清 3 件事：

1. "从 Trace 收集到 Dataset" 这件事——
   能不能比喻成"客服把高频用户问题攒成 FAQ"，
   生产流量是题源，dataset 是精选题库？
2. 数据集版本管理（v1 → v2 增量加）和软件版本管理有什么共同点 / 不同点？
   为什么不直接每次都新建一个 dataset？
3. 回归测试 = 固定 dataset + 固定 evaluator。这跟传统软件单元测试的
   "固定输入 → 固定预期输出"对应得上吗？哪里不一样？
   （提示：LLM 输出有随机性 + evaluator 是软评分）

回答 300 字内，每问独立段落。
```

**自检**：能用一句话讲清「为什么"从生产 trace 攒 dataset"比"工程师拍脑袋写 dataset"靠谱」（提示：真实分布 vs 想象分布）。

---

### 任务 2 · 写 dataset CRUD 一条龙（10 分钟）

**做什么**：在 `_scratch/my_ls03_dataset.py` 里写一个完整的 CRUD 演示：

- 创建 dataset `my_test_ds_<时间戳>`
- 加 3 条 example（input/output 自定义）
- 查询所有 example 打印
- 更新第一条
- 最后**别删**——下一个任务还要用

**给 AI 的 prompt**：

```
我要在 _scratch/my_ls03_dataset.py 里写 dataset CRUD 一条龙。

请引导我（不直接给代码）：
1. ls_client.create_dataset 和 ls_client.create_examples 必须分两步吗？
   为什么不能一步传入 examples？
2. dataset_name 必须唯一吗？
   如果重复 create_dataset 同名的会怎样？
   (提示：会报错；要先 list_datasets 看是否存在)
3. update_example 要传哪些参数？
   只想改 outputs 不想改 inputs，inputs 字段能不能省？

每次只问我一个问题。
```

写完跑：

```bash
python _scratch/my_ls03_dataset.py
```

**自检**：能讲清「dataset 命名为什么常带时间戳」（提示：避免重名报错；版本可追溯）。

---

### 任务 3 · 用任务 2 的 dataset 跑评估（5 分钟）

**做什么**：在 `my_ls03_dataset.py` 里追加一段：

- 写一个最简 target_function（让 LLM 直接基于 inputs["question"] 答）
- 写一个最简 evaluator（关键词包含）
- 用 `evaluate()` 跑你刚建的 dataset

**给 AI 的 prompt**：

```
我要把任务 2 创建的 dataset 用 evaluate() 跑评估。

请引导我（不直接给代码）：
1. evaluate() 的 data 参数传 dataset_name 字符串还是 dataset 对象？两种都行吗？
2. target_function 的输入字段必须和 dataset 的 inputs key 完全一致吗？
   （提示：你 dataset 用的 inputs key 是什么？target 接收的是什么？）
3. evaluate 跑完后能不能继续追加 example 到同一个 dataset，
   再跑第二轮评估？UI 上两次实验是同一个 experiment 还是两个？

每次只问我一个问题。
```

跑完 UI 看分数。

**自检**：能讲清「dataset 是评估的"基底"，target 和 evaluator 可以替换；fix dataset 才能横向对比"算法 v1 vs v2"」。

---

### 任务 4 · 从 trace 手工收集（10 分钟）

**做什么**：

1. 在 `my_ls03_dataset.py` 加一个 `@traceable` 函数，比如 `summarize(text: str) -> str`，让 LLM 总结一段文字
2. 跑 3-5 次（用不同输入文本）
3. 去 LangSmith UI → 找到这些 trace → 选一条 → "Add to Dataset" → 加到任务 2 的 dataset
4. 回到代码 `list_examples` 看是不是新增了一条

**给 AI 的 prompt**：

```
我要从 LangSmith UI 手工把一条 trace 加到我的 dataset。

请引导我（不直接给代码）：
1. UI 操作步骤是什么？我在 trace 详情页应该点哪？
2. "Add to Dataset" 时，UI 会让我选 inputs 字段和 outputs 字段——
   它怎么知道这条 trace 的输入是什么？看的是 @traceable 函数的什么？
3. 加到 dataset 后，我在代码里 list_examples 立刻能看到吗？
   还是有延迟？（提示：通常很快，但底层有同步过程）

每次只问我一个问题。
```

**自检**：能讲清「这是真实生产流程的核心：trace → 人工筛 → 加入 dataset → 跑评估 → 改进 → 再 trace」（提示：闭环）。

---

### 任务 5 · 自检：跟 final 对比 + 思考"何时该建 dataset"（5 分钟）

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_ls03_dataset.py：
[贴代码]

参考答案 final/03_langsmith/03_dataset.py：
[让 AI 自己读]

请帮我对比 + 思考：
1. 我的 CRUD 流程有没有"删除前先 list 检查"防御？final 用了 try/except 吗？
2. dataset 命名的 idempotency（每次跑会不会创建无穷多个 dataset）我处理了吗？
3. "什么时候该建一个新 dataset" 这个产品决策问题：
   - 改 prompt 之后 → 重用旧 dataset 跑回归
   - 加新功能（新意图）→ 新建专门 dataset
   - 模型升级 → 重用旧 dataset 看分数变化
   这个判断你觉得对吗？还有别的场景吗？

哪些是风格差异，哪些是真问题？真问题告诉我"为什么 final 这样写更好"，
不要直接给修改后代码——让我自己改。
```

---

## 通关条件

- [ ] `python _scratch/my_ls03_dataset.py` 跑通完整 CRUD + evaluate
- [ ] 至少手工从 UI 把 1 条 trace 加到了 dataset
- [ ] 能用一句话讲清「dataset 怎么从生产环境'养'出来」
- [ ] LangSmith Datasets 页面能看到自己建的 dataset

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week4-03.md`：

```markdown
# Week 4 · 03_dataset — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → 跳 [04_capstone.md](04_capstone.md)（综合项目，把前 3 周学的东西串起来做一个完整 Agent）
- 想多练 → 给 dataset 加 `metadata` 字段（比如 `{"difficulty": "easy"}`），按难度过滤跑评估
- 真实场景 → 把你之前 week-3 跑过的 ReAct Agent trace 选 5 条加到 dataset，回归测一下
