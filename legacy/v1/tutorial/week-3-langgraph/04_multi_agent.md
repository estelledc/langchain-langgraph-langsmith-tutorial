# 04 · Multi-Agent — 多个 Agent 协作

> **本步带回家的概念**：复杂任务拆给多个专业 Agent，靠"父图嵌套子图"组织。两种主流模式：**Supervisor**（一个总控分派任务）和**并行**（多个独立任务同时跑、最后汇总）。
> **配套代码**：[`final/02_langgraph/04_multi_agent.py`](../../final/02_langgraph/04_multi_agent.py)
> **预计耗时**：50-65 分钟（最后一篇 LangGraph，密度仍较高）

---

## 准备 (5 分钟)

- [ ] [03_human_in_the_loop.md](03_human_in_the_loop.md) 通关
- [ ] 跑过 `python final/02_langgraph/04_multi_agent.py`，看到 Supervisor 段 + 并行段两次输出
- [ ] `_scratch/my_g04_multi.py` 已建好

---

## 任务卡

### 任务 1 · 跑起来感受两种模式（10 分钟）

**做什么**：

```bash
python final/02_langgraph/04_multi_agent.py
```

观察两段：

1. **Supervisor**：你给一个混合任务"了解 RAG + 写 Python 示例"。Supervisor 节点先判断该叫谁，调 researcher 子图返回研究结果，再叫 coder 子图返回代码，最后输出 FINISH 走 END。
2. **并行**：从 START 同时连到 research 和 analysis 两个节点，两个跑完后用 `add_edge(["research", "analysis"], "synthesize")` 汇合。

**关键观察**：Supervisor 段的 print 顺序——`[Supervisor] 决定：researcher` → `[Researcher] 开始工作` → `[Researcher] 完成` → `[Supervisor] 决定：coder` → ...

**给 AI 的 prompt**：

```
我刚跑了 final/02_langgraph/04_multi_agent.py 的两段。

请用日常类比帮我搞清 3 件事：

1. Supervisor 模式像不像"项目经理 + 专家团"——
   PM 接需求、判断该叫谁、把活分下去、收回结果再决定下一步？
   什么时候用 Supervisor 比纯 ReAct（一个 Agent + 多工具）好？
2. "把一个编译好的图当作另一个图的节点"——这个 Subgraph 概念，
   能不能比喻成"调用一个写好的 Python 函数"？区别是什么？
3. 并行模式用 add_edge(START, "A") + add_edge(START, "B") 让两个节点同时启动；
   再用 add_edge(["A", "B"], "C") 让 C 等两个都跑完才启动。
   这是不是就是"并发 + barrier"？

回答 350 字内，每问独立段落。
```

**自检**：能讲清「什么场景用 Supervisor，什么场景用并行」（提示：Supervisor 适合任务有依赖、需要动态决策；并行适合任务彼此独立可同时做）。

---

### 任务 2 · 写一个最简 Subgraph（10 分钟）

**做什么**：在 `_scratch/my_g04_multi.py` 里写一个**最小 Subgraph 嵌套**：

- 子图 `greeter`：单节点，让 LLM 用一句话问候
- 父图：START → greeter（作为节点）→ END

不带工具、不带 Supervisor，只验证"子图能当节点"。

**给 AI 的 prompt**：

```
我要在 _scratch/my_g04_multi.py 里写一个最简的 Subgraph 嵌套：
子图 greeter 单节点；父图调用子图作为一个节点，仅此而已。

请引导我（不直接给代码）：
1. 子图 compile() 后得到的 Runnable，怎么塞进父图当 node？
   是直接 add_node("name", subgraph) 吗？
2. 子图和父图的 State schema 必须一样吗？
   不一样会怎样？（提示：跟 messages 字段是否共享有关）
3. 子图内部的 START / END 是子图的还是父图的？
   父图调子图时，子图从子图的 START 开始执行，对吗？

每次只问我一个问题。
```

写完跑：

```bash
python _scratch/my_g04_multi.py
```

**自检**：能讲清「子图作为 node 跟普通函数 node 的差别」（提示：子图本身有 State，更"重"；适合封装"完整的子流程"，普通函数 node 适合"一步操作"）。

---

### 任务 3 · 加 Supervisor 协调两个子 Agent（15 分钟）

**做什么**：扩任务 2，写两个子图：

- `joker`：让 LLM 讲个笑话（带 SystemMessage "你是脱口秀演员"）
- `serious`：让 LLM 严肃回答（带 SystemMessage "你是严谨的学者"）

父图 Supervisor 节点根据用户输入决定叫谁，调完后回到 Supervisor 决定是 FINISH 还是继续。

**给 AI 的 prompt**：

```
我要把 my_g04_multi.py 扩成 Supervisor 模式：
- 两个子图：joker（讲笑话）/ serious（严肃回答）
- 父图 Supervisor 决定调谁，最后输出 FINISH 走 END

请引导我（不直接给代码）：
1. Supervisor 节点的 SystemMessage 怎么写，让 LLM 输出准确的下一步标签？
   （提示：明确列出可选项 + "只输出一个词" 的约束）
2. 父图 State 应该有几个字段？为什么除了 messages 还要 next_agent？
3. 子图调用完之后，父图从子图返回的 messages 字段怎么 merge 到父图 messages？
   （提示：跟 add_messages reducer 有关）

每次只问我一个问题。
```

跑测试：「先讲个程序员笑话，再严肃讲一下程序员真实工作压力」——观察 Supervisor 是否调了 joker 和 serious 各一次。

**自检**：能讲清「为什么 Supervisor 需要循环（researcher → supervisor → coder → supervisor → END），不能 researcher → coder → END」（提示：必须每完成一步都让 Supervisor 决定下一步是 FINISH 还是继续；硬连接灵活度差）。

---

### 任务 4 · 加并行——两个子任务同时跑（10 分钟）

**做什么**：再加一个**并行**示例（独立函数 `demo_parallel`），不嵌入 Supervisor：

- 两个独立任务：`fetch_weather`（让 LLM 假装查天气）+ `fetch_news`（让 LLM 假装查新闻）
- 从 START 同时连到这两个
- 两个都完成后 `add_edge(["fetch_weather", "fetch_news"], "summarize")`

**给 AI 的 prompt**：

```
我要在 my_g04_multi.py 加一个并行示例。
两个任务（查天气、查新闻）从 START 同时启动；都跑完后汇合到 summarize 节点。

请引导我（不直接给代码）：
1. add_edge(START, "A") 和 add_edge(START, "B") 同时存在时，
   LangGraph 是真的并发（多线程/异步）还是顺序跑两个？
2. add_edge(["A", "B"], "C") 这种 list 形式的来源节点，
   语义是 OR（任一完成就跑 C）还是 AND（都完成才跑 C）？
3. State schema 设计：weather_result 和 news_result 是两个独立字段
   还是塞同一个 dict？哪种更好？

每次只问我一个问题。
```

写完跑，观察总耗时（如果真并行，应该接近 max(weather_time, news_time)，不是 sum）。

**自检**：能讲清「LangGraph 的并行靠什么实现」（提示：两个节点从同一个上游同时被触发；底层是 asyncio 调度）。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_g04_multi.py：
[贴代码]

参考答案 final/02_langgraph/04_multi_agent.py：
[让 AI 自己读]

请帮我对比 4 处：
1. 子图的 ToolNode + tools_condition 接入方式我对吗？
2. Supervisor 决策的 SystemMessage 我写得明确度够吗？
3. 父图返回 next_agent 字段的更新方式（return dict 而不是直接 mutate）
4. 并行节点的 add_edge(["A","B"], "C") 写法我用对了吗？

哪些是风格差异，哪些是真问题？真问题告诉我"为什么 final 这样写更好"，
不要直接给修改后代码——让我自己改。
```

---

## 通关条件

- [ ] `python _scratch/my_g04_multi.py` 跑通
- [ ] 至少包含 1 个 Supervisor 模式（含 ≥ 2 个子图）+ 1 个并行模式
- [ ] [smith.langchain.com](https://smith.langchain.com) 看到嵌套 trace（父图 → 子图 → 节点的层级）
- [ ] 能用一句话讲清「Supervisor 适合什么 / 并行适合什么」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week3-04.md`：

```markdown
# Week 3 · 04_multi_agent — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## Week 3 收尾自评

走完 4 篇 LangGraph 教程后，问自己：

1. 现在能不能用一句话**分别**讲清以下 4 件事？
   - StateGraph 三件套是什么
   - tools_condition 在干嘛
   - Checkpointer + thread_id 解决什么问题
   - Supervisor vs 并行的差别

讲不清的回头补——不要硬走 Week 4。

---

## 通往下一站

- 全部通关 → 跳 [week-4-langsmith-and-project/01_tracing.md](../week-4-langsmith-and-project/01_tracing.md)（开始学怎么"看 Agent 内部发生什么"）
- 想多练 → 把 Supervisor 改成"3 个 Agent + Supervisor 限制每个 Agent 最多调 1 次"，自己写条件函数
