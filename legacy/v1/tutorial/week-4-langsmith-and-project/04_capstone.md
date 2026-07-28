# 04 · Capstone — 把前 3 周串成一个完整研究助手

> **本步带回家的概念**：单文件 demo 写多了不代表会写"项目"。真实项目是 4 个文件分工：tools 聚集合 / graph 描述流程 / agent 暴露接口 / eval 把质量门关上。这篇不教新概念，教**怎么把前 12 篇学到的东西按职责拆模块、组合成系统**。
> **配套代码**：[`final/04_project/agent.py`](../../final/04_project/agent.py) / [`graph.py`](../../final/04_project/graph.py) / [`tools.py`](../../final/04_project/tools.py) / [`eval.py`](../../final/04_project/eval.py)
> **预计耗时**：60-90 分钟（密度最低，更多是"读懂 + 改一处"）

---

## 准备 (10 分钟)

- [ ] Week 1 / 2 / 3 / 4-01-03 全部通关
- [ ] 跑一次入口：

  ```bash
  python final/04_project/agent.py --question "什么是 RAG？"
  ```

  约 30-90 秒，看到完整研究流程：Planner → Researcher（多轮 + 工具）→ Writer 输出报告
- [ ] 去 [smith.langchain.com](https://smith.langchain.com) 找这次的 trace，**点开看完整执行树**——这是本篇的教学主体
- [ ] `_scratch/my_capstone/` 目录已建好（这次是个目录，不是单文件）

---

## 任务卡

### 任务 1 · 看 trace 树读懂"项目级流程"（15 分钟）

**做什么**：在 LangSmith UI 找到刚才那次 `agent.py` 的 trace。

**重点观察 4 件事**：

1. **顶层 run** 的 name（应该是图编译时的默认名）
2. **节点层级**：`planner` → `researcher` → `tools` → `researcher` → `writer`，每个节点是独立子 run
3. **researcher 多次出现**：看 ReAct 循环的实际形状（多少轮？每轮调了什么工具？）
4. **每个节点的 inputs / outputs**：State 的字段（`research_plan` / `collected_info` / `final_report`）在哪些节点被填进去

**给 AI 的 prompt**：

```
我刚跑了 final/04_project/agent.py 并看了 LangSmith trace 的完整树。

请帮我把 trace 上看到的执行流，对照 graph.py 文件里的图定义，
讲清 4 个问题（用日常类比 + 简短回答）：

1. trace 树的层级（planner → researcher → tools → researcher → writer）
   是不是 graph.py 里 add_edge / add_conditional_edges 的可视化？
   能不能比喻成"流程图实际跑出来的足迹"？
2. researcher 重复出现是 ReAct 循环（researcher → tools → researcher）。
   循环退出条件是 should_continue_research 函数——
   它返回 "use_tools" 时去 tools 节点，返回 "write" 时去 writer 节点。
   这个"路由函数"在 trace 上能看到吗？
3. State（ResearchState）的 5 个字段，在哪些节点会被改？
   - research_plan：__ 节点写
   - collected_info：__ 节点写
   - final_report：__ 节点写
   - iteration_count：__ 节点改
   - messages：__ 节点都会追加
4. 这套架构（Planner → ReAct → Writer）对应的设计模式叫什么？
   能不能比喻成"开会先 PM 拆任务、专家干活、秘书写纪要"？

每问独立段落，每问 100 字内。
```

**自检**：能用一句话讲清「graph.py 是一份"项目流程的可执行说明书"」（提示：把流程图变代码）。

---

### 任务 2 · 拆四文件分工（10 分钟）

**做什么**：依次扫读以下文件**结构（不抠每行代码）**——只看 docstring + 函数签名 + import 关系：

1. `final/04_project/tools.py` — 5 个工具的 `@tool` 函数，导出 `ALL_TOOLS`
2. `final/04_project/graph.py` — `ResearchState` + 4 个节点 + `build_research_graph()` 工厂
3. `final/04_project/agent.py` — argparse + `run_research()` + `interactive_mode()` + `main()`
4. `final/04_project/eval.py` — dataset + 4 个 evaluator + `run_evaluation()`

**给 AI 的 prompt**：

```
我快速扫读了 final/04_project/ 下的 4 个文件（tools.py / graph.py / agent.py / eval.py）。

请引导我建立模块分工心智模型（不直接复述代码内容）：

1. 这 4 个文件的依赖关系是什么？谁 import 谁？
   （提示：agent.py imports graph.py imports tools.py，eval.py imports agent.py）
2. 为什么不把 tools 直接写在 graph.py 里？分文件的好处是什么？
   （想想：如果以后要复用 tools 给另一个 graph 用呢？）
3. agent.py 同时支持 CLI、演示问题、交互模式——
   这种"一个入口多个使用方式"的设计对应什么实际场景？
   （提示：开发自测、给同事 demo、做产品 PoC）
4. eval.py 没参与生产流程（agent.py 跑生产请求时不调 eval.py），
   它独立存在是为什么？什么时候跑它？
   （提示：CI / 改完代码后回归测试）

每问 100 字内，从日常类比开始。
```

**自检**：能用一段话讲清「这 4 个文件的边界为什么这么切」（按"复用范围"和"调用时机"切，不是按代码量切）。

---

### 任务 3 · 自己加一个工具（15 分钟）

**做什么**：在 `_scratch/my_capstone/` 里**复制** `tools.py` / `graph.py` / `agent.py`（用 `cp` 命令），然后**只改你自己的 tools.py**：

- 加一个新工具 `count_chinese_chars(text: str) -> int`：数中文字符数
- 在 `ALL_TOOLS` 列表里把它加上

```bash
mkdir -p _scratch/my_capstone
cp final/04_project/{tools,graph,agent}.py _scratch/my_capstone/
# 改 _scratch/my_capstone/tools.py，加新工具
# 改 _scratch/my_capstone/graph.py 和 agent.py 的 import 路径（如果有需要）
```

**给 AI 的 prompt**：

```
我把 tools/graph/agent.py 复制到 _scratch/my_capstone/，
要在我的 tools.py 加一个 count_chinese_chars 工具。

请引导我（不直接给代码）：
1. 我改 _scratch/my_capstone/tools.py 后，
   _scratch/my_capstone/graph.py 的 import 还能找到我的工具吗？
   （提示：sys.path / 相对 import 怎么处理）
2. 加完工具后跑 _scratch/my_capstone/agent.py，
   怎么验证 LLM 真的"知道"这个新工具？
   （提示：构造一个明显需要数字符数的问题）
3. 如果新工具的 docstring 写得不好，会发生什么？
   （提示：LLM 可能不调它，或者调错；这跟 week-2/01 任务 4 报错练习一脉相承）

每次只问我一个问题。
```

跑：

```bash
python _scratch/my_capstone/agent.py --question "请帮我数一下「LangChain 真好用」是几个中文字"
```

**自检**：去 LangSmith 看你的 trace，能在 tools 节点的子调用里看到 `count_chinese_chars` 被调过。

---

### 任务 4 · 跑 eval.py 评估你改完的 Agent（10 分钟）

**做什么**：用 `final/04_project/eval.py` 评估**你改的**那个 agent。

```bash
# 改 _scratch/my_capstone/eval.py（复制过来再改）
cp final/04_project/eval.py _scratch/my_capstone/
# 改 import 把 from agent import run_research 改成 from _scratch.my_capstone.agent import run_research
# 或者更简单：cd 进 _scratch/my_capstone/ 跑

cd _scratch/my_capstone && python eval.py my_capstone_v1
```

跑完去 LangSmith Experiments 看 4 个 evaluator 的分数。

**给 AI 的 prompt**：

```
我跑了 my_capstone 版的 eval.py，看到 LangSmith Experiments 里的 4 个分数：
- report_length：__
- keyword_coverage：__
- llm_quality_judge：__
- has_structure：__

请帮我思考 3 件事（不要直接给优化建议，引导我自己想）：

1. 哪个 evaluator 的分数最低？这反映了 Agent 的什么短板？
2. 如果我要改进，应该改 graph.py 的哪个节点？
   （比如分数低的是 has_structure，应该改 writer_node 的 prompt）
3. 改完之后再跑一次 eval，experiment_prefix 用什么名字才能跟原版对比？
   （提示：v2 / 改进版 / 时间戳，差别是什么）

每问 80 字内。
```

**自检**：能讲清「eval 不是一次性活动，是每次改 prompt / 改架构都要跑一次的回归门」。

---

### 任务 5 · 自检：通览整个项目（10 分钟）

**给 AI 的 prompt**（最后一次任务卡式 prompt，用满）：

```
我快通关 capstone。请帮我做一次"项目级自检"——
针对 final/04_project/ 整套代码，回答以下问题（每问 100 字内）：

1. 用一段话讲清"这个研究助手 Agent 是怎么工作的"——
   假设你在向一个不懂代码的 PM 解释。
2. 这个项目用到的 LangChain / LangGraph / LangSmith 三层
   分别承担什么角色？缺一个会怎样？
3. 这个项目里**没有**用到的 LangGraph 高级特性是什么？
   （HITL? 多 Agent? Subgraph?）以后什么场景会用到？
4. 如果我要把这个项目改造成"客服助手"——
   tools / graph / eval 各要改什么？哪些不用动？
5. 我现在能不能从 0 开始写一个类似规模的 Agent 项目？
   还差什么？（最诚实的回答，不要客套）

最后一问尤其重要——基于我们这 4 周的对话历史，
你能给我什么诚实的"还差什么"的反馈？
```

**自检**：把 AI 的回答存到 `_scratch/journal/2026-XX-XX-week4-04-capstone-self-review.md` —— 这是你 4 周学习的"出师证"。

---

## 通关条件

- [ ] `python _scratch/my_capstone/agent.py` 跑通完整流程
- [ ] 至少加了 1 个新工具，并在 trace 里看到它被调过
- [ ] 至少跑过一次自己版本的 eval.py，看到 4 个评估分数
- [ ] 写了"项目级自检"日志（任务 5）
- [ ] 能用 5 分钟向一个不懂代码的人讲清"这个 Agent 怎么工作的"

---

## 卡点日志（最后一次必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week4-04-capstone.md`：

```markdown
# Week 4 · 04_capstone — 卡点日志（最终）

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）

## 4 周学习总复盘

### 现在能不打草稿讲清的概念
- LangChain：__
- LangGraph：__
- LangSmith：__
- LCEL：__
- ReAct：__
- HITL：__
- Checkpointer：__
- LLM-as-Judge：__

（讲不清的回头看那一章卡点日志）

### 4 周里最有效的 prompt（贴 3-5 个）

### 我用 AI 学习的方式跟刚开始有什么变化

### 下一步想做什么
（继续深入哪个方向？看哪些资料？）
```

---

## 通往下一站

恭喜！你已经走完了这套 4 周的脚手架。下一步建议：

1. **真做一个项目**：从你工作 / 兴趣里挑一个真实任务（自动写 PR 摘要、整理收藏夹、追新闻），用 capstone 这套架构改造一遍
2. **看官方进阶**：[LangGraph Cookbook](https://langchain-ai.github.io/langgraph/tutorials/) 里的 "Self-RAG" / "Plan-and-Execute" 是这套架构的官方进阶变种
3. **读源码**：直接读 `langgraph/prebuilt.py`，看 `tools_condition` / `ToolNode` 这些"省你写代码"的函数到底实现了什么——有了 4 周基础，读起来不会卡了

最后：把 [tutorial/README.md](../README.md) 状态自己更新成 ✅ Done (4/4)，给自己点个赞。
