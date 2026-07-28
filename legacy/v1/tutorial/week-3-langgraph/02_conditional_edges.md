# 02 · Conditional Edges — 让图自己决定走哪条路

> **本步带回家的概念**：条件边 = 节点之间的"路由开关"。结合 `tools_condition` 这种内置函数，你能用 ~30 行代码实现一个 ReAct Agent（LLM 推理 → 调工具 → 看结果 → 再推理）。再加 Checkpointer，跨轮对话也免费送。
> **配套代码**：[`final/02_langgraph/02_conditional_edges.py`](../../final/02_langgraph/02_conditional_edges.py)
> **预计耗时**：50-65 分钟

---

## 准备 (5 分钟)

- [ ] [01_simple_graph.md](01_simple_graph.md) 通关，能讲清条件边和普通边的差别
- [ ] 跑过 `python final/02_langgraph/02_conditional_edges.py` 一次，看到 3 段输出
- [ ] `_scratch/my_g02_react.py` 已建好

---

## 任务卡

### 任务 1 · 跑起来感受 ReAct 循环（10 分钟）

**做什么**：

```bash
python final/02_langgraph/02_conditional_edges.py
```

3 段 demo：
1. **ReAct Agent**：用 `tools_condition` 自动路由——LLM 输出有 tool_calls 就跳 tools，否则结束
2. **带 Checkpointer 的多轮对话**：MemorySaver + thread_id，3 轮对话能上下文连贯
3. **自定义多分支路由**：先分类用户意图，再路由到不同处理节点

**关键观察**：第二段第 3 轮"这两个城市哪个温度更高？"——Agent 能记住前两轮拿到的天气数据，这就是 Checkpointer 的力量。

**给 AI 的 prompt**：

```
我刚跑了 final/02_langgraph/02_conditional_edges.py 的 3 段 demo。

请用日常类比帮我搞清 3 件事：

1. tools_condition 这个内置函数到底在判断什么？
   它检查 messages 的最后一条，看里面有没有 tool_calls 字段——
   能不能比喻成"看消息上有没有贴'我要打电话给某人'的便签"？
2. ReAct 的循环（agent → tools → agent → tools → ... → END）
   是怎么靠条件边实现的？为什么 tools 之后必须回到 agent，
   不能直接 END？
3. Checkpointer + thread_id 是什么组合？
   有了这个，多用户场景下"小明的对话"和"小红的对话"
   怎么不互相串台？

回答 300 字内，每问独立段落，从日常类比开始。
```

**自检**：你能用一句话讲清「ReAct 模式 = LLM 推理（Reasoning）+ 调工具（Acting）的反复循环」+「Checkpointer 的本质是什么」（提示：每次 invoke 后给 State 拍快照，下次同 thread_id 来了把快照恢复出来）。

---

### 任务 2 · 自己拼一个 ReAct Agent（15 分钟）

**做什么**：在 `_scratch/my_g02_react.py` 里**完整重写**一个 ReAct Agent，工具自选 2 个：

- 选 1：`flip_coin() -> str` 抛硬币（return "正面" / "反面"）
- 选 2：`roll_dice(sides: int = 6) -> int` 掷骰子
- 选 3：`temp_convert(value: float, unit: str) -> str` 摄氏华氏互转

挑 2 个写。

**给 AI 的 prompt**：

```
我要在 _scratch/my_g02_react.py 里写一个最简 ReAct Agent，
2 个工具自选 [说出你选的两个]。

请引导我（不直接给代码）：
1. ToolNode 和 llm.bind_tools(tools) 各自负责什么？
   ToolNode 是"工具执行器"还是"工具路由器"？
2. 用 tools_condition 时，agent 节点必须返回什么样的消息
   才能让 tools_condition 决定路由？（提示：必须是包含 tool_calls 的 AIMessage）
3. graph 的 4 步骤（add_node × 2、add_edge START→agent、
   add_conditional_edges、add_edge tools→agent）顺序能换吗？
   为什么？

每次只问我一个问题。
```

代码写完后让 Agent 回答："抛 3 次硬币然后掷一次骰子，告诉我结果。"

```bash
python _scratch/my_g02_react.py
```

观察：Agent 应该调 `flip_coin` 三次（或一次循环）+ `roll_dice` 一次。

**自检**：能讲清「`bind_tools(tools)` 改变了 LLM 的什么行为」（提示：让 LLM 学会输出特殊格式的 tool_calls，不再是纯文本回答）。

---

### 任务 3 · 加 Checkpointer 跑多轮（10 分钟）

**做什么**：把任务 2 的图加上 MemorySaver 和 thread_id，跑 3 轮对话，第 3 轮要"引用"前两轮：

- 第 1 轮：「抛 1 次硬币」
- 第 2 轮：「再抛 2 次」
- 第 3 轮：「**前面**抛了几次硬币？正面比反面多吗？」

**给 AI 的 prompt**：

```
我要给 ReAct Agent 加 Checkpointer 实现多轮对话。

请引导我（不直接给代码）：
1. graph.compile(checkpointer=memory) 这一改动，
   底层框架多做了什么事？（提示：每次 invoke 前后会自动读写 state）
2. config 里的 thread_id="xxx" 是干嘛的？
   如果两个不同 thread_id 调用，状态是怎么隔离的？
3. 同一个 thread_id 下，第 2 轮 invoke 时框架内部
   是怎么把第 1 轮的 messages 拼到当前请求里的？
   （提示：跟 add_messages reducer 有关）

每次只问我一个问题，从日常类比开始。
```

写完跑：

```bash
python _scratch/my_g02_react.py
```

**自检**：能讲清「为什么 thread_id 是字符串不是整数」（开放思考：用户 ID / session ID 通常是 UUID 或邮箱）+「MemorySaver 重启进程会丢，要持久化用什么」（提示：SqliteSaver / PostgresSaver）。

---

### 任务 4 · 自定义条件函数（10 分钟）

**做什么**：除了 `tools_condition`，你能写自己的条件函数。在 `my_g02_react.py` 里加一个限制：

- Agent 调工具次数超过 3 次就强制结束（不再让 LLM 继续推理）

**给 AI 的 prompt**：

```
我要给 ReAct Agent 加一个安全开关：
工具调用累计超过 3 次就强制 END，避免 LLM 死循环。

请引导我（不直接给代码）：
1. 我应该改 State schema 加一个 tool_call_count 字段吗？
   还是从 messages 里数 ToolMessage 数量？两种做法各有什么优劣？
2. 自定义条件函数的签名是什么？返回什么？
   它和 tools_condition 接口一致吗？
3. 自定义条件函数要替换 tools_condition，还是组合使用？
   如果组合，怎么写？

每次只问我一个问题。
```

实现完跑一次故意会触发循环的问题：「请连续抛 5 次硬币」——观察是否在第 3 次工具调用后被强制结束。

**自检**：能讲清「为什么 LangGraph 比手写 while + if 更好维护」（提示：图结构可视化、State 存档恢复、所有路由都在 add_edge/add_conditional_edges 显式声明）。

---

### 任务 5 · 自检：跟 final 对比（10 分钟）

**做什么**：把你的 `_scratch/my_g02_react.py` 和 `final/02_langgraph/02_conditional_edges.py` 摆一起。

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_g02_react.py：
[贴代码]

参考答案 final/02_langgraph/02_conditional_edges.py：
[让 AI 自己读]

请帮我在 4 个层次对比：
1. ReAct 图拓扑（边的数量和方向）
2. ToolNode / tools_condition 的用法（我有没有用错？）
3. Checkpointer 接入方式（compile 时的 keyword 是不是 checkpointer=）
4. 自定义条件函数的位置（应该在哪个节点之后挂条件边？）

哪些是风格差异，哪些是真问题？
真问题告诉我"为什么 final 这样写更好"，不要直接给代码——让我自己改。
```

**自检**：改完跑通 + 能用 `graph.get_graph().draw_ascii()` 画出你心里的图。

---

## 通关条件

- [ ] `python _scratch/my_g02_react.py` 跑通
- [ ] 至少包含 1 个 ReAct Agent + 1 个 Checkpointer 多轮对话
- [ ] 至少有 1 个自定义条件函数（不只是用 `tools_condition`）
- [ ] [smith.langchain.com](https://smith.langchain.com) 看到 ReAct 的 trace（agent 节点和 tools 节点交替出现）
- [ ] 能用一句话讲清「条件边 + tools_condition 怎么实现 ReAct」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week3-02.md`：

```markdown
# Week 3 · 02_conditional_edges — 卡点日志

## 卡点
- 任务 X：卡了 ___ 分钟，卡在 ___

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → 跳 [03_human_in_the_loop.md](03_human_in_the_loop.md)（学会让图"暂停等用户审批"）
- 卡在 ReAct 循环 → 用纸笔画图：3 节点 + 4 条边（含条件边），每个节点旁标注 messages 长什么样
- 想多练 → 把任务 4 的"3 次上限"改成"按工具种类分别限制"（calculate 限 5 次，weather 限 2 次）
