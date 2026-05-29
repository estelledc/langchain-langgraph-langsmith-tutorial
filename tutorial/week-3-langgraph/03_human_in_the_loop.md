# 03 · Human-in-the-Loop — 让图中途停下等人

> **本步带回家的概念**：图可以在执行到一半时**暂停**，等用户审批 / 改写 / 拒绝，再继续。靠 Checkpointer 做"暂停 = 存档"，靠 `interrupt_before` 或 `interrupt()` 触发暂停，靠 `Command(resume=...)` 或 `update_state` 注入人类决策。
> **配套代码**：[`final/02_langgraph/03_human_in_the_loop.py`](../../final/02_langgraph/03_human_in_the_loop.py)
> **预计耗时**：40-55 分钟

---

## 准备 (5 分钟)

- [ ] [02_conditional_edges.md](02_conditional_edges.md) 通关，能讲清 Checkpointer + thread_id
- [ ] 跑过 `python final/02_langgraph/03_human_in_the_loop.py`（先看自动化部分；交互部分输入 y 试一次）
- [ ] `_scratch/my_g03_hitl.py` 已建好

---

## 任务卡

### 任务 1 · 跑起来感受"暂停 → 审批 → 恢复"（10 分钟）

**做什么**：

```bash
python final/02_langgraph/03_human_in_the_loop.py
```

3 段 demo：
1. **多步审核工作流**（自动化）：plan → approve → execute → END，模拟"批准/取消"分支
2. **interrupt_before**（交互）：图在执行 `tools` 节点**之前**暂停，等你输 y/n
3. **interrupt() 节点内动态暂停**（交互）：在 `human_review` 节点里调 `interrupt({...})`，把内容传给外面，等用户输入返回值

第 2 段输 `y` 让它真删（模拟）；第 3 段直接回车接受原稿，或者输入"再短一点"看 LLM 改写。

**为什么需要 HITL**：

LLM 不是 100% 靠谱，且很多业务场景**法规强制要人工把关**：
- 删文件、转账、发邮件——错了不能撤回
- 内容审核——AI 写完得人来过一遍才能发
- 工单分派——AI 推荐了，但最终决定权在 ops

LangGraph 的 HITL 不是"让 LLM 等着"，是"让整张图在某个状态点暂停存档；等人来批，再从存档恢复继续跑"。

**给 AI 的 prompt**：

```
我刚跑了 final/02_langgraph/03_human_in_the_loop.py，看了 3 段。

请用日常类比帮我搞清 3 件事：

1. interrupt_before=["tools"] 这个参数让图"在执行 tools 之前暂停"——
   能不能比喻成"网购下单页有个'确认订单'按钮，
   付款（执行 tools）之前必须人工点一下"？
2. 暂停后，graph.get_state(config) 返回什么？
   它跟 graph.invoke 时返回的 state 字典是同一个东西吗？
   （提示：state 是数据快照，get_state 是查询快照）
3. 恢复执行有两种方式：
   - graph.invoke(None, config=config) — 直接续跑
   - graph.update_state(config, {...}) — 改了 state 再续跑
   这两个分别在什么场景用？

回答 300 字内，每问独立段落。
```

**自检**：能用一句话讲清「Checkpointer 是 HITL 的基础设施」（提示：没有 Checkpointer 就没"暂停存档"，更别说"恢复"）。

---

### 任务 2 · 用 interrupt_before 做"危险操作审批"（15 分钟）

**做什么**：在 `_scratch/my_g03_hitl.py` 里，写一个最简版本：

- 1 个工具：`mock_pay(amount: float) -> str`（模拟转账，return 成功消息）
- ReAct Agent 图（参考 `02_conditional_edges`）
- `interrupt_before=["tools"]`：每次调工具前暂停

跑一次测试：「请帮我转账 1000 元给张三」——观察暂停 → 输 y → 恢复。

**给 AI 的 prompt**：

```
我要在 _scratch/my_g03_hitl.py 里写一个最简的"转账前需要审批"的 Agent。

请引导我（不直接给代码）：
1. 工具 mock_pay 的 docstring 应该写得有多详细？
   （提示：HITL 场景下，docstring 不光给 LLM 看，
    人工审核时也要"看 LLM 想干嘛"——你怎么让人能快速读懂参数？）
2. 编译图时，interrupt_before=["tools"] 和 checkpointer=memory
   两个参数，缺一个会怎样？
3. 第一次 invoke 后，怎么从 state 拿出"LLM 打算调用的工具名 + 参数"
   给人看？（提示：messages 最后一条 AIMessage 上有 tool_calls 列表）

每次只问我一个问题。
```

写完跑：

```bash
python _scratch/my_g03_hitl.py
```

**自检**：能讲清「为什么 invoke 第二次时传 None 就能恢复」（提示：因为 thread_id 一样，Checkpointer 把上次暂停时的 state 还给图，图从中断点继续）。

---

### 任务 3 · 拒绝路径——update_state 注入"取消"（10 分钟）

**做什么**：扩任务 2，在用户输 n（拒绝）时，调 `graph.update_state(config, {...}, as_node="agent")` 注入一条 AIMessage `"操作已被用户取消"`。然后查 state，确认下一步是 END 而不是 tools。

**给 AI 的 prompt**：

```
我要给"转账审批 Agent"加拒绝路径：用户输 n 时，
不让工具执行，而是注入一条"已取消"的 AI 消息，让图自然走到 END。

请引导我（不直接给代码）：
1. update_state 的 as_node 参数是干嘛的？
   它怎么影响"图认为下一步该走哪"？
   （提示：as_node="agent" 让图觉得 agent 节点刚跑完，
    所以走 agent 后面的 conditional edge）
2. 我注入的 AIMessage 不能带 tool_calls 字段——为什么？
   （提示：tools_condition 看到 tool_calls 就会路由到 tools，违背意图）
3. 注入完之后再 graph.invoke(None, config=config)，
   这一次它会跑什么？

每次只问我一个问题。
```

实现完，分别测两次：
- 输入 y → 应该看到"转账已成功"
- 输入 n → 应该看到"操作已取消"

**自检**：能讲清「`update_state` 是 HITL 给图打补丁的口子」（提示：可以改任意 state 字段，不只是 messages）。

---

### 任务 4 · interrupt() 节点内暂停——拿用户反馈改写草稿（10 分钟）

**做什么**：参考 `final` 的 `demo_interrupt_in_node`，在 `my_g03_hitl.py` 里写一个**写作助手**：

- generate_draft：LLM 写一段 50 字介绍主题
- human_review：调 `interrupt({"draft": ..., "instruction": "请反馈"})` 暂停
- revise：拿到反馈，让 LLM 改写

跑：「请介绍 LangGraph」→ 暂停时输入"再短一点"→ 看 LLM 改写。

**给 AI 的 prompt**：

```
我要在 _scratch/my_g03_hitl.py 加一个"写作助手"图：
generate → human_review（interrupt 暂停）→ revise → END。

请引导我（不直接给代码）：
1. interrupt({...}) 在节点函数里调用，它返回什么？
   返回值什么时候才有？（提示：第二次 invoke 用 Command(resume=xx) 时）
2. interrupt_before（编译时声明）和 interrupt()（节点内调用）
   有什么本质差别？什么时候用哪个？
3. Command(resume=feedback) 的 resume 数据，怎么"喂"给 interrupt() 的返回值？
   是按节点配对的吗？

每次只问我一个问题，从日常类比开始。
```

**自检**：能讲清「`interrupt_before` 是声明式（编译期定义在哪暂停），`interrupt()` 是命令式（运行时按条件暂停）」。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_g03_hitl.py：
[贴代码]

参考答案 final/02_langgraph/03_human_in_the_loop.py：
[让 AI 自己读]

请帮我对比：
1. interrupt_before vs interrupt() 两种暂停方式我用对了吗？
2. update_state 时 as_node 设的对吗？（用错会让图走错下一步）
3. 拒绝路径有没有 graceful shutdown（不是直接 raise，而是优雅注入）？

哪些是风格差异，哪些是真问题？真问题告诉我"为什么 final 这样写更好"，
不要直接给修改后代码——让我自己改。
```

---

## 通关条件

- [ ] `python _scratch/my_g03_hitl.py` 跑通（含交互部分）
- [ ] 至少包含 1 个 `interrupt_before` 用法 + 1 个 `interrupt()` 用法
- [ ] 拒绝路径用 `update_state` 优雅处理（不是 raise / sys.exit）
- [ ] 能用一句话讲清「HITL 在 LangGraph 里 = Checkpointer + interrupt + 恢复机制」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week3-03.md`：

```markdown
# Week 3 · 03_human_in_the_loop — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → 跳 [04_multi_agent.md](04_multi_agent.md)（多 Agent 协作）
- 卡在 interrupt 用法 → 看 LangGraph 官方 cookbook 的 HITL 章节，但用 `HOW_TO_LEARN_WITH_AI.md` 第 1 条心法（拿类比问 AI）问透
- 想多练 → 把任务 4 的"一次反馈"扩成"循环反馈直到用户输 ok 才结束"——条件边 + `interrupt()` 组合
