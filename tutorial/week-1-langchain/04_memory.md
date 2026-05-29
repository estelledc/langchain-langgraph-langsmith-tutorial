# 04 · Memory — 让多轮对话"记住"上一轮说的话

> **本步带回家的概念**：多轮对话要"记忆"，本质是把历史消息每次都塞进 prompt。LangChain 的 `RunnableWithMessageHistory` 帮你管这件事，配合 `session_id` 实现多用户隔离。
> **配套代码**：[`final/01_langchain/04_memory.py`](../../final/01_langchain/04_memory.py)
> **预计耗时**：30-40 分钟

---

## 准备 (5 分钟)

- [ ] 已经走完 [03_chains.md](03_chains.md)
- [ ] `python final/01_langchain/04_memory.py` 能输出 3 段（手动历史 / RunnableWithMessageHistory / 滑动窗口）
- [ ] 打开 Claude Code 或 Cursor，cd 到本仓库根
- [ ] 在 `_scratch/` 新建空文件 `my_04_memory.py`（`touch _scratch/my_04_memory.py`）

> **会看到的 warning（不是 bug）**：跑 final 时终端可能跳 `LangChainDeprecationWarning: RunnableWithMessageHistory is deprecated. Use LangGraph's built-in persistence instead.`。这是 langchain 1.x 把状态管理推荐路径迁到 langgraph 了，但 RunnableWithMessageHistory 仍然能用——你 week-3 学 LangGraph 时会看到"新方式"。先按本篇学就行，**别一上来就跳到 langgraph**。

---

## 任务卡

### 任务 1 · 跑起来观察"AI 真的记住了"（5 分钟）

**做什么**：

```bash
python final/01_langchain/04_memory.py
```

观察 3 段输出。重点看第 2 段【RunnableWithMessageHistory】：

- Alice 第 1 句："我叫 Alice，我是数据科学家"
- Alice 第 2 句："我最近在研究大语言模型"
- Alice 第 3 句："你还记得我的名字和职业吗？" → AI 答出 `Alice / 数据科学家`
- 然后 Bob 进来问"你知道 Alice 是谁吗？" → AI 答**不知道**（session 隔离）

**为什么先跑**：记忆是抽象概念，先看到"AI 真的复述了 3 句话之前你说的名字"，再回头问"它怎么记住的"。

**给 AI 的 prompt**：

```
我刚跑了 final/01_langchain/04_memory.py，看到 AI 在第 3 句对话里
准确复述了我前面说的"Alice / 数据科学家"。

请用日常类比帮我建立直觉：
1. LLM 本身是"无状态"的（每次调用都是空白脑子），
   它怎么"记住"我前面说的话？
   （提示：不是 LLM 真记住，是有人帮它"每次把历史塞进 prompt"）
2. RunnableWithMessageHistory 这个东西，
   像不像"会议秘书"——
   每次开会前把上次纪要打印好放你桌上，会后把新纪要塞回档案？
3. session_id 像不像"客户编号"——
   秘书按编号分文件夹，Alice 的纪要绝不会跑到 Bob 的桌上？

每个问题 2 句话以内，回答完不要主动加新内容。
```

**自检**：你能用一句话讲「为什么"LLM 记住对话"本质上是个假象」（提示：实际是历史每次都重传给 LLM）。

---

### 任务 2 · 反例：先写一个"没有记忆"的两轮对话（10 分钟）

**做什么**：在 `_scratch/my_04_memory.py` 写一段最小代码，**不加任何 memory**，让 LLM 答两轮：

- 第 1 轮：「我叫小明」
- 第 2 轮：「我叫什么名字？」

预期：LLM 在第 2 轮答不出（因为它根本没看到第 1 轮）。

**给 AI 的 prompt**：

```
我要在 _scratch/my_04_memory.py 写一段"反例代码"——故意不加 memory。
目标：让 LLM 在两次独立 invoke 里，证明它"失忆"。

约束：
- 不要直接给我代码
- 先列大纲（这段最小代码大概 5-10 行，干 4 件事，分别是哪 4 件？）
- 每次 invoke 都用一个全新的 messages 列表，不要保留任何历史
- 等我说"OK 给代码"才给

回答用日常类比解释"为什么这样调用 LLM 一定会失忆"。
```

跟 AI 把大纲对完后让它给代码，自己粘进 `my_04_memory.py`，函数命名 `demo_no_memory()`，跑：

```bash
python _scratch/my_04_memory.py
```

观察 LLM 第 2 轮答不出"小明"——可能答"我不知道你叫什么"或随便编一个。

**自检**：你能讲清「为什么把第 1 轮和第 2 轮都用同一个全新 messages = [...] 调用，第 2 轮就一定失忆」（提示：每次 LLM 看到的输入只有当前那一轮）。

---

### 任务 3 · 加上 memory，让 AI 真的记住（10 分钟）

**做什么**：参考 final 里的 `demo_runnable_with_history`，在 `my_04_memory.py` 加一个函数 `demo_with_memory()`，用 `RunnableWithMessageHistory` 包装链，让任务 2 的两轮对话变成"记住名字"。

**给 AI 的 prompt**：

```
我在 my_04_memory.py 已经有 demo_no_memory()（任务 2 的反例）。
现在要加一个 demo_with_memory()，让两轮对话变成"AI 记住名字"。

参考 final/01_langchain/04_memory.py 的 demo_runnable_with_history。

请引导我理解，不要一次性给完整代码：

1. 先用日常类比解释——
   ChatMessageHistory + RunnableWithMessageHistory 这两个东西
   像不像"中间人帮你保管聊天记录 + 自动把记录塞进 prompt"？
   能不能比喻成"会议秘书"职业？
2. 这套机制需要 4 件东西：
   - 一个 store（dict）
   - 一个 get_session_history(session_id) 函数
   - prompt 里要预留 MessagesPlaceholder 槽位
   - chain_with_history = RunnableWithMessageHistory(...) 包装
   每件东西的作用是什么？
3. config = {"configurable": {"session_id": "xxx"}} 这个 session_id
   为什么不能省略？

回答前 2 个问题等我消化，再让我贴大纲让你审。
```

跟 AI 对完，自己写代码，main 里调起来：

```python
if __name__ == "__main__":
    print("=== 反例：无 memory ===")
    demo_no_memory()
    print("\n=== 有 memory ===")
    demo_with_memory()
```

观察"有 memory"那段，LLM 第 2 轮真的答出"小明"。

**自检**：能跑通 + 能讲清「`MessagesPlaceholder(variable_name="history")` 在 prompt 里这个空位，被 RunnableWithMessageHistory 填进去的是什么数据」（提示：是 `[HumanMessage("我叫小明"), AIMessage("...")]` 这样的列表）。

---

### 任务 4 · 测 session_id 隔离（5 分钟）

**做什么**：在 `demo_with_memory()` 里加两个 session（`alice` 和 `bob`），让它们各自跟 LLM 自我介绍，然后**互相打听**——验证 LLM 在 bob 的会话里**完全不知道** alice。

**给 AI 的 prompt**：

```
我在 demo_with_memory() 里现在只用了一个 session_id。
我想加一个对照实验：

- session_id = "alice" 的会话：alice 说"我叫 Alice，是医生"
- session_id = "bob" 的会话：bob 问"你知道 Alice 是谁吗？"

预期：bob 的会话里 LLM 答"不知道"——因为 store[alice] 和 store[bob] 是两个独立的 ChatMessageHistory。

请用日常类比解释 session_id 在工程上的真实价值：
1. 在多用户产品里（比如客服机器人同时服务 1000 个用户），
   为什么必须按 session_id 区分？不区分会发生什么灾难场景？
2. 生产环境里 store 真的会用 dict 吗？还是会换成什么？
   （提示：dict 在进程重启就丢了，多机部署也不共享）

每个问题 2 句话以内。
```

写完跑起来，观察输出：alice 那行 AI 答"是医生"，bob 那行 AI 答"不知道"。

**自检**：能讲清「为什么 store 在多用户 / 多机部署的生产环境必须换成 Redis 之类的外部存储」。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**做什么**：把你的 `_scratch/my_04_memory.py` 和 `final/01_langchain/04_memory.py` 摆一起。

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_04_memory.py：
[贴你的代码]

参考答案 final/01_langchain/04_memory.py：
（让 AI 直接读这个文件）

请帮我分析：
1. 哪里我写得不一样？
2. 不一样的地方，哪些是无所谓的"风格差异"
   （变量名、prompt 文本、main 顺序）？
3. 哪些是真的会影响结果的"问题"？
   （比如：MessagesPlaceholder 漏了、input_messages_key 配错、
   session_id 没传、store 写在函数外丢了状态）

如果有问题，告诉我"为什么这样写更好"——
但不要直接给我修改后的代码，让我自己改。

额外问题：final 里还演示了"滑动窗口历史"（demo_window_history），
只保留最近 N 轮。这个特性在什么场景下用？
（提示：长对话 token 超限）
```

按 AI 指出的"真问题"自己改一遍。

**自检**：改完跑通 + 能讲清「滑动窗口历史」用在什么场景（提示：跟"对话越久 token 越多越费钱"有关）。

---

## 通关条件

- [ ] `python _scratch/my_04_memory.py` 能跑通
- [ ] 至少包含 `demo_no_memory()`（反例）+ `demo_with_memory()`（用 RunnableWithMessageHistory）两个函数
- [ ] 能用代码对比"失忆"和"记得"两版的差别
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到本次新增的 ≥ 2 条 Trace，且能在 Trace 的 prompt 里看到历史消息真的被塞进去了
- [ ] 能讲清「history store 在多用户场景为什么必须按 session_id 区分」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week1-04.md`：

```markdown
# Week 1 · 04_memory — 卡点日志

## 卡点
- 任务 X：卡了 ___ 分钟，卡在 ___
- ...

## "原来如此"时刻
- AI 哪句话让我突然懂了？

## 想留作复用的 prompt
[贴 1-2 个最有效的 prompt]

## 还没搞懂的（留尾巴）
- ___
```

---

## 通往下一站

- 全部通关 → 跳 [05_rag_basic.md](05_rag_basic.md)
- 卡在某个任务 → 发 AI："给我一个再小一号的练习"，让它把任务再拆细
- 想实操"滑动窗口"或"摘要历史"→ 直接问 AI："final 的 demo_window_history 我也想吃透，先让我自己写一个 max_messages=4 的版本，然后引导我理解为什么是乘 2"
