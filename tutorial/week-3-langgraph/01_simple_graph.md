# 01 · Simple Graph — 把"流程"画成图

> **本步带回家的概念**：StateGraph = 一张有向图 + 一份共享状态字典。每个节点是个普通 Python 函数，读 state、改 state、传给下一个节点。比 AgentExecutor 那种"黑盒循环"更可控。
> **配套代码**：[`final/02_langgraph/01_simple_graph.py`](../../final/02_langgraph/01_simple_graph.py)
> **预计耗时**：45-60 分钟

---

## 准备 (5 分钟)

- [ ] Week 2 的 `01_tools_agent.md` 通关（至少能讲清 Agent 循环是什么）
- [ ] 跑过 `python final/02_langgraph/01_simple_graph.py` 一次，看到 4 段输出（含末尾 ASCII 图）
- [ ] `_scratch/my_g01_graph.py` 已建好

---

## 任务卡

### 任务 1 · 跑起来感受"图执行"（10 分钟）

**做什么**：

```bash
python final/02_langgraph/01_simple_graph.py
```

观察 4 段：
1. **单节点聊天**：State 只装 messages，一个节点把消息送 LLM 拿回答
2. **三节点流水线**：preprocess → call_llm → postprocess
3. **循环图**：increment 节点配条件边，自己跳回自己 3 次
4. **ASCII 可视化**：A → B 和 A → C 的并行图

**为什么 LangChain 已经有 LCEL，还要 LangGraph**：

LCEL 的链是**线性 + 静态**——`prompt | llm | parser` 顺序定死。但真实 Agent 经常要：
- 循环（"再调一次工具")
- 分支（"如果分类是 X 就走 A，否则 B"）
- 暂停等用户（HITL）

LangGraph 用"图 + 状态"统一这些场景，比手写 if/while 干净多了。

**给 AI 的 prompt**：

```
我刚跑了 final/02_langgraph/01_simple_graph.py。
看到 4 段：单节点 chatbot、三节点流水线、循环图、ASCII 可视化。

请用日常类比帮我搞清 3 件事：

1. StateGraph 中的 "State" 是什么角色？
   能不能比喻成"工厂流水线上的一个传送带托盘，每个工位（节点）
   都从托盘上取东西、放东西"？
2. add_messages 这个 reducer 是干嘛的？
   普通 dict 直接覆盖 vs. add_messages 追加，差别会让对话变成什么样？
3. 为什么 START 和 END 是显式画出来的，不是 Python 函数自动有的？

回答 250 字内，每问一段，不堆术语。
```

**自检**：你能用一句话讲清「为什么有了 LCEL 还需要 LangGraph」（提示：循环、分支、HITL 三大场景）。

---

### 任务 2 · 挖空写 State 和单节点（10 分钟）

**做什么**：

1. 打开 `final/02_langgraph/01_simple_graph.py`，**只看 demo_simple_chatbot 函数前 15 行**（State 定义 + 节点函数）
2. 在 `_scratch/my_g01_graph.py` 里**完全自己重写**这段（不复制 final），目标：单节点 chatbot 能跑

**给 AI 的 prompt**：

```
我要在 _scratch/my_g01_graph.py 里写一个最简 LangGraph 单节点 chatbot。

请引导我（不直接给代码）：
1. State 用 TypedDict 定义。messages 字段为什么必须用
   Annotated[list, add_messages]？如果只写 list 会怎样？
2. 节点函数的签名是什么样？输入 / 输出分别是什么？
   （提示：输入是 State，输出不是 State 而是 dict）
3. StateGraph 构建的 5 个步骤是什么？
   （add_node / add_edge / 入口 / 出口 / compile）

每次只问我一个问题。回答用日常类比开始。
```

把代码自己写出来，跑：

```bash
python _scratch/my_g01_graph.py
```

**自检**：你能讲清「节点函数为什么返回 dict 而不是新 State」（提示：返回的 dict 是"我要更新哪些字段"，框架自己 merge 进 State）。

---

### 任务 3 · 加第二个节点变流水线（10 分钟）

**做什么**：在你的 `my_g01_graph.py` 里，把单节点扩成**两节点**：

- 新增节点 `add_emoji`：拿到 LLM 回答后，前面加 `"🤖 "`，存回 messages
- 顺序：`START → chatbot → add_emoji → END`

**给 AI 的 prompt**：

```
我已经有单节点 chatbot 跑通。现在要在它后面加一个 add_emoji 节点，
功能是把 LLM 回答的最后一条消息前面拼上 "🤖 " 字符。

请引导我（不直接给代码）：
1. add_emoji 节点函数应该怎么修改 state["messages"]？
   是直接改 state 还是返回 dict？返回的 dict 该装什么？
   （提示：add_messages reducer 会追加，但你这里要"替换最后一条"——
    要怎么处理？）
2. add_edge 怎么连："chatbot" 之后必须接 "add_emoji" 吗？
3. 如果我把 add_emoji 改成"在最后一条 AI 消息后面加 emoji"
   而不是替换，行为差别在哪？

每次只问我一个问题。
```

写完跑通后，让你的 chatbot 回答两次，观察：第一次 emoji 在不在？第二次呢？

**自检**：你能讲清「add_messages reducer + 不可变 message 的组合，导致『修改已有消息』有点反直觉」（提示：不能直接 mutate，得新建 AIMessage 替换）。

---

### 任务 4 · 加循环——条件边（10 分钟）

**做什么**：参考 `final` 的 `demo_loop_graph`，在 `my_g01_graph.py` 里**单独**写一个**纯循环**示例（不涉及 LLM，避免被 LLM 输出干扰理解循环本身）：

- State 只有 `count: int`
- 节点 `bump`：count += 1，print 一行
- 条件边：count < 5 → 跳回 bump；否则 → END

**给 AI 的 prompt**：

```
我要在 _scratch/my_g01_graph.py 里加一个纯循环示例，理解条件边。
不带 LLM，只数 count 从 0 加到 5。

请引导我（不直接给代码）：
1. add_conditional_edges 三个参数（来源、条件函数、映射 dict）
   各代表什么？跟普通 add_edge 比，多了什么？
2. 条件函数返回的字符串，和映射 dict 的 key 必须严格对应吗？
   写错会怎样？（提示：可以建议我故意写错一个看看）
3. 循环图有没有"无限循环"的风险？LangGraph 怎么防？
   （提示：跟 recursion_limit 这个 invoke 参数有关）

每次只问我一个问题。
```

写完跑：

```bash
python _scratch/my_g01_graph.py
```

观察 print 5 次后退出。

**自检**：你能讲清「条件边和 if/else 的差别」（提示：if/else 是节点内部逻辑；条件边是节点之间的"路由"，更显式 + 可视化）。

---

### 任务 5 · 自检：跟 final 对比 + 看 ASCII 图（10 分钟）

**做什么**：

1. 把你的 `_scratch/my_g01_graph.py` 和 `final/02_langgraph/01_simple_graph.py` 摆一起
2. 在你的代码末尾加 `print(graph.get_graph().draw_ascii())` 画图
3. 跟 AI 对比

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_g01_graph.py：
[贴你的代码]

参考答案 final/02_langgraph/01_simple_graph.py：
[让 AI 自己读这个文件]

请帮我分析：
1. State 定义层面的差别（字段数、类型注解、reducer 用法）
2. 节点函数风格的差别（返回 dict 完整性、副作用打印等）
3. 边的拓扑差别（我有没有意外的多/少一条？画 ASCII 图能验证）
4. final 用了 4 个 demo 函数把 4 个独立图分开；
   我如果都塞一个图里会不会有问题？（提示：State schema 冲突）

哪些是风格差异，哪些是真问题？真问题告诉我"为什么 final 这样写更好"，
不要直接给修改后代码——让我自己改。
```

**自检**：改完跑通 + 你的 `draw_ascii()` 输出能正确反映你心里的图（"原来我这条边漏画了"是常见发现）。

---

## 通关条件

- [ ] `python _scratch/my_g01_graph.py` 跑通
- [ ] 至少包含一个 ≥ 2 节点的图 + 一个用条件边的循环图
- [ ] 至少跑过一次 `graph.get_graph().draw_ascii()` 看到 ASCII 图
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到 LangGraph Trace（每个节点是独立 span）
- [ ] 能用一句话讲清「StateGraph = State + Node + Edge」三件套

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week3-01.md`：

```markdown
# Week 3 · 01_simple_graph — 卡点日志

## 卡点
- 任务 X：卡了 ___ 分钟，卡在 ___

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → 跳 [02_conditional_edges.md](02_conditional_edges.md)（让 LLM 自己决定走哪条边）
- 卡在条件边那块 → 用纸笔画图：3 个节点 + 2 条边，标上 State 在每条边上的变化
- 想多练 → 把任务 3 的两节点流水线扩成三节点（preprocess → chatbot → postprocess），观察 State 在每个节点的变化
