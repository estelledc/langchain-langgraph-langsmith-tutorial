# 03 · Chains — 用 `|` 把组件串成流水线

> **本步带回家的概念**：LCEL（LangChain Expression Language）让你用 `|` 把组件串成链；每个组件都实现了同一个接口（Runnable），所以可以自由拼。
> **配套代码**：[`final/01_langchain/03_chains.py`](../../final/01_langchain/03_chains.py)
> **预计耗时**：30-40 分钟

---

## 准备 (5 分钟)

- [ ] 已经走完 [02_prompt_template.md](02_prompt_template.md)
- [ ] `python final/01_langchain/03_chains.py` 能输出 6 段（基础链 / JSON / 并行 / 透传 / Lambda / 组合）
- [ ] 打开 Claude Code 或 Cursor，cd 到本仓库根
- [ ] 在 `_scratch/` 新建空文件 `my_03_chain.py`（`touch _scratch/my_03_chain.py`）

> **耗时提醒**：这个文件实测跑完 ≈ 5-7 分钟（6 个 demo 串行调 LLM，公网网络下更快但仍有 3 分钟级别）。看到终端长时间停在某段输出**不要以为卡死**，等就好。

---

## 任务卡

### 任务 1 · 跑起来观察 6 段输出（5 分钟）

**做什么**：

```bash
python final/01_langchain/03_chains.py
```

观察 6 段输出。重点看 3 段：

- 【基础 LCEL 链】：`prompt | llm | parser` 三段串联的最小形式
- 【RunnableParallel — 并行链】：同一个输入，两条链并行跑，结果合成 dict（`pros` / `cons`）
- 【RunnableLambda — 自定义函数】：在链尾加一个普通 Python 函数（`word_count`）

**为什么先跑**：链是抽象概念，先看到"输入 → 输出"的真实数据形态，再回头问"中间发生了什么"。

**给 AI 的 prompt**：

```
我刚跑了 final/01_langchain/03_chains.py，看到 6 段输出。

请用日常类比帮我建立直觉，不用术语堆砌：
1. 基础链 prompt | llm | StrOutputParser() 像不像
   工厂里的 3 段流水线（贴标签 → 加工 → 装箱）？
2. RunnableParallel(pros=链A, cons=链B) 像不像
   一份订单同时发给两个班组，最后合并成 dict 报告？
3. RunnableLambda 是把"普通 Python 函数"塞进链里——
   像不像在流水线尾巴临时加一道"人工质检"工序？

每个类比 2 句话以内，回答完不要主动加新内容。
```

**自检**：你能用 1 句话各自复述这 3 个类比，且能答出「为什么 final 里 6 段都用 `|` 串，而不是写成嵌套函数调用」。

---

### 任务 2 · 拆链：理解 `|` 和 Runnable 接口（5 分钟）

**做什么**：不写代码，纯对话理解 `|` 的本质。

**给 AI 的 prompt**：

```
我想搞懂 LCEL 的 `|` 操作符。请引导我，不要一次性讲完，每次只问一个问题：

1. unix 命令 `cat file | grep word | wc -l` 的 `|` 是什么含义？
   （提示：上一步的输出当下一步的输入）
2. LCEL 的 `prompt | llm | StrOutputParser()` 跟 unix 管道的相同点是什么？
   不同点是什么？（提示：unix 传字节流，LCEL 传什么？）
3. 链里"每段必须实现什么接口"才能被 `|` 串起来？
   （提示：跟 Java 的 interface / Python 的鸭子类型有关，
   LangChain 里这个接口叫 Runnable，要求实现 invoke / stream / batch）
4. 如果我想把一个普通函数 `def upper(s): return s.upper()` 也塞进链里，
   不能直接 `prompt | llm | upper`，得用什么包一下？
   （提示：RunnableLambda）

每次问完等我回答再继续。
```

**自检**：你能用一句话讲「Runnable 接口为什么是 LCEL 能拼起来的关键」（提示：所有部件长成一个样，所以可以乐高式拼接）。

---

### 任务 3 · 挖空 + 自己拼一条 ≥ 4 段链（10 分钟）

**做什么**：在 `_scratch/my_03_chain.py` 里拼一条 ≥ 4 段的链：

```
prompt → llm → StrOutputParser → RunnableLambda(自定义函数)
```

自定义函数要求：把 LLM 回答整段转大写（英文）或加一个前缀（中文）。例：`f"【AI 回答】{text}"`。

**给 AI 的 prompt**：

```
我要在 _scratch/my_03_chain.py 写一条 4 段链：
prompt → llm → StrOutputParser → RunnableLambda(我的函数)

我的函数：把 LLM 回答前面加一个 "【AI 回答】" 前缀。

请引导我思考，不要直接给完整代码，每次问一个问题：

1. 自定义函数应该用 RunnableLambda(my_func) 包，
   还是直接 `prompt | llm | StrOutputParser() | my_func`？
   （提示：直接拼会报错——为什么？跟 Runnable 接口有关）
2. 链的顺序里，哪些可以颠倒？哪些不能？
   - StrOutputParser 和 RunnableLambda 能颠倒吗？
   - llm 和 prompt 能颠倒吗？
3. 如果我的函数想接收"原始用户输入 + LLM 输出"两个东西，
   普通的 RunnableLambda(func) 行不行？需要换成什么？
   （提示：RunnablePassthrough.assign 或者 RunnableParallel）

回答完前 2 个问题，再让我贴大纲让你审。
```

跟 AI 对完后，自己写代码，main 里调一次。跑：

```bash
python _scratch/my_03_chain.py
```

**自检**：能跑通 + 能回答「`StrOutputParser` 和 `RunnableLambda` 顺序为什么不能颠倒」（提示：parser 把 LLM 的 AIMessage 对象转 str，Lambda 期望的是 str，颠倒就传错类型了）。

---

### 任务 4 · 玩 RunnableParallel：一个输入跑两条链（5 分钟）

**做什么**：参考 final 里的 `demo_parallel`，在 `my_03_chain.py` 加一个函数 `demo_my_parallel()`：同一个 `{thing}` 输入，并行跑「3 个优点」和「3 个缺点」两条链，输出 `{"pros": ..., "cons": ...}`。

**给 AI 的 prompt**：

```
我要在 _scratch/my_03_chain.py 加一个函数 demo_my_parallel()。
要求：
- 用 RunnableParallel(pros=pros_chain, cons=cons_chain) 把两条独立链合并
- 输入一个 {"thing": "远程办公"}，输出 dict（含 pros / cons 两个 key）

请引导我思考，不要直接给代码：
1. 为什么 RunnableParallel 能减少总耗时？
   （提示：两条链各自一次 LLM 调用，串行 2*N 秒，并行 N 秒）
2. RunnableParallel 输出的 dict 的 key 怎么决定的？
3. 如果 pros_chain 和 cons_chain 接受的输入字段名不一样
   （比如一个用 {thing}，一个用 {item}），还能塞进同一个 RunnableParallel 吗？

每个问题 2 句话内。回答完前 2 个再让我贴大纲让你审。
```

写完跑起来，观察输出 dict 的两段。

**自检**：能讲清「RunnableParallel 的两条链共享什么、各自独立什么」（共享：输入；独立：prompt / llm 调用 / 输出）。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**做什么**：把你的 `_scratch/my_03_chain.py` 和 `final/01_langchain/03_chains.py` 摆一起。

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_03_chain.py：
[贴你的代码]

参考答案 final/01_langchain/03_chains.py：
（让 AI 直接读这个文件）

请帮我分析：
1. 哪里我写得不一样？
2. 不一样的地方，哪些是无所谓的"风格差异"
   （变量名、prompt 文本、main 调用顺序）？
3. 哪些是真的会影响结果的"问题"？
   （比如：RunnableLambda 没包、parser 漏了、链顺序颠倒）

如果有问题，告诉我"为什么这样写更好"——
但不要直接给我修改后的代码，让我自己改。

额外问题：final 还演示了 JsonOutputParser、RunnablePassthrough、链组合，
我没写。这 3 个里你建议我下一步先吃透哪个？为什么？
```

按 AI 指出的"真问题"自己改一遍。

**自检**：改完跑通 + 能讲清「为什么 LCEL 比手写嵌套调用 `parser(llm.invoke(prompt.format(...)))` 好维护」（提示：跟"流式 / 调试 / 替换组件" 有关）。

---

## 通关条件

- [ ] `python _scratch/my_03_chain.py` 能跑通
- [ ] 自己写的链 ≥ 4 段（任务 3）
- [ ] 至少包含一个 `RunnableLambda`（任务 3）和一个 `RunnableParallel`（任务 4）
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到本次新增的 ≥ 2 条 Trace，且能在 Trace 的"Steps"里看到链路的每一段
- [ ] 能用一句话讲清「为什么 LCEL 比手写嵌套调用好维护」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week1-03.md`：

```markdown
# Week 1 · 03_chains — 卡点日志

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

- 全部通关 → 跳 [04_memory.md](04_memory.md)
- 卡在某个任务 → 发 AI："给我一个再小一号的练习"，让它把任务再拆细
- 对 JsonOutputParser / 链组合还有兴趣 → 直接问 AI："final 里的 demo_json_output / demo_chain_composition 我也想吃透，先解释 demo_json_output"
