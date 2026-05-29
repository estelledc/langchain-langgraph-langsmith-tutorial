# 01 · Tracing — 看见 Agent 内部发生了什么

> **本步带回家的概念**：LLM 应用是黑盒——你给一个问题，里面跑了 5 次 LLM 调用、3 次工具、2 次重试。LangSmith Trace = 把这盒子打开摊在桌上，每一步耗时、token、输入输出全可见。
> **配套代码**：[`final/03_langsmith/01_tracing.py`](../../final/03_langsmith/01_tracing.py)
> **预计耗时**：30-45 分钟（密度比 week-3 略低）

---

## 准备 (5 分钟)

- [ ] Week 3 全部通关
- [ ] [SETUP.md](../../SETUP.md) 提到的 LangSmith API key 已经填到 .env，且 `LANGCHAIN_TRACING_V2=true`
- [ ] 跑过 `python final/03_langsmith/01_tracing.py`，看到 4 段输出
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到刚才的 Trace
- [ ] `_scratch/my_ls01_tracing.py` 已建好

---

## 任务卡

### 任务 1 · 跑起来 + 真去 LangSmith UI 转一圈（10 分钟）

**做什么**：

```bash
python final/03_langsmith/01_tracing.py
```

跑完立刻去 [smith.langchain.com](https://smith.langchain.com)，左侧 Projects → `study`，找到刚才那批 Trace。点开任意一条，**重点看 4 件事**：
1. **执行树**：左侧的层级（chain → llm 调用 → tool 调用）
2. **Inputs / Outputs**：每个节点的输入输出（点开能看到原始 JSON）
3. **Latency / Tokens**：每步耗时和 token 用量
4. **Tags / Metadata**：你在代码里设的 `run_name="explain_concept"`、`tags=["demo"]`、`metadata={"version": "1.0"}` 在 UI 哪里显示

**为什么先看 UI**：

很多教程只讲"怎么开 trace"，但不告诉你 trace 长什么样、怎么读。这一步是让你**先建立 UI 心智模型**，再回头看代码就能问对问题。

**给 AI 的 prompt**（先去 UI 转一圈再问）：

```
我刚跑了 final/03_langsmith/01_tracing.py 并在 LangSmith UI 看了这次的 trace 树。
我注意到 LangChain 链路（PromptTemplate | llm | StrOutputParser）的每个组件
都自动出现在执行树里。

请用日常类比帮我搞清 3 件事：

1. LangSmith 的 Trace 跟程序员熟悉的 stack trace 有什么共同点 / 不同点？
   是不是更像"飞机黑匣子"——除了报错，平时也在录？
2. 自动追踪是怎么实现的？为什么我不用在 LangChain 代码里加任何东西
   就能上报？（提示：跟环境变量 + LangChain 内部 callback 有关）
3. tags 和 metadata 的差别——我应该把"环境=dev"放 tags 还是 metadata？

回答 250 字内，每问独立段落，从日常类比开始。
```

**自检**：能讲清「为什么 LangChain 自动追踪开关只是一个 env var」（提示：所有 Runnable 内部都有 callback hook，env var 一开就把上报 callback 挂上）。

---

### 任务 2 · 给普通函数加 @traceable（10 分钟）

**做什么**：在 `_scratch/my_ls01_tracing.py` 里，写两个**纯 Python 函数**（不直接调 LangChain），用 `@traceable` 装饰：

- `count_words(text: str) -> int` — 数字数
- `capitalize_first(text: str) -> str` — 把首字母大写

再写一个父函数 `process_text(text)` 调用这两个，也加 `@traceable`。跑一次后去 UI 看嵌套 trace。

**给 AI 的 prompt**：

```
我要在 _scratch/my_ls01_tracing.py 里给普通 Python 函数加 @traceable，
然后看嵌套 Trace。

请引导我（不直接给代码）：
1. @traceable 装饰器干了什么？它跟 LangChain 自动追踪是同一套机制吗？
2. 嵌套追踪是怎么"自动嵌套"的——
   父函数加了 @traceable，子函数也加了 @traceable，
   LangSmith 怎么知道子函数是父函数的孩子？
   （提示：跟 contextvar / 当前 run 的栈有关）
3. @traceable 的 name=, tags=, metadata= 参数怎么用最划算？
   什么场景必须自定义 name？

每次只问我一个问题。
```

写完跑：

```bash
python _scratch/my_ls01_tracing.py
```

去 UI 看嵌套是不是真的层级显示。

**自检**：能讲清「为什么 @traceable 要装饰子函数才能进 trace 树」（提示：不装饰就是个普通 Python 函数，框架看不到它的边界，自然不会上报）。

---

### 任务 3 · 用 RunTree 手动控制 + 加 Feedback（10 分钟）

**做什么**：参考 `final` 里 `demo_run_tree`，在 `my_ls01_tracing.py` 里写一段：

- 用 `RunTree` 手动建一个父 run
- 创建一个子 run（模拟"检索"步骤，inputs={"query": "test"}）
- 子 run end 后 patch
- 父 run end 后 patch

跑完去 UI 看你手动建的那条 trace。

**给 AI 的 prompt**：

```
我要在 _scratch/my_ls01_tracing.py 里用 RunTree 手动建 trace。

请引导我（不直接给代码）：
1. RunTree 跟 @traceable 比，多了什么自由度？什么场景必须用 RunTree？
   （提示：你想 trace 一段非 LangChain、非 Python 的代码——比如调外部 HTTP API）
2. .post() 和 .patch() 分别什么时候调？少调一个会怎样？
   （提示：post 是创建，patch 是更新——end 后必须 patch 才能把 outputs 上报）
3. 如果一个子 run 报错，应该怎么 end？
   （提示：end(error="xxx") 会把这个 run 标红）

每次只问我一个问题。
```

**进阶**：跑完拿到一个 run_id，用 `ls_client.create_feedback(run_id=..., key="rating", score=0.8)` 给它打分，去 UI 看分数显示。

**自检**：能讲清「RunTree 是给什么场景用的」（提示：非 LangChain 代码、混编系统、自定义 SDK 内部）。

---

### 任务 4 · 自检：跟 final 对比 + 看 trace 数据结构（5 分钟）

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_ls01_tracing.py：
[贴代码]

参考答案 final/03_langsmith/01_tracing.py：
[让 AI 自己读]

请帮我对比 3 处：
1. @traceable 的参数（name / tags / metadata）我有没有用得太杂或太少？
2. RunTree 的 post / patch 时机我对吗？
3. final 用 get_current_run_tree() 拿当前 run id 给 feedback —— 这个 idiom 我会用吗？

哪些是风格差异，哪些是真问题？
```

去 UI 把你的 trace 和 final 的 trace 摆一起对比"看上去能不能一目了然出执行流"。

**自检**：能讲清「trace 不只是 debug 工具，也是产品监控工具」（提示：上线后接入 dashboard、按 tag 过滤错误率、看延迟分布）。

---

## 通关条件

- [ ] `python _scratch/my_ls01_tracing.py` 跑通
- [ ] 至少 3 种追踪方式各练一次：自动追踪 LCEL 链 / `@traceable` 普通函数 / `RunTree` 手动
- [ ] 在 LangSmith UI 能找到自己跑的 trace 并看清嵌套层级
- [ ] 至少给一条 trace 加过 Feedback
- [ ] 能用一句话讲清「LangSmith Trace 解决了 LLM 应用的哪个核心痛点」（提示：黑盒可观测）

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week4-01.md`：

```markdown
# Week 4 · 01_tracing — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → 跳 [02_evaluation.md](02_evaluation.md)（如何系统评估 LLM 输出质量）
- 跑通了但还不知道 trace 实际怎么帮上忙 → 故意写一个 bug（比如 prompt 拼错占位符）跑一次，再回 UI 看 trace 找 bug
- 想多练 → 给 Week 3 的 ReAct Agent 加 metadata={"agent_version": "v1"}，对比加 vs 不加在 UI 的过滤体验
