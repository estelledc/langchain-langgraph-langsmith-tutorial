# 01 · Tools & Agent — 让 LLM 能"动手"做事

> **本步带回家的概念**：LLM 本身只会出文字。给它"工具"（函数），它就能查时间、算数学、查天气；Agent 是把"挑工具 + 调工具 + 看结果 + 再决定下一步"这套循环跑起来的指挥棒。
> **配套代码**：[`final/01_langchain/06_tools_agent.py`](../../final/01_langchain/06_tools_agent.py)
> **预计耗时**：45-60 分钟（比 week-1 单篇略长，第一次接触"循环执行"概念）

---

## 准备 (5 分钟)

- [ ] Week 1 的 5 篇都通关（至少能讲清 LCEL 和 Memory 的核心概念）
- [ ] 跑过 `python final/01_langchain/06_tools_agent.py` 一次，看完一段就行
- [ ] `_scratch/my_06_tools.py` 已建好（`touch _scratch/my_06_tools.py`）

---

## 任务卡

### 任务 1 · 跑起来感受 Agent 循环（10 分钟）

**做什么**：

```bash
python final/01_langchain/06_tools_agent.py
```

观察 `verbose=True` 打出来的执行流。每次 Agent 都会重复这个套路：

```
> Entering new AgentExecutor chain...
[LLM 选了一个工具，比如 calculate]
> Invoking: `calculate` with `{...}`
[工具的返回结果]
[LLM 看结果，决定继续调下一个工具 / 直接给最终答案]
> Finished chain.
```

特别留意第一个问题"现在几点了？北京今天天气怎么样？"——你会看到 Agent 调了 **2 个工具**（先 `get_current_time`，再 `search_weather`），然后才汇总回答。

**为什么先跑**：Agent 的核心是"循环"，光看代码看不出"循环"长什么样；跑一次看 verbose 输出，循环就具象了。

**给 AI 的 prompt**（直接复制到 CC/Cursor）：

```
我刚跑了 final/01_langchain/06_tools_agent.py，看到 verbose 输出里
LLM 自己决定调用了哪些工具、调用顺序、看到结果后再决定下一步。

请用日常类比帮我搞清两件事：

1. 没有工具的 LLM（比如 week-1 的 hello_llm）和有工具的 Agent，
   差别像"图书馆查资料的人"和"图书馆里的什么角色"？
2. AgentExecutor 的循环（Thought → Action → Observation → Thought ...），
   能不能比喻成做菜时的"看食谱 → 加料 → 尝味 → 决定下一步加什么"？

回答 200 字内，不堆术语，每问独立段落。
```

**自检**：你能用一句话讲清「为什么 LLM 不能直接知道"现在几点"」（提示：训练数据冻结时间 + 没有外部世界访问能力）。

---

### 任务 2 · 写你的第一个 @tool（10 分钟）

**做什么**：在 `_scratch/my_06_tools.py` 里，写一个工具 `count_words(text: str) -> int`，数一段文字的中文字符数（不含标点、空格）。

**给 AI 的 prompt**：

```
我要在 _scratch/my_06_tools.py 里加一个工具函数：
- 名字 count_words
- 输入：一段中文（可能含标点和空格）
- 输出：纯中文字符数

约束：
- 不要直接给我代码
- 先用列大纲方式告诉我：@tool 装饰器需要我做哪几件事？
  （提示：跟 docstring、类型注解、参数说明有关）
- 我列完大纲再让你给代码

每次只问我一个问题。
```

跟 AI 把大纲对完后再要代码，自己粘进去，加 `if __name__ == "__main__"` 打印 `count_words.invoke({"text": "你好，世界！hello"})` 的结果。

```bash
python _scratch/my_06_tools.py
```

**自检**：能讲清「`@tool` 装饰器把普通函数变成什么了」（提示：把普通 callable 包成 `BaseTool` 实例，附带名字、描述、参数 schema）+「为什么 docstring 不是给程序员看的，是给 LLM 看的」。

---

### 任务 3 · 把工具挂到 Agent 上（15 分钟）

**做什么**：在 `_scratch/my_06_tools.py` 里，给你的 `count_words` 加一个伙伴工具，再用 AgentExecutor 跑起来。

伙伴工具自选：
- 选 A：`reverse_text(text: str) -> str` 把字符串反转
- 选 B：`upper_text(text: str) -> str` 把字符串转大写

**给 AI 的 prompt**：

```
我已经写好了 count_words 和 [reverse_text / upper_text]，
现在要把它们挂到一个 AgentExecutor 上跑。

请引导我（不直接给代码）：
1. ChatPromptTemplate 必须包含哪几个 Placeholder？为什么少一个 Agent 就跑不起来？
   （提示：跟 chat_history、input、agent_scratchpad 有关）
2. agent_scratchpad 这个名字有什么意思？是给谁用的"草稿纸"？
3. AgentExecutor 的 verbose=True 和 max_iterations 这两个参数，
   各自防的是什么坑？

每次只问我一个问题。回答用日常类比开始。
```

把 prompt + agent + executor 写完，让 Agent 回答：「请帮我把"你好世界"反转之后告诉我反转后的字符数」——它应该先调 reverse_text，再调 count_words。

```bash
python _scratch/my_06_tools.py
```

**自检**：能讲清 `agent_scratchpad` 是什么（提示：Agent 把"调过哪些工具、看到了什么"存在这里，下一轮 LLM 才能基于这个推理下一步）。

---

### 任务 4 · 报错练习——故意给坏描述（10 分钟）

**做什么**：现在你的 `count_words` docstring 应该写得很清楚。**故意把它改差**：

- 选 1：把 docstring 整段删掉
- 选 2：把 docstring 改成 `"数数"`（极简）
- 选 3：参数 `text` 的描述删掉，只留参数名

跑同一个问题：「请帮我数一下"你好世界"是几个字」，看 Agent 的反应。

可能出现两种情况：
- Agent 还是猜对了工具（说明 LLM 鲁棒性强）
- Agent 调错了 / 不调工具直接瞎答 / 报错

**给 AI 的 prompt**：

```
我把 count_words 的 docstring 故意改成了 [说出你改的内容]，
然后跑同一个问题，结果 Agent [描述你看到的现象]。

我以为会发生 [你猜的现象]，实际看到 [实际现象]。

请用 2-3 句话讲清：
1. LLM 选工具的时候，主要看什么？是函数名、docstring、还是参数 schema？
2. 工程上，docstring 写得好对 Agent 的成功率影响有多大？
   （能不能给个直觉量级，比如"差和好之间差 X%"）

不要长篇大论，每问一段。
```

**自检**：你能讲清「为什么 LangChain 工具的 docstring 是写给 LLM 看的，不是给开发者看的」+「下次写工具时，docstring 应该包含哪 3 件事」。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**做什么**：把你的 `_scratch/my_06_tools.py` 和 `final/01_langchain/06_tools_agent.py` 摆一起。

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_06_tools.py：
[贴你的代码]

参考答案 final/01_langchain/06_tools_agent.py：
[让 AI 自己读这个文件，或贴关键片段]

请帮我分析 4 个层次：
1. 工具数量和工具复杂度差别（final 用了 5 个工具+ StructuredTool，我用了几个？）
2. ChatPromptTemplate 的 messages 顺序和占位符，我有没有少 / 多写？
3. AgentExecutor 的参数（verbose、max_iterations 等），我和 final 的差别会带来什么后果？
4. 哪些是"风格差异"，哪些是"真问题"？

真问题告诉我"为什么这样写更好"，但不要直接给修改后代码——让我自己改。
```

按 AI 指出的"真问题"自己改一遍。

**自检**：改完跑通 + 能讲清「StructuredTool 跟 @tool 装饰器的差别和适用场景」（提示：参数复杂、需要明确类型校验时用 StructuredTool；简单参数 @tool 更省事）。

---

## 通关条件

- [ ] `python _scratch/my_06_tools.py` 能跑通
- [ ] 至少包含 2 个 `@tool` 装饰的函数 + 1 个 AgentExecutor
- [ ] Agent 能正确处理 ≥ 1 个需要"调多个工具串起来"的问题
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到 Agent Trace（含工具调用子节点）
- [ ] 能用一句话讲清「Agent 循环 = LLM 选工具 → 调工具 → 看结果 → LLM 再决策」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week2-01.md`：

```markdown
# Week 2 · 01_tools_agent — 卡点日志

## 卡点
- 任务 X：卡了 ___ 分钟，卡在 ___

## "原来如此"时刻
- AI 哪句话让我突然懂了？

## 想留作复用的 prompt
[贴 1-2 个最有效的 prompt]

## 还没搞懂的（留尾巴）
- ___
```

---

## 通往下一站

- 全部通关 → 跳 [02_structured_output.md](02_structured_output.md)（让 LLM 返回 Pydantic 对象，生产 90% 场景必用）
- 卡在某个任务 → 发 AI："给我一个再小一号的练习"，让它把任务再拆细
- 想多练 → 给 Agent 加第三个工具（比如 `random_choice(items: list[str]) -> str`），看它能不能正确选用
