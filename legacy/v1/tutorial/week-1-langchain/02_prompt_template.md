# 02 · Prompt Template — 把"模板"和"变量"分开

> **本步带回家的概念**：Prompt 模板把「模板字符串 + 变量」分开，方便复用；ChatPromptTemplate 在此基础上加了角色（System / Human / AI）区分。
> **配套代码**：[`final/01_langchain/02_prompt_template.py`](../../final/01_langchain/02_prompt_template.py)
> **预计耗时**：30-40 分钟

---

## 准备 (5 分钟)

- [ ] 已经走完 [01_hello_llm.md](01_hello_llm.md)，能跑通 `_scratch/my_01_hello.py`
- [ ] `python final/01_langchain/02_prompt_template.py` 能输出 4 段（PromptTemplate / ChatPromptTemplate / FewShot / Partial）
- [ ] 打开 Claude Code 或 Cursor，cd 到本仓库根
- [ ] 在 `_scratch/` 新建空文件 `my_02_prompt.py`（`touch _scratch/my_02_prompt.py`）

---

## 任务卡

### 任务 1 · 跑起来观察 4 段输出（5 分钟）

**做什么**：

```bash
python final/01_langchain/02_prompt_template.py
```

观察终端输出的 4 段：`PromptTemplate` / `ChatPromptTemplate` / `FewShotChatMessagePromptTemplate` / `Partial`。每段都先打印「渲染后的 Prompt」再打印「LLM 回答」。

**为什么先跑**：4 个概念光看名字记不住，先看到 4 段输出长什么样，再回头问"它们各自像什么"。这是用类比建立直觉的第一步。

**给 AI 的 prompt**（直接复制到 CC/Cursor，不要改）：

```
我刚跑了 final/01_langchain/02_prompt_template.py，看到 4 段输出：
- PromptTemplate（纯字符串模板）
- ChatPromptTemplate（带 system/human 角色）
- FewShotChatMessagePromptTemplate（带示例对的模板）
- Partial（预先填一半变量）

请用日常类比帮我把这 4 个概念分别讲清楚——不用术语堆砌，每个一句话：
1. PromptTemplate 像不像"邮件模板"（占位符 {姓名}）？
2. ChatPromptTemplate 像不像"会议纪要模板"（主持人/参会人各自说话）？
3. FewShot 像不像"考试做题前先看的几道例题"？
4. Partial 像不像"邮件签名预先填好，只留正文动态写"？

每个类比 1 句话，回答完不要主动加新内容，等我下一个问题。
```

**自检**：你能用一句话各自复述这 4 个类比，且能讲清「为什么不用 f-string 直接拼，要专门搞 PromptTemplate」（提示：跟"复用 / 测试 / 在 LangSmith 看到模板原型"有关）。

---

### 任务 2 · 挖空写 PromptTemplate（10 分钟）

**做什么**：

1. 打开 `final/01_langchain/02_prompt_template.py`，**只看 `demo_prompt_template` 这个函数**
2. 把里面 4 行（`template = PromptTemplate.from_template(...)` + `prompt_str = template.format(...)` + `chain = template | llm | StrOutputParser()` + `result = chain.invoke({...})`）复制到 `_scratch/my_02_prompt.py`
3. **关掉 final 文件**，别再偷看
4. 在文件顶补上必要的 import 和 `llm = make_llm(...)`（参考你 01 那个文件）

**给 AI 的 prompt**：

```
我在 _scratch/my_02_prompt.py 里只复制了 PromptTemplate 这一段：

[把你刚复制的 4 行贴这里]

请引导我理解，不要直接给答案，每次只问我一个问题，等我回答再继续：

1. template.format(language="Python", n=5) 这个 format 方法是干嘛的？
   它返回的是 PromptTemplate 还是 str？
2. LCEL 的 `|` 符号是什么意思？
   能不能用 unix 管道（cat file | grep word | wc -l）来类比？
3. 如果我把 `| StrOutputParser()` 这一段删掉，chain.invoke() 返回的会是什么？
   （提示：跟 LLM 原始返回对象 vs 纯字符串有关）
```

跟 AI 把 3 个问题对完后，自己加 `if __name__ == "__main__": demo_prompt_template()` 跑一遍。

**自检**：你能讲清「`|` 操作符把 3 个东西串在一起的本质是什么」（提示：每个东西都实现了同一个接口）。

---

### 任务 3 · 自己写 ChatPromptTemplate（10 分钟）

**做什么**：不要看 final，跟 AI 对话写一个函数 `demo_my_chat()`，参数化 `domain` 和 `concept`，让 LLM 当某领域专家解释某概念。

**给 AI 的 prompt**：

```
我要在 _scratch/my_02_prompt.py 里加一个函数 demo_my_chat()。
要求：
- 用 ChatPromptTemplate.from_messages 写一个两段式模板
- system 段：让 LLM 当 {domain} 领域专家
- human 段：请解释 {concept}
- 组成 chain（prompt | llm | StrOutputParser）
- main 里调两次：一次问"机器学习"+"过拟合"，一次问"烹饪"+"美拉德反应"

约束：
- 不要直接给我代码
- 先列大纲（这个函数大概有几步？每步干嘛？）
- 等我说"OK 给代码"才给

回答用日常类比解释 system / human 各自像什么。
```

跟 AI 把大纲对完后再让它给代码，自己粘进 `_scratch/my_02_prompt.py`，跑：

```bash
python _scratch/my_02_prompt.py
```

**自检**：能跑通 + 能讲清「为什么 ChatPromptTemplate 要分 system/human，不能像 PromptTemplate 那样塞一坨字符串」（提示：跟 LLM 训练时的对话格式有关）。

---

### 任务 4 · 玩 Partial 理解"预先固定一半"（5 分钟）

**做什么**：在 `my_02_prompt.py` 里加一个函数 `demo_my_partial()`，沿用任务 3 的模板，但用 `template.partial(domain="Python")` 把 domain 固定成 "Python"，然后循环问 3 个 Python 概念（自己挑，比如「装饰器」「上下文管理器」「列表推导式」）。

**给 AI 的 prompt**：

```
我现在要把任务 3 的 ChatPromptTemplate 用 partial 预先固定一半变量。

请用"邮件签名"类比帮我理解：
- 没用 partial 之前，每次调用都要传完整的 {domain, concept}
  （像每写一封邮件都要从头写"敬礼... 此致... 署名"）
- 用了 partial(domain="Python") 之后，只需要传 {concept}
  （像把签名档预先填好，只写正文）

请回答：
1. 这种"预先固定一半"在工程上有什么实用价值？
   （提示：跟"同一个团队成员复用同一个角色设定"有关）
2. partial 返回的还是 ChatPromptTemplate 吗？还是别的对象？
3. 如果我 partial(domain="Python") 之后又 invoke 时传 {"domain": "Java", "concept": "..."}，会发生什么？

每个问题 2 句话以内。
```

跟 AI 对完，自己写代码跑起来，观察 3 次回答都是 Python 视角的。

**自检**：能讲清「partial 在多用户 / 多场景下的复用价值」（举一个真实场景：比如不同租户共享同一套 prompt 但各自有自己的"领域"前缀）。

---

### 任务 5 · 自检：跟 final 对比（5 分钟）

**做什么**：把你的 `_scratch/my_02_prompt.py` 和 `final/01_langchain/02_prompt_template.py` 摆一起。

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_02_prompt.py：
[贴你的代码]

参考答案 final/01_langchain/02_prompt_template.py：
（让 AI 直接读这个文件）

请帮我分析：
1. 哪里我写得不一样？
2. 不一样的地方，哪些是无所谓的"风格差异"（变量名、顺序、注释多寡、prompt 文本不同）？
3. 哪些是真的会影响结果的"问题"（比如：忘了 StrOutputParser？partial 用错？变量名拼错）？

如果有问题，告诉我"为什么这样写更好"——
但不要直接给我修改后的代码，让我自己改。
```

按 AI 指出的"真问题"自己改一遍 `my_02_prompt.py`。

**自检**：改完跑通 + 能讲清「自己版本和 final 在 prompt 设计上的真正差异」（不是变量名差异，是结构差异）。

---

## 通关条件

- [ ] `python _scratch/my_02_prompt.py` 能跑通
- [ ] 至少包含 `demo_prompt_template`（任务 2）+ `demo_my_chat`（任务 3）+ `demo_my_partial`（任务 4）三个函数
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到本次新增的 ≥ 3 条 Trace
- [ ] 能用一句话讲清「为什么用 LCEL 的 `|` 而不是嵌套调用 `parser(llm.invoke(prompt.format(...)))`」（提示：跟"统一接口 / 流式 / 调试" 有关）
- [ ] 至少跑过一次带 `partial` 的链

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week1-02.md`：

```markdown
# Week 1 · 02_prompt_template — 卡点日志

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

- 全部通关 → 跳 [03_chains.md](03_chains.md)
- 卡在某个任务 → 发 AI："给我一个再小一号的练习"，让它把任务再拆细
- 对 FewShot 还想多了解（任务里没专门覆盖）→ 直接问 AI："final 里 demo_few_shot 的 examples 列表为什么放 3 个不放 1 个？放越多越好吗？"
