# 01 · Hello LLM — 跟 AI 一起喊出第一句话

> **本步带回家的概念**：LLM 调用 = 给文本 → 收文本，但中间有「角色」区分（System / Human / AI）。
> **配套代码**：[`final/01_langchain/01_hello_llm.py`](../../final/01_langchain/01_hello_llm.py)
> **预计耗时**：30-45 分钟

---

## 准备 (5 分钟)

- [ ] [SETUP.md](../../SETUP.md) 跑通了，`python final/01_langchain/01_hello_llm.py` 能输出
- [ ] [HOW_TO_LEARN_WITH_AI.md](../../HOW_TO_LEARN_WITH_AI.md) 至少扫过一遍
- [ ] 打开 Claude Code 或 Cursor，cd 到本仓库根
- [ ] 在 `_scratch/` 新建空文件 `my_01_hello.py`（`touch _scratch/my_01_hello.py`）

---

## 任务卡

### 任务 1 · 跑起来再说（5 分钟）

**做什么**：

```bash
python final/01_langchain/01_hello_llm.py
```

观察终端输出：三段（invoke / stream / batch），每段都有 LLM 的真实回答。

**为什么先跑**：你不需要先理解代码再跑——先看到「它能干什么」，再回头问「它怎么干的」。这是用 AI 学陌生代码的第一招。

**给 AI 的 prompt**（直接复制到 CC/Cursor，不要改）：

```
我刚跑了 final/01_langchain/01_hello_llm.py，看到三段输出：invoke、stream、batch。
请用日常类比帮我讲清楚两个问题：

1. ChatOpenAI 这一类对象，本质是什么角色？
   （提示：是不是像微信里的"小爱同学"接口？）
2. SystemMessage / HumanMessage / AIMessage 三种消息的关系，
   能不能比喻成微信群里的不同角色？

要求：
- 回答控制在 200 字内
- 不要堆术语
- 回答完不要主动加新内容，等我下一个问题
```

**自检**：如果 AI 用了「客服窗口 / 接线员 / 角色扮演」类的比喻，且你能用一句话复述「这三种 Message 分别像谁说话」，✓。

---

### 任务 2 · 挖空写 LLM 实例化（5 分钟）

**做什么**：

1. 打开 `final/01_langchain/01_hello_llm.py`，**只看创建 llm 的那段**（大概 19-21 行）
2. 把那段复制到 `_scratch/my_01_hello.py`
3. **关掉 final 文件**，别再偷看

**给 AI 的 prompt**：

```
我在 _scratch/my_01_hello.py 里只复制了 LLM 实例化那段代码：

[把你刚复制的那几行贴这里]

请引导我理解，不要直接给答案：
1. make_llm 这个工厂函数，背后大概帮我做了什么？
   （提示：可以让我猜它内部做了什么拼装）
2. 如果我直接用 ChatOpenAI(...) 自己构造，需要传哪些参数才能连上 DashScope？

每次只问我一个问题，等我回答再继续。
```

**自检**：你能说清「为什么 _common.py 把构造 LLM 这件事单独抽出来」（提示：跟"模型 ID 改一处全仓库生效"有关）。

---

### 任务 3 · 自己写第一个 invoke 函数（10 分钟）

**做什么**：不要看 final，跟 AI 对话写一个函数 `demo_simple()`，让 LLM 回答「1+1=?」。

**给 AI 的 prompt**：

```
我要在 _scratch/my_01_hello.py 里加一个函数 demo_simple()，
让 LLM 回答 "1+1=?" 这个问题，最后 print 出回答。

约束：
- 不要直接给我代码
- 先列大纲（这个函数大概有几步？每步干嘛？）
- 等我说"OK 给代码"才给

回答用日常类比解释步骤，每个步骤一句话。
```

跟 AI 把大纲对完后再让它给代码，自己粘进 `_scratch/my_01_hello.py` 的最下面，加 `if __name__ == "__main__": demo_simple()`，然后跑：

```bash
python _scratch/my_01_hello.py
```

**自检**：能跑通 + 能用自己的话讲「这个函数 4 步分别是干嘛」。

---

### 任务 4 · 报错练习——故意搞砸再修（5 分钟）

**做什么**：在 `_scratch/my_01_hello.py` 里**故意改坏一处**：

- 选 1：把 `HumanMessage` 改成 `humanmessage`
- 选 2：把 `messages = [...]` 改成 `messages = "1+1=?"`（直接传字符串）
- 选 3：删掉 `temperature=0.7` 这行（这个其实不会报错，留着看为什么）

跑起来，观察报错。

**给 AI 的 prompt**：

```
我故意改坏了 _scratch/my_01_hello.py 的 [说出你改的位置]，
我以为会发生 [你猜的现象]，结果报错：

[贴报错最后 5 行]

请别直接修——给我 3 个候选原因让我猜哪个最可能。
回答简洁，每个候选 1 句话。
```

修回来后，把那一处的「正确写法」和「错误写法」并排在 `_scratch/journal/2026-XX-XX-week1-01.md` 里。

**自检**：你能讲清「为什么 LangChain 强制用 HumanMessage 而不是字符串」。

---

### 任务 5 · 加 stream 流式输出（5 分钟）

**做什么**：参考 `final` 里的 `demo_stream`，在 `my_01_hello.py` 加一个流式输出版本 `demo_stream_simple()`。

**给 AI 的 prompt**：

```
final/01_langchain/01_hello_llm.py 里有 demo_stream 函数，
用了 llm.stream() 和 chunk.content。

请帮我理解 3 个问题：
1. 流式输出在用户体验上有什么差别？为什么 ChatGPT 是逐字蹦出来？
2. for chunk in llm.stream(messages) 这个循环，每次循环 chunk 里装的是什么？
3. 如果我不写 flush=True，会怎样？

回答简洁，每个问题 2 句话以内。
```

理解完，自己照葫芦画瓢，在 `my_01_hello.py` 加 `demo_stream_simple()`，main 里调起来。跑起来观察"逐字蹦"的体感。

**自检**：你能讲清「为什么 stream 模式适合长回答 / 不适合 batch 调用」。

---

### 任务 6 · 自检：跟 final 对比（5 分钟）

**做什么**：把你的 `_scratch/my_01_hello.py` 和 `final/01_langchain/01_hello_llm.py` 摆一起。

**给 AI 的 prompt**：

```
我自己写的 _scratch/my_01_hello.py：
[贴你的代码]

参考答案 final/01_langchain/01_hello_llm.py：
[贴 final 代码，或者让 AI 自己读这个文件]

请帮我分析：
1. 哪里我写得不一样？
2. 不一样的地方，哪些是无所谓的"风格差异"（变量名、顺序、注释多寡）？
3. 哪些是真的会影响结果的"问题"？

如果有问题，告诉我"为什么这样写更好"——
但不要直接给我修改后的代码，让我自己改。
```

按 AI 指出的"真问题"自己改一遍 `my_01_hello.py`。

**自检**：改完跑通 + 能讲清你的版本和 final 的真正差别（去掉风格差异后还剩多少）。

---

## 通关条件

- [ ] `python _scratch/my_01_hello.py` 能跑通
- [ ] 至少包含 `demo_simple()`（invoke 调用）和 `demo_stream_simple()`（stream 调用）两个函数
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到 ≥ 2 条新 Trace
- [ ] 能用一句话讲清「为什么 LangChain 把消息分成 System / Human / AI 三种角色」

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week1-01.md`：

```markdown
# Week 1 · 01_hello_llm — 卡点日志

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

- 全部通关 → 跳 [02_prompt_template.md](02_prompt_template.md)
- 卡在某个任务 → 发 AI："给我一个再小一号的练习"，让它把任务再拆细
- 全跑完但感觉没学到东西 → 回头看 HOW_TO_LEARN_WITH_AI.md 第 7 条心法，用「自己讲一遍」自查
