# 05 · RAG Basic — 让 LLM 引用你的私有文档

> **本步带回家的概念**：RAG（Retrieval-Augmented Generation，检索增强生成）= 把外部文档**切片 → 向量化 → 检索相关片段 → 喂给 LLM 当上下文**。解决"LLM 不知道你公司私有数据"的问题。
> **配套代码**：[`final/01_langchain/05_rag_basic.py`](../../final/01_langchain/05_rag_basic.py)
> **预计耗时**：40-50 分钟（最长，因为概念多）

---

## 准备 (5 分钟)

- [ ] 已经走完 [04_memory.md](04_memory.md)
- [ ] `python final/01_langchain/05_rag_basic.py` 能跑通——会先打印「向量索引构建完成」，再打印 3 段 RAG 问答 + 1 段相似度搜索
- [ ] 看到的输出里 LLM 真的回答了 LangGraph / LangSmith / RAG 的具体内容（这些信息只在 final 内置文档里，不是 LLM 通用知识）
- [ ] 打开 Claude Code 或 Cursor，cd 到本仓库根
- [ ] 在 `_scratch/` 新建空文件 `my_05_rag.py`（`touch _scratch/my_05_rag.py`）

> **跑不通常见原因**：如果终端报 `openai.PermissionDeniedError: Error code: 403 ... text-embedding-v3`，说明你的 DashScope key 没开通 embedding 模型权限。回 [DashScope 控制台](https://dashscope.aliyuncs.com) → 模型广场 → 开通"通用文本向量"。这一步**只这篇用得到**，前面 04 篇都不需要。

---

## 任务卡

### 任务 1 · 跑起来观察"AI 真的在引用文档"（5 分钟）

**做什么**：

```bash
python final/01_langchain/05_rag_basic.py
```

观察输出。重点看：

- 启动时打印「原始文档：5 篇，切分后：N 个块」——文档被切成了 N 份小卡片
- 【基础 RAG 链】3 个问题：LLM 答出来的内容（如「LangGraph 的核心特性：StateGraph / 条件边 / Checkpointer / Human-in-the-loop」）**完全来自 final 内置文档**，不是 LLM 训练数据里的通用知识
- 【向量相似度搜索】最后一段：直接显示了"L2 距离"分数，分数小的就是检索到的最相关文档

**为什么先跑**：RAG 这 4 步（切片 / 向量化 / 检索 / 生成）抽象，先看到"AI 答出来的就是 final 文档里的原话"，再回头问"它怎么找到这段的"。

**给 AI 的 prompt**：

```
我刚跑了 final/01_langchain/05_rag_basic.py，看到 AI 在回答
"LangGraph 有哪些核心特性"时，准确说出了
"StateGraph / 条件边 / Checkpointer / Human-in-the-loop"——
这 4 个词正好是 final 内置文档里的原话。

请用日常类比帮我建立直觉：
1. RAG 解决的是什么问题？为什么 ChatGPT 通用知识答不出
   "我们公司内部 wiki 写了什么"？
   （提示：跟"训练数据截止时间 / 私有数据没进训练集"有关）
2. RAG 的工作过程像不像"找资料写报告"？
   - 把书撕成卡片（切片）
   - 给每张卡片贴主题标签（向量化）
   - 翻卡片找跟问题最相关的几张（检索）
   - 把卡片贴在草稿纸上一起交给 ChatGPT 写报告（生成）
3. 这 4 步里，哪一步是"AI 真的智能"，哪一步是"纯粹的工程拼装"？

每个问题 2 句话以内。
```

**自检**：你能用一句话讲「RAG 让 LLM 看起来"知道"私有数据，但本质不是 LLM 真学会了，而是每次问都把相关文档塞进 prompt」。

---

### 任务 2 · 拆 RAG 4 步骤（10 分钟）

**做什么**：不写代码，跟 AI 对话拆透 final 里 RAG 的 4 步对应到哪几行代码。

**给 AI 的 prompt**：

```
请帮我把 final/01_langchain/05_rag_basic.py 里的 4 个 RAG 步骤
对应到具体的代码段——不要直接全部讲完，每讲一步等我说"OK 下一个"再继续。

1. 切片（splitting）
   - 用的是哪个类？参数是什么意思？
   - chunk_size=300 / chunk_overlap=50 / separators=["\n\n", "\n", "。", ...]
     这 3 个参数各管什么？为什么要"重叠"？
   - 用"读书做笔记"类比解释：为什么要切，不能整本书塞 LLM？

2. 向量化（embedding）
   - 用的是 OpenAIEmbeddings(model="text-embedding-v3", base_url=DASHSCOPE_BASE_URL)
   - "向量"是什么？凭什么意思相近的两段文字向量也相近？
   - 用"图书馆图书分类"类比：每本书有个杜威分类号，
     主题相近的书号也相近——向量是不是这种"自动生成的分类号"？

3. 检索（retrieval）
   - vectorstore.as_retriever(search_kwargs={"k": 2}) 里的 k 是什么意思？
   - retriever.invoke(question) 返回什么数据形态？
   - 用"翻卡片"类比：从一摞卡片里挑出 k 张跟问题最相关的，怎么做到的？

4. 增强生成（augmented generation）
   - 看 demo_basic_rag 里的 rag_chain，
     {"context": retriever | format_docs, "input": RunnablePassthrough()} 这段干嘛？
   - prompt 里的 {context} 占位符被填的是什么？

每一步讲完等我说"OK 下一个"。
```

跟 AI 一步步对完，能用自己的话各自复述。

**自检**：你能把 4 步分别用一个动作概括（撕卡片 / 贴分类号 / 翻卡片 / 拼草稿），且能指出 final 里每一步对应的 1-2 行代码。

---

### 任务 3 · 改 chunk_size 体感切片直觉（15 分钟）

**做什么**：复制 `final/01_langchain/05_rag_basic.py` 到 `_scratch/my_05_rag.py`，**只改 splitter 的 chunk_size**，跑两次：

- 第 1 次：`chunk_size=50`（极小）
- 第 2 次：`chunk_size=1500`（极大）

每次都跑 `demo_basic_rag` 的 3 个问题，对比答案质量。

**给 AI 的 prompt**：

```
我把 final/01_langchain/05_rag_basic.py 复制到 _scratch/my_05_rag.py，
准备改 splitter 的 chunk_size 跑实验：
- 默认 chunk_size=300
- 实验 1：chunk_size=50
- 实验 2：chunk_size=1500

请用"读书做笔记"类比帮我预判结果，等我跑完再对照：

1. chunk_size=50 太小会出什么问题？
   （提示：一张卡片只能写半句话，回答时拿到的"相关卡片"信息太碎）
2. chunk_size=1500 太大会出什么问题？
   （提示：一张卡片塞太多内容，相关的、不相关的都混在一起，
   LLM 注意力分散；而且一次塞进 prompt 的 token 暴涨）
3. chunk_overlap=50 这个"重叠"是干嘛的？
   （提示：避免在句子中间硬切断关键信息）

请只讲预判，不要直接告诉我跑出来会怎样。等我跑完贴结果再帮我分析。
```

跑两次实验，把"启动时的切分块数"和"3 个问题的回答"分别记下来（贴到当日 journal 也行）。然后让 AI 对照预判和实测。

**自检**：你能讲清「chunk_size 不是越小越好也不是越大越好」（提示：太小=碎片化丢失上下文，太大=注意力分散+token 浪费），且能说出你的实验里哪个 chunk_size 表现最差、为什么。

---

### 任务 4 · 思考：什么场景该用 RAG / 什么场景不该用（10 分钟）

**做什么**：这一步不写代码，做"框架性思考"——RAG 不是万能的，理解它的边界比实现它更重要。

**给 AI 的 prompt**：

```
我刚学完 RAG 的实现。请帮我建立"什么时候用 RAG / 什么时候不用"的直觉。

请列：
- 3 个适合 RAG 的场景，每个 1 句话说明为什么适合
- 3 个不适合 RAG 的场景，每个 1 句话说明为什么不适合

参考方向（不要照抄，自己想例子）：
- 适合的方向：知识更新频繁、私有数据、需要可追溯来源、
  数据量超出 context window
- 不适合的方向：需要跨多文档推理、需要数学/逻辑严密计算、
  数据量很小（直接塞 prompt 更简单）、需要实时数据

列完后请反问我：
- 我所在的实习项目里，有哪些场景"看起来像 RAG"，
  但其实换成"调用一个查询接口 / 直接把文档塞 prompt"会更简单？

我会自己想答案再回复你。
```

跟 AI 对完后，自己写下「我能想到的一个适合 RAG 的真实场景」+「一个看起来像但其实不该用 RAG 的场景」到当日 journal。

**自检**：你能给出至少一个具体例子，说明"先想清楚是不是 RAG 问题，再写 RAG 代码"为什么重要。

---

## 通关条件

- [ ] 跑通 `final/01_langchain/05_rag_basic.py`
- [ ] 自己改了 `chunk_size=50` 和 `chunk_size=1500` 各跑一次，**能用自己的话对比说出差别**
- [ ] [smith.langchain.com](https://smith.langchain.com) 的 `study` 项目下能看到本次新增的 Trace，且能在 Trace 里看到 retriever 的独立步骤（不只是 LLM 调用）
- [ ] 能讲清「RAG 适合什么 / 不适合什么」，至少各 2 个例子

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week1-05.md`：

```markdown
# Week 1 · 05_rag_basic — 卡点日志

## 卡点
- 任务 X：卡了 ___ 分钟，卡在 ___
- ...

## chunk_size 实验记录
- chunk_size=50：切分后 ___ 块，回答质量 ___
- chunk_size=300（默认）：切分后 ___ 块，回答质量 ___
- chunk_size=1500：切分后 ___ 块，回答质量 ___
- 我的结论：___

## "原来如此"时刻
- AI 哪句话让我突然懂了？

## RAG 适用场景思考
- 我能想到的一个真实"适合 RAG"的场景：___
- 我能想到的一个真实"不该用 RAG"的场景：___

## 还没搞懂的（留尾巴）
- ___
```

---

## 通往下一站

恭喜走完 week-1！这周的 5 篇教程覆盖了 LangChain 最核心的 5 个概念：

- 01 · LLM 调用 — 给文本收文本，三种角色
- 02 · Prompt 模板 — 模板和变量分开
- 03 · LCEL 链 — `|` 把组件拼成流水线
- 04 · Memory — 多轮对话的"假装记忆"
- 05 · RAG — 让 LLM 引用私有文档

接下来：

- **整理本周卡点日志**：把 5 篇 `_scratch/journal/2026-XX-XX-week1-0X.md` 翻一遍，
  圈出 3 个"我反复卡住"的点 + 3 个"原来如此"瞬间。
  这是你 week-1 真正的学习产出。
- **week-2 在另一个计划里**：会进 `tutorial/week-2-tools-and-agent/`，
  开始学 Tools + Agent，让 LLM 不只是聊天，还能真的"做事"。
- 卡在某个任务 → 发 AI："给我一个再小一号的练习"，让它把任务再拆细
- 想再深入 RAG → 直接问 AI："final 里 demo_rag_with_sources 用 RunnablePassthrough.assign 同时返回答案和来源，这个写法的语义我没完全吃透，引导我看懂"
