# 概念小词典 — 零基础选手最容易卡的 18 个术语

> 看官方文档卡住时，多半是被某个术语挡住——文档默认你懂。这页用日常类比把这些术语先消化，再回去看文档就顺了。
>
> 每条结构：**类比** → **技术定义** → **为什么需要它** → **延伸阅读**。

---

## 元概念（你听得最多的几个）

### 1. LLM（Large Language Model，大语言模型）

**类比**：一个"读过几乎所有公开文字"的高能补全器。给它前半句，它猜后半句。

**技术定义**：基于 Transformer 架构的神经网络，参数量 10亿+，训练在海量文本上做"下一个 token 预测"。

**为什么需要它**：传统程序需要硬编码规则——"如果用户问 X 就回 Y"。LLM 不需要规则，它"自己感觉应该说什么"，省了 99% 的规则工作量。代价：不可控、有幻觉。

**延伸**：[01_hello_llm.md](../tutorial/week-1-langchain/01_hello_llm.md)

---

### 2. Token（令牌）

**类比**：LLM 看世界的"最小单位"。中文一个字 ≈ 1-2 token；英文一个常用词 ≈ 1 token，复杂词 2-4 token。

**技术定义**：经过 BPE / WordPiece 等分词后产生的整数序列。LLM 的输入输出都是 token，不是字符。

**为什么要管它**：API 按 token 计费；模型有"上下文窗口"长度限制（比如 8K / 32K token），超过会报错或截断。

**实用知识**：

```python
# 估算 token 数（粗略：中文 ≈ 1.5 token / 字，英文 ≈ 0.75 token / 词）
import tiktoken
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
len(enc.encode("hello world"))  # 2
```

---

### 3. Prompt（提示词）

**类比**：你跟 LLM 说话时**整段输入**——包含你想让它做什么的全部上下文。

**技术定义**：传给 LLM 的字符串（或消息列表）。在 ChatGPT 的对话框里你打的每段字都是 prompt。

**关键点**：写 prompt 是个工程问题——同样的目标，prompt 不同，结果天差地别。

**延伸**：[02_prompt_template.md](../tutorial/week-1-langchain/02_prompt_template.md)；[prompts-cheatsheet.md](prompts-cheatsheet.md) 是真实写法。

---

### 4. System / Human / AI Message

**类比**：一个微信群有 3 种角色——

- **System**：群规（你跟 LLM 设定的"扮演什么角色"）
- **Human**：用户在群里说的话
- **AI**：LLM 回的话

**技术定义**：OpenAI / Anthropic 的 chat 接口把 prompt 拆成有"角色"的消息列表，模型对不同角色权重不同（System 通常被强约束）。

**为什么需要分**：让 LLM 区分"指令"vs"内容"。你要 LLM 当"严谨学者"——这放 System；你问问题——放 Human。

---

## LangChain 五大件

### 5. ChatPromptTemplate

**类比**：邮件模板。`亲爱的 {name}，您 {date} 的订单 {order_id} 已发货` 这种带占位符的字符串。

**技术定义**：把"prompt 字符串 + 占位符变量"分开管理的对象，调用 `.invoke({变量字典})` 渲染成实际 prompt。

**为什么需要**：避免在代码里写 `f"亲爱的 {name}..."` 这种字符串拼接——变量改了好几处都要找；模板抽出来，改一处全用。

**延伸**：[02_prompt_template.md](../tutorial/week-1-langchain/02_prompt_template.md)

---

### 6. LCEL（LangChain Expression Language）

**类比**：Unix 管道（`cat foo.txt | grep error | wc -l`）。前一个的输出是后一个的输入，用 `|` 串。

**技术定义**：LangChain 的运算符重载，让 `prompt | llm | parser` 这样的表达式构建一条可执行链。每个组件都实现了 `Runnable` 接口。

**为什么需要**：

```python
# ❌ 没 LCEL：手写每步调用，难维护
filled = prompt.invoke({"x": "y"})
response = llm.invoke(filled)
text = parser.invoke(response)

# ✅ LCEL：声明式
chain = prompt | llm | parser
text = chain.invoke({"x": "y"})
```

**延伸**：[03_chains.md](../tutorial/week-1-langchain/03_chains.md)

---

### 7. Memory（对话记忆）

**类比**：客服知道你之前说过"我叫小明"，下次回答能直接称呼"小明"——而不是每次都问"请问您是？"。

**技术定义**：把历史 messages 存起来，每次调 LLM 时一起塞进去，让 LLM 看到完整对话史。LangChain 的 `RunnableWithMessageHistory` 帮你管这件事。

**为什么需要**：LLM 默认是无状态的，每次调用是独立请求。要做多轮对话必须自己管 history。

**延伸**：[04_memory.md](../tutorial/week-1-langchain/04_memory.md)

---

### 8. Tool（工具）

**类比**：AI 助手的"外挂"。LLM 自己只会出文字，加了 calculate 工具就能算数学，加了 search 就能查网。

**技术定义**：用 `@tool` 装饰的 Python 函数，附带 docstring（描述）和参数 schema。LLM 通过"function calling"机制选择调用。

**为什么需要**：让 LLM 突破"只会出文字"的能力边界，能查实时信息、改外部状态。

**关键点**：docstring 是给 LLM 看的——LLM 选工具靠读它。写不好工具会被错调或不调。

**延伸**：[01_tools_agent.md](../tutorial/week-2-tools-and-agent/01_tools_agent.md)

---

## RAG 相关 4 个

### 9. Embedding（向量嵌入）

**类比**：把每个词 / 句子翻译成 768 维（或更多）的"坐标"。语义相近的句子在坐标空间里也相邻。

**技术定义**：把文本输入 embedding 模型（比如 `text-embedding-3-small` / `text-embedding-v3`），得到一个浮点数 list（向量）。

**为什么需要**：

- 关键字搜索：要求字面匹配（"猫"找不到"喵星人"）
- 向量搜索：按语义相似找（"猫"能找到"喵星人"）

---

### 10. Vector DB（向量数据库）

**类比**：专门存"坐标 + 原文"的高速搜索引擎。给一个"查询坐标"，返回"附近的几个坐标对应的原文"。

**技术定义**：支持向量近邻搜索（KNN / ANN）的存储系统。常见：FAISS（本地）、Chroma（轻量）、Pinecone（云）、Milvus（分布式）。

**为什么需要**：embedding 后的向量要持久化 + 快速搜索，普通数据库做不了高效近邻搜索。

---

### 11. RAG（Retrieval-Augmented Generation）

**类比**：开卷考试。学生（LLM）答题前先翻参考书（向量库）找相关页面，再基于找到的内容答题——不靠纯记忆。

**技术定义**：4 步骤——
1. **切片**：把文档切成小块（chunk）
2. **向量化**：每个 chunk 算 embedding，存进 vector db
3. **检索**：用户问问题时，先把问题 embedding，查 db 找相似的几个 chunk
4. **增强生成**：把 chunk 拼到 prompt 里给 LLM 答

**为什么需要**：

- LLM 训练数据有截止日期，不知道你的私有知识
- LLM 有"幻觉"——给它真实参考能减少瞎编

**延伸**：[05_rag_basic.md](../tutorial/week-1-langchain/05_rag_basic.md)

---

### 12. Chunk（切片）/ chunk_size

**类比**：把整本书撕成 N 页，每页 500 字。搜索时按页召回，不召回整本书。

**技术定义**：把长文档按字符 / token 数切成小段，每段叫 chunk。`chunk_size` 是每段大小。

**为什么需要切**：

- 太小（< 100 字符）：上下文不完整，LLM 回答片段化
- 太大（> 2000 字符）：检索时召回的"相关段落"不够精确，浪费 token

**经验值**：500-1500 字符 / 100-500 token 是多数场景的甜区。

---

## Agent / LangGraph 相关 3 个

### 13. Agent（智能体）

**类比**：项目经理 + 专家团。PM 接需求 → 决定该叫谁 → 专家干活 → 收回结果 → 决定下一步。

**技术定义**：能"自主决定调用哪些工具"的 LLM 系统。最简单的形式：LLM + 一组 Tool + 一个循环（让 LLM 看结果再决定下一步）。

**为什么需要**：单次 LLM 调用解决不了复杂任务（"帮我查 3 个城市天气并对比"）。Agent 把"决策"权交给 LLM，框架负责执行。

**延伸**：[01_tools_agent.md](../tutorial/week-2-tools-and-agent/01_tools_agent.md)

---

### 14. ReAct（Reasoning + Acting）

**类比**：边想边做。"我先查 A → 看结果想想 → 再查 B → 看结果再想 → ... → 答案 OK 退出"。

**技术定义**：Agent 的一种主流执行模式：LLM 输出 Thought（推理）+ Action（调工具）→ 看 Observation（工具结果）→ 再 Thought / Action ... 直到决定输出最终答案。

**为什么需要**：和 Plan-and-Execute（先全规划再执行）相比，ReAct 更灵活——计划不一定一次对，每步看结果再调整更鲁棒。

**延伸**：[02_conditional_edges.md](../tutorial/week-3-langgraph/02_conditional_edges.md)

---

### 15. StateGraph（状态图）

**类比**：流程图（带向的箭头从一个节点到另一个）+ 共享数据字典（每个节点都能读写）。

**技术定义**：LangGraph 核心抽象。State 是共享数据；Node 是 Python 函数（读 state、返回要更新的字段）；Edge 是节点间的连接（普通边或条件边）。

**为什么需要**：LCEL 的链是线性 + 静态。真实 Agent 要循环、分支、HITL（人工审核）——StateGraph 用图统一这些场景。

**延伸**：[01_simple_graph.md](../tutorial/week-3-langgraph/01_simple_graph.md)

---

## LangSmith 相关 3 个

### 16. Trace（追踪）

**类比**：飞机黑匣子。飞行（LLM 调用）的每一步——什么时间、什么参数、用了多少油（token）、收到什么应答——全录下来。出问题查录像。

**技术定义**：每次 LLM 调用 / chain 执行的完整执行树记录，含 inputs / outputs / latency / tokens / errors。LangSmith 自动收集 + UI 可视化。

**为什么需要**：LLM 应用是黑盒——一个用户请求触发 5 次 LLM、3 次工具。哪步耗时多、哪步答错——光看代码看不出，靠 trace。

**延伸**：[01_tracing.md](../tutorial/week-4-langsmith-and-project/01_tracing.md)

---

### 17. Evaluator（评估器）

**类比**：考试阅卷标准。一份卷子可以有多个评分维度——字数、关键词、理解准确度——每个维度独立打分。

**技术定义**：函数 `(run, example) -> {"key": str, "score": float, "comment": str}`。LangSmith 的 `evaluate()` 跑批时调每个 evaluator 给每条样本打分。

**为什么需要**：LLM 输出对错没硬标准，靠多维评分组合判断。常见三类：

- **结构类**（长度、关键词覆盖、JSON 合法）
- **语义类**（LLM-as-Judge）
- **业务类**（自定义业务规则）

**延伸**：[02_evaluation.md](../tutorial/week-4-langsmith-and-project/02_evaluation.md)

---

### 18. LLM-as-Judge（LLM 当裁判）

**类比**：让另一个老师（独立 LLM）批改第一个老师（被评估 LLM）的卷子。

**技术定义**：自定义 evaluator 中调用一个 LLM 作为评分器。给它（问题 + 参考答案 + 预测答案），让它输出 0-1 分数 + 理由。

**为什么需要**：

- 关键词匹配等"硬"评估太死（"喵星人"和"猫"是同义但匹配不上）
- LLM-as-Judge 抓语义等价性

**注意**：

- LLM-as-Judge 自己也会错——配合人工抽查
- 自洽性差——同一题让 judge 跑 3 遍取众数能稳

**延伸**：[02_evaluation.md](../tutorial/week-4-langsmith-and-project/02_evaluation.md)

---

## 怎么用这个词典

学习时遇到不懂的术语：

1. 先来这页查——找到就读"类比 → 定义 → 为什么需要"
2. 找不到 → 用 [prompts-cheatsheet.md 模板 1.1](prompts-cheatsheet.md#模板-11--跟已知概念类比最常用) 问 AI
3. 问透了的概念，回来这页提个 PR 加进来——这是仓库越用越好的来源

---

## 版本

- v1（2026-05）：初版 18 个核心术语
