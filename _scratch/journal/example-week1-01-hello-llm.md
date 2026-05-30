# Week 1 · 01_hello_llm — 卡点日志（example）

> 这是**真实卡点日志的样例**——记录作者第一次走 `tutorial/week-1-langchain/01_hello_llm.md` 时遇到的卡点和顿悟时刻。
>
> 你写自己的日志时，文件名用日期：`2026-XX-XX-week1-01.md`。
> 这份 example 不会被 git ignore 是为了让 fork 者能看见——你自己写的（其他 `journal/*.md`）会自动 ignored。

---

## 卡点

### 任务 2：挖空写 LLM 实例化（卡了 18 分钟）

**卡在哪**：复制 final 那 3 行代码到 `_scratch/my_01_hello.py` 后，自己跑就报：

```
ModuleNotFoundError: No module named 'final'
```

我以为 `from final._common import make_llm` 这种 import 自动会 work，因为 IDE 没标红。

跟 AI 对话时一开始就贴 traceback 让它修——AI 直接给我加了一段 `sys.path.insert(...)`，但我不知道为什么要加。

**重启对话后改用心法 3（"我以为 vs 实际"）**重新问，AI 这次先问我"你跑代码时 cwd 是哪？"——我才发现：

- 我在 `_scratch/` 目录下跑 `python my_01_hello.py`
- final 那 3 行在 `final/01_langchain/01_hello_llm.py` 里能用，是因为它有 `sys.path.insert(0, parent.parent.parent)` 这一行（我没复制）
- 我的 `_scratch/my_01_hello.py` 没那行，且 cwd 在 `_scratch/`，Python 找不到 `final` 包

**修法**：要么也加那行 sys.path，要么去仓库根跑：

```bash
# 仓库根目录
python _scratch/my_01_hello.py
```

我选了第二种——干净不加 boilerplate。

---

### 任务 4：报错练习（卡了 5 分钟）

**故意改坏**：把 `messages = [HumanMessage(content=...)]` 改成 `messages = "1+1=?"`。

**我以为**：报错"必须传消息列表"。

**实际**：

```
ValidationError: ... messages
  Input should be a valid list
```

跟我猜的方向对，但具体类型来自 pydantic 的 `ValidationError`——这才意识到 LangChain 用 pydantic 做参数校验。**这个"原来如此"比直接背"messages 必须是 list" 印象深得多**。

---

## "原来如此"时刻

1. **`from final._common import make_llm` 不是魔法——是 Python 的包路径解析**。我之前以为 import 在哪都行，IDE 不报错就能跑。其实 IDE 是看着 `__init__.py` + 工程根猜的，运行时只看 cwd + sys.path。

2. **`AIMessage.response_metadata['token_usage']` 这种字段很有用**。我跑 final 时 print 了，看到一次 invoke 用了 ~50 tokens——立刻意识到 token 不是抽象概念，是真要算钱的（哪怕通义千问免费层）。

3. **`stream` 不是"省时间"，是"改用户体验"**。流式输出耗时跟 invoke 几乎一样，但用户看到字一个个蹦出来感知"AI 在思考"。这跟 ChatGPT 的实现完全一致。

---

## 想留作复用的 prompt

**最有效的 prompt（任务 4 报错诊断）**：

```
我跑 _scratch/my_01_hello.py 时把 messages 改成字符串 "1+1=?"，
我以为会看到"必须传消息列表"，结果报 ValidationError。

请用 2-3 句话讲清：
1. ValidationError 是哪个库抛的？
2. 为什么 LangChain 选 pydantic 而不是手写 isinstance 判断？

不要堆术语，回答 200 字内。
```

为什么有效：

- "我以为 vs 实际" 让 AI 知道我心智模型在哪
- "哪个库抛的" 把追问从现象推到底层
- "为什么不用 isinstance" 让 AI 解释设计选择，不只是描述行为
- 200 字限制让 AI 不胡扯长

---

## 还没搞懂的（留尾巴，下周回头）

1. **System / Human / AI 三种 Message 在底层 HTTP 请求里长什么样**？
   按 OpenAI 协议应该是 `{"role": "system" | "user" | "assistant", "content": "..."}` 的列表，但还没自己抓包看。
   待办：用 `httpx.HTTPTransport` 拦截一次请求看真 body。

2. **`temperature=0` 真的能保证完全确定输出吗**？
   理论上 sampling 关了应该是 deterministic，但实际跑两次 invoke 同 prompt，回答有时还是不同（标点 / 字数差几个字）。是 LLM 服务端有别的随机源？
   待办：开 `temperature=0` 跑同 prompt 10 次对比 → 找答案。

---

## 通关情况

- [x] `python _scratch/my_01_hello.py` 跑通
- [x] 包含 `demo_simple()` + `demo_stream_simple()` 两个函数
- [x] LangSmith 看到 ≥ 2 条新 Trace
- [x] 能讲清"为什么把消息分 System/Human/AI 三种角色"
  → System = 群规 / Human = 用户提问 / AI = 模型回答；分角色让 LLM 训练时学会区分"指令"和"内容"，受 OpenAI 微调的 RLHF 数据格式影响

**总耗时**：~50 分钟（含跟 AI 对话 + 写日志）。比 tutorial 标的 30-45 分钟略多，因为任务 2 卡了 18 分钟。

---

## 下一步

- 跳 `02_prompt_template.md`
- 把"留尾巴 1（HTTP 抓包）" 加到本周末复盘的待办里
