# 03 · Streaming & Resilience — 流式 + 重试 + 容错

> **本步带回家的概念**：生产里 LLM 应用上线必撞两件事——用户等 30 秒不耐烦（**流式**解决） + LLM 偶尔 timeout / 503（**retry / fallback** 解决）。这一篇把 week-1 学的基础 stream 升级，再加生产可靠性两件套。
>
> **配套代码**：本篇没有 final/.py（补充篇）
> **预计耗时**：35-50 分钟
> **官方文档对应**：[Streaming](https://python.langchain.com/docs/how_to/streaming/) + [How to add retry to LCEL](https://python.langchain.com/docs/how_to/runnable_runtime_secrets/)，但 zero 版补了"什么时候用"的判断和挖空写

---

## 准备 (5 分钟)

- [ ] [02_structured_output.md](02_structured_output.md) 通关
- [ ] [01_hello_llm.md](../week-1-langchain/01_hello_llm.md) 的 stream 用法还记得（基础 chunk-by-chunk）
- [ ] `_scratch/my_w2_03_resilience.py` 已建好

---

## 任务卡

### 任务 1 · 三种 stream API 的差别（10 分钟）

LangChain 有 3 个流式 API，零基础选手最容易混：

| API | 流什么 | 用在哪 |
|---|---|---|
| `llm.stream(messages)` | 单次 LLM 调用的 token chunk | week-1/01 那种最简单场景 |
| `chain.stream(input)` | LCEL 链最终输出的 chunk | LCEL 链场景 |
| `chain.astream_events(input, version="v2")` | 链中**每个**节点的事件流 | 复杂 chain / Agent，要"中间步骤可见" |

**做什么**：在 `_scratch/my_w2_03_resilience.py` 里用同一个 chain 跑三种方式，亲眼看差别。

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from final._common import make_llm

llm = make_llm(temperature=0.5)
chain = (
    ChatPromptTemplate.from_template("用 100 字介绍 {topic}，并列 3 个常见误解")
    | llm
    | StrOutputParser()
)
```

跑 3 种：

```python
# 方式 1：chain.stream
print("=== chain.stream ===")
for chunk in chain.stream({"topic": "Python GIL"}):
    print(chunk, end="", flush=True)
print()

# 方式 2：chain.astream_events（async）
import asyncio

async def run_events():
    print("\n=== astream_events v2 ===")
    async for event in chain.astream_events({"topic": "Python GIL"}, version="v2"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif kind in ("on_chain_start", "on_chain_end"):
            print(f"\n[{kind}] name={event.get('name')}")
    print()

asyncio.run(run_events())
```

**给 AI 的 prompt**：

```
我跑了 chain.stream 和 chain.astream_events 两种方式。

请用日常类比帮我搞清 3 件事（不直接给代码）：

1. chain.stream 流的是"链的最终输出"——
   能不能比喻成"水龙头出来的水"？
2. astream_events 流的是"每个节点的事件"——
   是不是"工厂流水线上的实时摄像头"，看每个工位什么时候开始/结束？
3. 我什么时候用 stream 什么时候用 astream_events？
   （提示：用户要看到流式回答 vs 调试 / 监控用）

每问 80 字内，从日常类比开始。
```

**自检**：能讲清「stream = 最终输出流；astream_events = 中间过程流」。

---

### 任务 2 · 流 LangGraph 节点的中间状态（10 分钟）

**做什么**：复用 week-3/02 的 ReAct Agent 思路（但简化），写一个 2 节点图：

- agent_node：让 LLM 决定用哪个工具或直接答
- tools_node：执行工具

跑 `graph.stream()` 和 `graph.astream_events()`，对比哪个更适合"实时给用户看 Agent 在干嘛"。

> 这一题如果你还没学到 week-3/02，可以**先跳过**——回头学完再来。

**给 AI 的 prompt**：

```
我有一个 LangGraph 的 ReAct Agent（agent → tools → agent → ... → END）。
要给用户实时看 Agent 调了哪个工具、收到什么结果。

请引导我（不直接给代码）：
1. graph.stream() 的默认 stream_mode 流什么？是节点输出还是整个 state？
2. stream_mode 还有 "values" / "updates" / "debug" 这几种，
   分别流什么？哪种适合给用户看？
3. astream_events 的 v2 协议在 LangGraph 上怎么用？
   能不能精确到"工具调用开始 / 结束"事件？

每问 100 字内。
```

**自检**：能讲清「LangGraph 的 stream_mode 选 'updates' 给用户看进度，'values' 给开发者看完整状态」。

---

### 任务 3 · `with_retry` 自动重试（10 分钟）

LLM API 偶尔会：

- 503 Service Unavailable（OpenAI / DashScope 后端瞬时挂）
- timeout（网络抖动）
- rate limit（429）

**做什么**：给链加自动重试。

```python
from langchain_core.runnables import RunnableConfig

llm = make_llm()

# 给 llm 包一层 retry
llm_with_retry = llm.with_retry(
    stop_after_attempt=3,                          # 最多重试 3 次
    wait_exponential_jitter=True,                  # 指数退避 + 抖动
    retry_if_exception_type=(ConnectionError, TimeoutError),  # 只对网络类异常重试
)

# 用法跟普通 llm 一样
response = llm_with_retry.invoke("hello")
```

**给 AI 的 prompt**：

```
我要给 LLM 加重试。请引导我（不直接给代码）：

1. 哪些异常应该重试？哪些不应该？
   （提示：401 Unauthorized 重试一万次也没用——那是 key 错；
    但 503 / timeout / 429 是瞬时问题，重试有意义）
2. 指数退避 + jitter 是什么？为什么不用固定间隔？
   （提示：服务雪崩时所有客户端同时 retry 会加剧问题）
3. stop_after_attempt=3 包含第一次调用吗？
   总共会调几次（first try + retry）？

每问 80 字内。
```

**自检**：故意把 base_url 改错（比如 `https://wrong.example.com`），看 retry 真的尝试了 3 次再放弃。

---

### 任务 4 · `with_fallbacks` 主备切换（10 分钟）

retry 解决"瞬时挂"，fallback 解决"主服务长期不可用"——切到备用 LLM 接着跑。

**做什么**：

```python
primary_llm = make_llm(model="qwen-plus")           # 主：通义千问
backup_llm  = make_llm(model="qwen-turbo")          # 备：另一个模型（演示用）

llm_with_fallback = primary_llm.with_fallbacks([backup_llm])

# 用法跟普通 llm 一样
response = llm_with_fallback.invoke("用 50 字介绍 LangChain")
```

**给 AI 的 prompt**：

```
with_fallbacks 跟 with_retry 的差别我有点混淆。

请帮我理清（不直接给代码）：
1. 用一张表对比 retry vs fallback：触发条件 / 切换对象 / 性能影响 / 适用场景。
2. 能不能组合用？语法是什么？
   （比如：先 retry 3 次，全失败再 fallback 到备用）
3. 生产里更常用哪个？为什么？
   （提示：跟 SLA / 多供应商策略有关）

回答 250 字内。
```

**自检**：把主 llm 故意改成不存在的模型（比如 `model="qwen-99999-plus"`），跑起来看 fallback 真的切到 backup。

---

### 任务 5 · 自检 + 思考"什么场景必须用这两件套"（5 分钟）

**给 AI 的 prompt**：

```
我学完了 with_retry / with_fallbacks。
请帮我做"工程感"自检（不要给代码，只回答判断题）：

1. 一个内部小工具（你自己用）需要 retry 吗？需要 fallback 吗？
2. 一个 ToB 产品的 LLM 客服 bot（SLA 要求 99.5% 可用）必须有什么？
3. 一个 ToC 产品的"AI 写作助手"用户量百万级，
   除了 retry / fallback 还需要什么？
   （提示：跟成本 / 限流 / 优雅降级有关）

每问 50 字内。诚实回答，不要客套。
```

**自检**：能列出"工程化 LLM 应用的可靠性 4 件套"（retry / fallback / rate limit / 降级策略）。

---

## 通关条件

- [ ] `python _scratch/my_w2_03_resilience.py` 跑通
- [ ] 至少跑过 chain.stream + astream_events 两种
- [ ] 至少 1 个 with_retry 用法 + 1 个 with_fallbacks 用法
- [ ] 能用一句话讲清「retry vs fallback 的核心差别」
- [ ] 知道生产 LLM 应用的"可靠性 4 件套"

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建 `2026-XX-XX-week2-03.md`：

```markdown
# Week 2 · 03_streaming_and_resilience — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → [week-3-langgraph/01_simple_graph.md](../week-3-langgraph/01_simple_graph.md)（开始 LangGraph）
- 想多练 → 给 week-3 学过的 ReAct Agent 加 with_retry 和 with_fallbacks，跑一次故意触发 fallback
- 想深入 → 看 LangChain 文档的 `astream_log` API（流式调试用，比 events 更底层）
