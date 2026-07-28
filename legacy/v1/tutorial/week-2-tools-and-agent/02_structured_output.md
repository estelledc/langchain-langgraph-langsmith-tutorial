# 02 · Structured Output — 让 LLM 直接返回 Pydantic 对象

> **本步带回家的概念**：让 LLM 输出 JSON / Pydantic 对象而不是自由文本——靠 `.with_structured_output(YourSchema)`。生产里 90% 的 LLM 应用都需要这个，因为下游代码要的是字段不是字符串。
>
> **配套代码**：本篇没有 final/.py（这是补充篇，自己跟 AI 写完即可）
> **预计耗时**：30-45 分钟
> **官方文档对应**：[Structured outputs](https://python.langchain.com/docs/how_to/structured_output/)，但 zero 版补了"为什么需要"和挖空写

---

## 准备 (5 分钟)

- [ ] [01_tools_agent.md](01_tools_agent.md) 通关
- [ ] [docs/concepts.md](../../docs/concepts.md) 13 (Agent) 概念已扫一遍
- [ ] `_scratch/my_w2_02_structured.py` 已建好

---

## 任务卡

### 任务 1 · 先看反例：自由文本输出的痛苦（5 分钟）

**做什么**：在 `_scratch/my_w2_02_structured.py` 写 5 行代码，让 LLM 答："给我 3 个适合周末出游的城市，每个城市标推荐理由"。

```python
from final._common import make_llm
from langchain_core.messages import HumanMessage

llm = make_llm(temperature=0.5)
response = llm.invoke([HumanMessage(content="给我 3 个适合周末出游的城市，每个城市标推荐理由")])
print(response.content)
```

跑两次。观察输出格式——可能这次是：

```
1. 杭州 — 西湖美景...
2. 苏州 — 园林文化...
3. ...
```

也可能下次是：

```
推荐城市如下：
- 杭州：西湖美景...
- ...
```

**痛点**：你想用代码处理这 3 个城市（提取出 list），但每次格式不同——正则匹配马上就崩。

**给 AI 的 prompt**：

```
我跑同样的 prompt 两次，LLM 给的格式不一样
（一次是数字编号，一次是 bullet point）。

请用日常类比帮我搞清 2 件事：

1. LLM 为什么"自由发挥"格式？是它故意还是不可控？
   能不能比喻成"让员工写汇报，不规定格式他每次都不一样"？
2. 在生产代码里，这种"格式漂移"会怎么坑你？
   假设我用 split("\n") 后做 strip 提取城市名——
   什么情况下会断？

回答 200 字内，每问独立段落。
```

**自检**：能讲清"自由文本对下游代码不友好"。

---

### 任务 2 · 用 with_structured_output + Pydantic（10 分钟）

**做什么**：把任务 1 改成结构化输出。

**给 AI 的 prompt**：

```
我要让 LLM 返回一个稳定结构的对象（不是自由文本）：
- 城市列表，每个城市包含 name 和 reason 两个字段。

请引导我（不直接给代码）：

1. Pydantic BaseModel 定义这个 schema 应该有几个类？
   1 个还是 2 个？为什么？（提示：列表用什么类型表示？）
2. 在 LangChain 里怎么把 schema "绑"到 llm 上？
   是改 prompt 还是改 llm 对象？
   （提示：llm.with_structured_output(YourSchema) 这个方法返回的是什么）
3. 调用之后拿到的是 Pydantic 对象还是 dict？
   .dict() / .model_dump() 哪个是 v2 用的？

每次只问我一个问题。
```

代码骨架（不要直接抄，跟 AI 对话写）：

```python
from pydantic import BaseModel, Field
from final._common import make_llm

class City(BaseModel):
    name: str = Field(description="城市名称")
    reason: str = Field(description="推荐理由，简短一句")

class CityList(BaseModel):
    cities: list[City]

llm = make_llm(temperature=0.5)
structured_llm = llm.with_structured_output(CityList)
result: CityList = structured_llm.invoke("给我 3 个适合周末出游的城市，每个城市标推荐理由")

for c in result.cities:
    print(f"{c.name}: {c.reason}")
```

跑两次，观察：**格式 100% 稳定**——`result.cities[0].name` 永远是字符串，`reason` 永远是字符串。

**自检**：能讲清「`Field(description=...)` 这个描述给谁看的」（提示：跟工具的 docstring 一样——给 LLM 看，让它知道怎么填这个字段）。

---

### 任务 3 · 嵌套 schema：让 LLM 抽取信息（10 分钟）

**做什么**：写一个抽取器，输入一段招聘描述文本，让 LLM 抽出结构化字段。

```python
class JobPosting(BaseModel):
    title: str = Field(description="职位名称")
    company: str = Field(description="公司名称")
    location: str = Field(description="工作地点")
    salary_min: int | None = Field(default=None, description="月薪下限，单位元；如果没说则 None")
    salary_max: int | None = Field(default=None, description="月薪上限，单位元；如果没说则 None")
    requirements: list[str] = Field(description="任职要求条目")
```

**给 AI 的 prompt**：

```
我要让 LLM 从一段招聘文本里抽信息成 JobPosting 对象。

请引导我（不直接给代码）：
1. salary_min / salary_max 用 int | None + default=None
   是为了什么？没说工资时 LLM 应该填什么？（None vs 0 vs 不填）
2. requirements: list[str] —— LLM 怎么决定切几条？
   "需要 Python 3 年和 React 1 年"会被切 1 条还是 2 条？
   （提示：跟 LLM 怎么"读" Field 的 description 有关）
3. 如果文本里没明确写公司名，LLM 会幻觉编一个吗？
   怎么在 description 里加约束让它不编？
   （提示："如果文本未提及，填'未明示'"）

每问 100 字内。
```

测试输入（贴在 invoke 里）：

```
"招聘后端工程师 - Python，远程办公或北京。月薪 25-40k。要求 3+ 年 Python 经验，
熟悉 FastAPI，加分项：有 LLM 应用开发经验。本公司是一家成长期 AI 创业公司。"
```

跑完打印 result，观察 6 个字段是否都对。

**自检**：故意改成"待遇面议"，看 salary_min/max 是不是 None。

---

### 任务 4 · 跟工具调用对比——它们在底层是同一件事（5 分钟）

**做什么**：思考一个问题——`with_structured_output` 和 `bind_tools` 看上去是两个东西，但其实……

**给 AI 的 prompt**：

```
我学过 week-2/01 的 bind_tools，知道它让 LLM 能调用 @tool 函数。
现在学 with_structured_output，让 LLM 返回 Pydantic 对象。

请帮我连接这两个概念：

1. 在底层 OpenAI / DashScope API 协议里，
   "调用工具"和"返回结构化对象"是不是同一个机制？
   （提示：function calling / tool_calls 里的 arguments 字段长什么样？）
2. with_structured_output(YourSchema) 内部能不能理解为
   "把 YourSchema 包装成一个虚拟工具，强制 LLM 调用它"？
3. 那为什么 LangChain 还要分两个 API？
   什么场景用 bind_tools 什么场景用 with_structured_output？

每问 80 字内。
```

**自检**：能用一句话讲清「结构化输出本质是 function calling 的特殊用法」。

---

### 任务 5 · 自检（5 分钟）

`_scratch/my_w2_02_structured.py` 应该包含：

- [x] 任务 1 的反例（自由文本）
- [x] 任务 2 的稳定结构（CityList）
- [x] 任务 3 的招聘抽取器（JobPosting）

**给 AI 的 prompt**：

```
我的 _scratch/my_w2_02_structured.py：
[贴代码]

请帮我做项目级 code review（不要直接改代码）：
1. Pydantic 字段的 description 写得有信息量吗？还是流于"职位名称"这种重复字段名？
2. 如果我改成 OpenAI gpt-4o 跑同一段代码，with_structured_output 还能用吗？
   （提示：跟 LLM 提供商有没有 function calling 支持有关）
3. 我的 prompt 写得足够明确让 LLM 不幻觉吗？
   有什么 prompt 写法能让结构化输出更稳？

哪些是风格差异，哪些是真问题？真问题告诉我"为什么这样写更好"。
```

---

## 通关条件

- [ ] `python _scratch/my_w2_02_structured.py` 跑通
- [ ] 至少 2 个 Pydantic schema（一个简单 + 一个嵌套）
- [ ] 至少跑过一次"无明确字段"的输入，看 LLM 怎么处理（None / 默认 / 幻觉）
- [ ] 能用一句话讲清「结构化输出 = function calling 的特殊用法」
- [ ] 知道在生产里什么时候用 `with_structured_output` vs 自由文本（90% 后端服务用前者）

---

## 卡点日志（必填）

打开 `_scratch/journal/`，新建当天文件 `2026-XX-XX-week2-02.md`：

```markdown
# Week 2 · 02_structured_output — 卡点日志

## 卡点

## "原来如此"时刻

## 想留作复用的 prompt

## 还没搞懂的（留尾巴）
```

---

## 通往下一站

- 全部通关 → [03_streaming_and_resilience.md](03_streaming_and_resilience.md)（流式 + retry/fallback）
- 想多练 → 用 `with_structured_output` 写一个"中文 → 英文翻译 + 元数据"抽取器（输出 `{translated: str, confidence: float, alternative_translations: list[str]}`）
