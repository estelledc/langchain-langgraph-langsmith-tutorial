# Debug Recipes — 报错时的诊断手册

> LangChain / LangGraph / LangSmith 高频报错速查 + 给 AI 的诊断 prompt 模板。报错先来这页搜，搜不到再开 AI 对话。

**配套**：[docs/test-runs.md](test-runs.md) 是作者本人 14 个 final 全跑过一遍的实战记录，遇到没在本页的报错，去那里再翻一遍。

---

## 0 · 报错诊断三步法（每次都按这个走）

```
Step 1: 看报错最后 1 行 — 错误类型是 ImportError / KeyError / TypeError / SSL 还是别的？
Step 2: 描述"我以为" — 你预期程序做什么？这一步逼出心智模型
Step 3: 砍最小复现 — 删到 5-10 行能触发同一个错；多半你自己就看出来了
```

**还不行**：去对应错误类型的章节查表；查不到 → 用[万能诊断 prompt](#万能诊断-prompt)。

---

## 1 · ImportError / ModuleNotFoundError

### 1.1 `ModuleNotFoundError: No module named 'langchain_classic'`

**根因**：langchain 1.x 把旧 `langchain.agents` 系列拆到独立包 `langchain-classic`。

**修法**：

```bash
pip install langchain-classic>=1.0
```

代码侧：

```python
# ❌ 旧（langchain 0.x）
from langchain.agents import AgentExecutor, create_tool_calling_agent

# ✅ 新（langchain 1.x）
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
```

涉及文件：`final/01_langchain/06_tools_agent.py`。详见 [test-runs.md 1.2](test-runs.md)。

---

### 1.2 `ModuleNotFoundError: No module named 'langchain_core.pydantic_v1'`

**根因**：langchain 1.x 完全切到 pydantic v2，移除了 v1 兼容层。

**修法**：

```python
# ❌ 旧
from langchain_core.pydantic_v1 import BaseModel, Field

# ✅ 新
from pydantic import BaseModel, Field
```

涉及文件：`final/01_langchain/03_chains.py`。详见 [test-runs.md 1.1](test-runs.md)。

---

### 1.3 `ModuleNotFoundError: No module named 'final'`

**根因**：你直接 `python final/01_langchain/01_hello_llm.py` 跑，但 final 内部 `from final._common import make_llm` 找不到 `final` 包路径。

**修法**：脚本顶部应该有：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
```

如果没有，照抄。或者用模块方式跑：

```bash
python -m final.01_langchain.01_hello_llm  # 注意 - 要换 _，因为 Python 包名
```

更简单：在仓库根目录直接 `python final/01_langchain/01_hello_llm.py`，且确保上面 sys.path 那两行在。

---

### 1.4 `ImportError: Install grandalf to draw graphs`

**根因**：`graph.get_graph().draw_ascii()` 需要 `grandalf` 库，requirements.txt 漏装。

**修法**：

```bash
pip install grandalf
```

或者把那行注释掉——可视化不影响主体功能。

---

## 2 · KeyError / 环境变量

### 2.1 `KeyError: 'DASHSCOPE_BASE_URL'` 或 `'DASHSCOPE_API_KEY'`

**根因**：.env 没加载 / 没填完整 / 拼写错。

**诊断步骤**：

```bash
# 1. 确认 .env 在仓库根
ls -la .env

# 2. 确认变量名拼写
grep DASHSCOPE .env  # 应该看到两行：BASE_URL 和 API_KEY

# 3. 验证 dotenv 能加载
python -c "from dotenv import load_dotenv; load_dotenv(override=True); import os; print(os.environ.get('DASHSCOPE_API_KEY', 'NOT FOUND')[:8])"
```

如果第 3 步打印 `NOT FOUND`：

- 确认 `python` 跑的就是仓库 `.venv` 里那个
- 确认 .env 文件没 BOM、没多余空格
- 确认变量名前没空格

---

### 2.2 `KeyError: 'LANGCHAIN_API_KEY'`

**根因**：LangSmith key 没配，但 `LANGCHAIN_TRACING_V2=true`。

**修法**：

```bash
# .env 里加：
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_xxx_你的_key
LANGCHAIN_PROJECT=study
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

**临时关掉 trace**（如果你只想调代码，不需要 trace）：

```bash
# .env
LANGCHAIN_TRACING_V2=false
```

---

## 3 · 网络 / SSL / 401 / 403

### 3.1 `openai.AuthenticationError: 401 Unauthorized`

**根因**：DashScope API key 错。

**诊断**：

```bash
# 直接 curl 测 key
curl -X POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  -H "Authorization: Bearer YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-plus","messages":[{"role":"user","content":"hi"}]}'
```

回 200 = key OK；回 401 = 重新去 [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com) 控制台拷一遍。

---

### 3.2 `openai.PermissionDeniedError: 403 - This token has no access to model text-embedding-v3`

**根因**：你的 DashScope key 只授权了 chat 模型，没开 embedding。

**修法**：

- 公网账号：去控制台找"模型权限"申请 embedding-v3 访问
- 公司网关 token：找平台同事开通

**临时绕开**：跳过 RAG 那一篇（`05_rag_basic.py`），先把 chat 类的题做完。

详见 [test-runs.md 2.1](test-runs.md)。

---

### 3.3 `SSLCertVerificationError: certificate verify failed`

**根因**：你在公司 MDM/mitmproxy 网络下，pip / langsmith / OpenAI SDK 要走自签 CA。

**修法**：

```bash
export SSL_CERT_FILE=/path/to/your-corp-ca.pem
export REQUESTS_CA_BUNDLE=/path/to/your-corp-ca.pem
```

**注意**：langsmith 0.8 的 client 同步 GET /info 走 urllib3，**不读** SSL_CERT_FILE，会单独失败。这是已知问题。

家用 / 公网用户不会撞这个，所以 SETUP.md 没写。

---

### 3.4 `Connection error` / `httpx.ConnectError`

**根因**：

- 网络真断了
- 公司网络要走代理
- DNS 污染

**诊断**：

```bash
# 测 DashScope 通不通
ping dashscope.aliyuncs.com
curl -I https://dashscope.aliyuncs.com

# 测 LangSmith
curl -I https://api.smith.langchain.com
```

**修法**：

```bash
# 走代理
export HTTPS_PROXY=http://your-proxy:port
export HTTP_PROXY=http://your-proxy:port
```

---

## 4 · 类型 / API 变更（langchain 1.x）

### 4.1 `'str' object has no attribute 'value'` (创建 dataset 时)

**根因**：langsmith 0.8 的 `create_dataset(data_type=...)` 不再接受字符串 `"kv"`，要传 `DataType` enum。

**修法**：直接删掉这个参数（kv 是默认值）：

```python
# ❌
ls_client.create_dataset(dataset_name="x", data_type="kv")

# ✅
ls_client.create_dataset(dataset_name="x")
```

涉及文件：`final/03_langsmith/03_dataset.py`。详见 [test-runs.md 1.4](test-runs.md)。

---

### 4.2 `ImportError: cannot import name 'LangChainStringEvaluator' from 'langsmith.evaluation'`

**根因**：langsmith 0.8 移除了内置 `LangChainStringEvaluator`。

**修法**：写自定义 LLM-as-Judge evaluator（更稳，不受 langsmith 版本影响）。

完整模板见 [final/03_langsmith/02_evaluation.py](../final/03_langsmith/02_evaluation.py) 里的 `llm_judge_evaluator` 函数；教学完整版见 [tutorial/week-4-langsmith-and-project/02_evaluation.md](../tutorial/week-4-langsmith-and-project/02_evaluation.md) 任务 3。

---

### 4.3 `EOFError: EOF when reading a line` (在 input() 触发)

**根因**：你在批跑 / CI / 非交互终端跑，stdin 被关，`input()` 直接抛 EOF。

**修法**：

```python
try:
    answer = input("是否继续？(y/n): ").strip().lower()
except EOFError:
    print("(检测到非交互终端，默认 n)")
    answer = "n"
```

涉及文件：`final/02_langgraph/03_human_in_the_loop.py`。详见 [test-runs.md 1.5](test-runs.md)。

---

## 5 · LangGraph 特有报错

### 5.1 `langgraph.errors.GraphRecursionError: Recursion limit of 25 reached`

**根因**：图陷入循环（比如 ReAct Agent 一直调工具不退出）。

**诊断**：

```python
# invoke 时显式调高限制 + 看 trace
result = graph.invoke(
    initial_state,
    config={"recursion_limit": 50}  # 默认 25
)
```

**真正修法**：在条件函数里加退出条件，比如限工具调用次数。教学版见 [tutorial/week-3-langgraph/02_conditional_edges.md](../tutorial/week-3-langgraph/02_conditional_edges.md) 任务 4。

---

### 5.2 `KeyError: 'tools'` 或 `KeyError: 'agent'` (路由 dict 不匹配)

**根因**：`add_conditional_edges` 的 mapping dict 里写的字符串，跟条件函数返回值对不上。

**例子**：

```python
# 条件函数返回 "use_tool"
def my_router(state) -> str:
    return "use_tool"

# 但路由 mapping 写的是 "tools"
builder.add_conditional_edges("agent", my_router, {"tools": "tool_node", ...})
# ❌ KeyError: 'use_tool'
```

**修法**：让两边的字符串一致。或者用 `tools_condition`（内置已经处理好）。

---

### 5.3 State 字段没被更新 / 节点函数返回了完整 State

**根因**：节点函数应该**返回 dict（要更新的字段）**，不是返回完整 State。

```python
# ❌ 错
def my_node(state: MyState) -> MyState:
    state["count"] += 1
    return state  # 返回完整 state 不规范

# ✅ 对
def my_node(state: MyState) -> dict:
    return {"count": state["count"] + 1}  # 只返回要更新的字段
```

**特殊情况**：用了 `add_messages` reducer 的字段：

```python
# messages 字段，return 的 list 会被追加（不是替换）
return {"messages": [new_message]}  # 追加 new_message 到现有 messages
```

---

## 6 · 性能 / 慢

### 6.1 `evaluate()` 跑批奇慢（> 20 分钟）

**根因**：默认顺序跑，5 个 example × 3 evaluator ≈ 15 次 LLM × 多轮。

**修法**：

```python
results = evaluate(
    target_function,
    data=dataset_name,
    evaluators=[...],
    max_concurrency=5,  # 5 个 example 并发
)
```

**注意**：免费 5K runs/月 跑批一次 ≈ 30 runs，量级可控。

---

### 6.2 单次 LLM 调用 5-15 秒（公司网络下）

**根因**：走公司网关绕跳；公网直连 DashScope 通常 1-3 秒。

**修法**：网络环境问题，代码层无解。开发时认账。

---

## 万能诊断 prompt

如果以上都不匹配你的报错，复制这个 prompt 给 Claude Code / Cursor：

```
我跑 [文件名]，预期 [你的预期]，实际报错：

[完整报错栈，至少最后 10 行]

我所在的环境：
- Python [python --version 输出]
- LangChain 版本 [pip show langchain 的 Version 行]
- 操作系统 [Mac / Win / Linux]
- 网络 [公司 / 家用 / VPN]

我已经试过：
- [尝试 1，结果]
- [尝试 2，结果]

请按以下格式回答：
1. 报错类型是 [Import / KeyError / 网络 / API 变更 / 其他]
2. 给我 3 个候选根因，按可能性排序，每个 1-2 句话
3. 不要直接修——让我先猜哪个最可能

不要建议加 try/except 或者其他"防御性"代码。
```

---

## 怎么往这页加新条目

撞到一个本页没列的报错并修好后，在 `_scratch/journal/` 当天日志里写：

```markdown
## 调试发现：[报错类型]

- 根因：...
- 修法：...
- 涉及文件：...
- 触发条件：...
```

存够 3-5 条后，整理成本页的新章节提 PR。

---

## 版本

- v1（2026-05）：初版，覆盖 6 大类共 16 个常见报错，对齐 langchain 1.3 / langsmith 0.8 / langgraph 1.2
