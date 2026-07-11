# LangChain Tutorial Zero — Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把仓库重构成「双轨 + 任务卡」结构，并完成 week-1 五篇标杆 tutorial。完成后任何零基础选手都能 fork 这个仓库、按 tutorial/week-1/ 走完一周。

**Architecture:** `final/` 放成品代码（参考答案）+ `tutorial/` 放学习剧本（任务卡 + 给 AI 的 prompt）+ 顶层 `HOW_TO_LEARN_WITH_AI.md` 元教程。学习者主战场在 `_scratch/`（gitignore）。

**Tech Stack:** Python 3.10+ / langchain >= 0.3 / langchain-openai / DashScope（通义千问）/ LangSmith / FAISS。Markdown + GitHub 渲染做教程载体。

**Spec：** [docs/superpowers/specs/2026-05-29-langchain-tutorial-refactor-design.md](../specs/2026-05-29-langchain-tutorial-refactor-design.md)

**Phase 范围：** 本计划只覆盖 Spec 的 Phase 0（骨架 + 元教程）+ Phase 1（week-1 五篇 tutorial 标杆）。Phase 2（week-2/3/4 批量推进）+ Phase 3（cheatsheet + 发布）会在 week-1 完成并复盘后单独出新计划。

**Repo cwd：** 所有命令都在 `$HOME/projects/langchain-tutorial-zero/` 下执行。

---

## File Structure

| 文件 | 状态 | 责任 |
|------|------|------|
| `README.md` | 重写 | 30 秒入口：你在哪 / 该读哪 / 跳到 SETUP |
| `HOW_TO_LEARN_WITH_AI.md` | 新建 | 元教程：怎么用 CC/Cursor 学这套 |
| `SETUP.md` | 新建 | 环境搭建（API key、venv、依赖、smoke test） |
| `.gitignore` | 修改 | 加 `_scratch/journal/*` 但保留 `.gitkeep` |
| `final/_common.py` | 新建 | 共享 boilerplate：load_dotenv + 模型名常量 |
| `final/01_langchain/*.py` | 移动+修改 | 从根目录 `01_langchain/` 移过来；用 _common；改 qwen3.5-plus → qwen-plus；加 anchor 注释 |
| `final/02_langgraph/*.py` | 同上 | |
| `final/03_langsmith/*.py` | 同上 | |
| `final/04_project/*.py` | 同上 | |
| `tutorial/README.md` | 新建 | week-by-week 学习路线索引 |
| `tutorial/week-1-langchain/01_hello_llm.md` | 新建 | gold standard 标杆 |
| `tutorial/week-1-langchain/02_prompt_template.md` | 新建 | 按 01 模板写 |
| `tutorial/week-1-langchain/03_chains.md` | 新建 | 同上 |
| `tutorial/week-1-langchain/04_memory.md` | 新建 | 同上 |
| `tutorial/week-1-langchain/05_rag_basic.md` | 新建 | 同上 |
| `_scratch/README.md` | 新建 | 提醒：在这里跟 AI 一起写自己版本 |
| `_scratch/journal/.gitkeep` | 新建 | 保留 journal 目录但忽略具体日志 |

---

## Task 1: Pre-flight — 验证当前 final 代码可运行

**Files:**
- Test: `01_langchain/01_hello_llm.py`（当前路径，move 之前先验）

- [ ] **Step 1: 检查 .env 是否存在**

```bash
ls -la .env 2>&1 | head
```

Expected：能看到 `.env` 文件（已有真实 key）。如果没有，**先停下**，去 `cp .env.example .env` 并填 key 再继续。

- [ ] **Step 2: syntax 检查所有 .py（不实际跑）**

```bash
python -c "
import ast, pathlib
for p in pathlib.Path('.').rglob('*.py'):
    if 'venv' in str(p): continue
    try:
        ast.parse(p.read_text())
    except SyntaxError as e:
        print(f'SYNTAX ERROR: {p}: {e}')
        raise SystemExit(1)
print('all 17 .py syntax ok')
"
```

Expected：`all 17 .py syntax ok`

- [ ] **Step 3: 真实跑一个最简单的 final（验证 .env 有效 + 网络通）**

```bash
python 01_langchain/01_hello_llm.py 2>&1 | tail -5
```

Expected：能看到 LLM 回答（"什么是 LangChain？" 的回答）+ 末尾 `✅ 运行完毕！`。如果报 401/404，先停下处理 key/baseurl/模型名问题。

- [ ] **Step 4: 不 commit，只验证**

这一步只是 pre-flight，不产生 commit。

---

## Task 2: 文件移位 — 把 4 个目录搬进 final/

**Files:**
- Move: `01_langchain/` → `final/01_langchain/`
- Move: `02_langgraph/` → `final/02_langgraph/`
- Move: `03_langsmith/` → `final/03_langsmith/`
- Move: `04_project/` → `final/04_project/`

- [ ] **Step 1: 用 git mv 保留历史**

```bash
mkdir -p final
git mv 01_langchain final/01_langchain
git mv 02_langgraph final/02_langgraph
git mv 03_langsmith final/03_langsmith
git mv 04_project final/04_project
```

- [ ] **Step 2: 验证移位成功**

```bash
ls final/
git status -s | head
```

Expected：`final/` 下有 4 个子目录；git status 显示一堆 `R` (rename) 行。

- [ ] **Step 3: 验证移位后仍能跑**

```bash
python final/01_langchain/01_hello_llm.py 2>&1 | tail -3
```

Expected：仍输出 `✅ 运行完毕！`（因为脚本内部不依赖位置，env 也不依赖路径）。

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: move 17 .py into final/ (双轨结构第一步)"
```

---

## Task 3: 抽 final/_common.py — 共享 boilerplate

**Files:**
- Create: `final/_common.py`

- [ ] **Step 1: 写 _common.py**

```python
"""
final/_common.py — 所有 final 脚本共享的 boilerplate
学习者别改这里；改它会影响所有 17 个示例。
"""

import os
from dotenv import load_dotenv

# override=True 确保 .env 覆盖 shell 环境变量
load_dotenv(override=True)

# DashScope（通义千问）实际可用的模型 ID
# 注意：原 README 里写的 "qwen3.5-plus" 不存在；正确 ID 是 qwen-plus
DEFAULT_MODEL = "qwen-plus"
DASHSCOPE_BASE_URL = os.environ["DASHSCOPE_BASE_URL"]
DASHSCOPE_API_KEY = os.environ["DASHSCOPE_API_KEY"]


def make_llm(model: str = DEFAULT_MODEL, temperature: float = 0.7, **kwargs):
    """统一构造 ChatOpenAI 实例，兼容 DashScope。"""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
        temperature=temperature,
        **kwargs,
    )
```

- [ ] **Step 2: 验证 _common.py 自身能 import**

```bash
python -c "from final._common import make_llm; llm = make_llm(); print(type(llm).__name__)"
```

Expected：`ChatOpenAI`（无报错）。

- [ ] **Step 3: Commit**

```bash
git add final/_common.py
git commit -m "feat: add final/_common.py 抽 load_dotenv + 模型构造"
```

---

## Task 4: 改造 17 个 .py — 用 _common，改模型名，加 anchor 注释

**Files:**
- Modify: `final/**/*.py`（17 个，逐个处理）

**统一改造规则**（每个 .py 都按这个改）：

1. 删除原文件顶部的 `from dotenv import load_dotenv` + `load_dotenv(override=True)` + `import os`
2. 改 `import` 段，加上 `from final._common import make_llm`
3. 把 `llm = ChatOpenAI(...)` 整段替换为 `llm = make_llm(temperature=0.7)`
4. 改完之后，文件顶部 docstring 后面加一行 anchor 注释：`# 配套教程：tutorial/week-X/0Y_xxx.md`（具体 week + 编号见映射表）

**配套教程映射表**：

| final 文件 | 对应 tutorial |
|------|------|
| `final/01_langchain/01_hello_llm.py` | `tutorial/week-1-langchain/01_hello_llm.md` |
| `final/01_langchain/02_prompt_template.py` | `tutorial/week-1-langchain/02_prompt_template.md` |
| `final/01_langchain/03_chains.py` | `tutorial/week-1-langchain/03_chains.md` |
| `final/01_langchain/04_memory.py` | `tutorial/week-1-langchain/04_memory.md` |
| `final/01_langchain/05_rag_basic.py` | `tutorial/week-1-langchain/05_rag_basic.md` |
| `final/01_langchain/06_tools_agent.py` | `tutorial/week-2-tools-and-agent/01_tools_agent.md` |
| `final/02_langgraph/01_simple_graph.py` | `tutorial/week-3-langgraph/01_simple_graph.md` |
| `final/02_langgraph/02_conditional_edges.py` | `tutorial/week-3-langgraph/02_conditional_edges.md` |
| `final/02_langgraph/03_human_in_the_loop.py` | `tutorial/week-3-langgraph/03_human_in_the_loop.md` |
| `final/02_langgraph/04_multi_agent.py` | `tutorial/week-3-langgraph/04_multi_agent.md` |
| `final/03_langsmith/01_tracing.py` | `tutorial/week-4-langsmith-and-project/01_tracing.md` |
| `final/03_langsmith/02_evaluation.py` | `tutorial/week-4-langsmith-and-project/02_evaluation.md` |
| `final/03_langsmith/03_dataset.py` | `tutorial/week-4-langsmith-and-project/03_dataset.md` |
| `final/04_project/agent.py` | `tutorial/week-4-langsmith-and-project/04_capstone.md` |
| `final/04_project/tools.py` | `tutorial/week-4-langsmith-and-project/04_capstone.md` |
| `final/04_project/graph.py` | `tutorial/week-4-langsmith-and-project/04_capstone.md` |
| `final/04_project/eval.py` | `tutorial/week-4-langsmith-and-project/04_capstone.md` |

**注意**：本计划只新建 week-1 的 tutorial（5 篇），但 anchor 注释要写**所有 17 个**——这样 fork 者顺着 anchor 注释看，能清楚预期未来还会有哪些 tutorial（即使现在还是 404）。

- [ ] **Step 1: 改 final/01_langchain/01_hello_llm.py 作为模板**

完整改造后的内容（直接覆盖原文件）：

```python
"""
01_hello_llm.py — LangChain 入门：直接调用 LLM

知识点：
- ChatOpenAI 对接 DashScope（OpenAI 兼容接口）
- HumanMessage / AIMessage / SystemMessage
- invoke() / stream() / batch() 三种调用方式
- LangSmith 自动追踪（.env 中 LANGCHAIN_TRACING_V2=true 即可）
"""
# 配套教程：tutorial/week-1-langchain/01_hello_llm.md

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from final._common import make_llm

# DashScope 兼容 OpenAI 格式，用 _common.make_llm() 统一构造
llm = make_llm(temperature=0.7)


def demo_invoke():
    """最简单的一次性调用"""
    print("=" * 50)
    print("【invoke 调用】")

    messages = [
        SystemMessage(content="你是一个简洁的 AI 助手，回答不超过 50 字。"),
        HumanMessage(content="什么是 LangChain？"),
    ]

    response: AIMessage = llm.invoke(messages)

    # response.content 是文本内容
    print(f"回答：{response.content}")
    # response.response_metadata 包含 token 用量等元数据
    print(f"Token 用量：{response.response_metadata.get('token_usage', {})}")


def demo_stream():
    """流式输出——适合长文本场景"""
    print("\n" + "=" * 50)
    print("【stream 流式输出】")

    messages = [
        HumanMessage(content="用三句话介绍一下 Python 语言的特点。"),
    ]

    print("回答：", end="", flush=True)
    for chunk in llm.stream(messages):
        # chunk 是 AIMessageChunk，content 是本次片段文本
        print(chunk.content, end="", flush=True)
    print()  # 换行


def demo_batch():
    """批量调用——一次发送多条消息"""
    print("\n" + "=" * 50)
    print("【batch 批量调用】")

    questions = [
        [HumanMessage(content="1+1=?")],
        [HumanMessage(content="太阳系有几颗行星？")],
        [HumanMessage(content="Python 之父是谁？")],
    ]

    responses = llm.batch(questions)
    for i, resp in enumerate(responses):
        print(f"Q{i+1}: {questions[i][0].content}")
        print(f"A{i+1}: {resp.content}\n")


if __name__ == "__main__":
    demo_invoke()
    demo_stream()
    demo_batch()

    print("\n✅ 运行完毕！请前往 https://smith.langchain.com 查看 'study' 项目的 Trace 记录。")
```

- [ ] **Step 2: 验证 01_hello_llm.py 仍能跑**

```bash
python -c "import sys; sys.path.insert(0, '.'); from final._common import make_llm; print('import ok')"
python final/01_langchain/01_hello_llm.py 2>&1 | tail -5
```

Expected：`import ok`，然后 LLM 输出 + `✅ 运行完毕！`。

⚠️ 注意：如果报 `ModuleNotFoundError: final`，是因为 final 里的脚本被当模块跑时没把 cwd 加 sys.path。**解决方式**：在每个 .py 顶部加：

```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
```

放在 docstring 之后、其他 import 之前。如果验证后 ok 不需要这个，就别加。

- [ ] **Step 3: 批量改造剩下 16 个 .py**

对每个文件做相同的改造：
1. 删除 `from dotenv import load_dotenv`、`load_dotenv(override=True)`、`import os`
2. 加 `from final._common import make_llm`（按需补 sys.path 行，看 Step 2 是否需要）
3. 把整个 `llm = ChatOpenAI(model="qwen3.5-plus", base_url=..., api_key=..., temperature=0.7)` 替换为 `llm = make_llm(temperature=0.7)`（如果原 temperature 不是 0.7，保留原值）
4. docstring 后插入 `# 配套教程：tutorial/week-X/0Y_xxx.md`（按映射表）

**实施提示**：可以并发用 Edit 工具改，每改完一个用 `python -c "import ast; ast.parse(open('FILE').read())"` 校 syntax。

- [ ] **Step 4: 全量 syntax check**

```bash
python -c "
import ast, pathlib
errors = []
for p in pathlib.Path('final').rglob('*.py'):
    try: ast.parse(p.read_text())
    except SyntaxError as e: errors.append(f'{p}: {e}')
print('OK' if not errors else '\n'.join(errors))
"
```

Expected：`OK`

- [ ] **Step 5: 抽样真跑 3 个验证**

```bash
python final/01_langchain/01_hello_llm.py 2>&1 | tail -3
python final/01_langchain/02_prompt_template.py 2>&1 | tail -3
python final/02_langgraph/01_simple_graph.py 2>&1 | tail -3
```

Expected：每个都正常输出 + 末尾的 ✅ 行（如果有）。

- [ ] **Step 6: Commit**

```bash
git add final/
git commit -m "refactor: 17 个 final/.py 用 _common 抽 boilerplate；qwen3.5-plus → qwen-plus；加 tutorial anchor 注释"
```

---

## Task 5: 建 _scratch/ 学习者主战场 + 修 .gitignore

**Files:**
- Modify: `.gitignore`
- Create: `_scratch/README.md`
- Create: `_scratch/journal/.gitkeep`

- [ ] **Step 1: 修 .gitignore**

在现有 .gitignore 末尾追加（用 Edit 工具，不要覆盖现有内容）：

```
# 学习者主战场——你写的代码不进 git
_scratch/*
# 但保留 README 和 journal 目录骨架
!_scratch/README.md
!_scratch/journal/
_scratch/journal/*
!_scratch/journal/.gitkeep
```

- [ ] **Step 2: 写 _scratch/README.md**

```markdown
# _scratch — 你的主战场

这个目录是给你自己写代码用的。每次跟 AI 对话产出的尝试性代码，都放这里。

## 怎么用

走 tutorial 时，每篇任务卡都会让你在这里建文件，比如：

- `_scratch/my_01_hello.py` — 你写的 01_hello_llm 版本
- `_scratch/journal/2026-XX-XX-week1-01.md` — 当天的卡点日志

## 注意

- 这个目录除了本 README 和 `journal/.gitkeep` 都被 .gitignore 了——你的代码不会被推到 GitHub
- 如果想保留某段你写得特别好的代码，复制到仓库根目录之外的地方
```

- [ ] **Step 3: 建 journal 目录**

```bash
mkdir -p _scratch/journal
touch _scratch/journal/.gitkeep
```

- [ ] **Step 4: 验证 .gitignore 行为**

```bash
echo "test" > _scratch/foo.py
git status -s _scratch/  # 应只看到 README + .gitkeep，不看 foo.py
rm _scratch/foo.py
```

Expected：`git status -s _scratch/` 只显示 README 和 `.gitkeep`，不显示 `foo.py`。

- [ ] **Step 5: Commit**

```bash
git add .gitignore _scratch/README.md _scratch/journal/.gitkeep
git commit -m "feat: 建 _scratch/ 学习者主战场（gitignore 排除代码、保留骨架）"
```

---

## Task 6: 建 tutorial/ 目录骨架 + week 索引

**Files:**
- Create: `tutorial/README.md`
- Create: `tutorial/week-1-langchain/`（空目录占位）
- Create: `tutorial/week-2-tools-and-agent/.gitkeep`
- Create: `tutorial/week-3-langgraph/.gitkeep`
- Create: `tutorial/week-4-langsmith-and-project/.gitkeep`

- [ ] **Step 1: 建目录**

```bash
mkdir -p tutorial/week-1-langchain
mkdir -p tutorial/week-2-tools-and-agent
mkdir -p tutorial/week-3-langgraph
mkdir -p tutorial/week-4-langsmith-and-project
touch tutorial/week-2-tools-and-agent/.gitkeep
touch tutorial/week-3-langgraph/.gitkeep
touch tutorial/week-4-langsmith-and-project/.gitkeep
```

- [ ] **Step 2: 写 tutorial/README.md**

```markdown
# Tutorial — 学习剧本

每篇 tutorial 是一份带任务卡 + 给 AI 的 prompt 的"学习剧本"。

**这不是文档，是脚手架**——你不读完它就懂，你按它跟 AI 对话产出代码，最后回头对比 `final/` 看自己写得怎样。

## 学习路线

| 周次 | 目录 | 概念 | 状态 |
|------|------|------|------|
| Week 1 | [week-1-langchain/](week-1-langchain/) | LangChain 核心：LLM、Prompt、LCEL、Memory、RAG | ✅ Ready |
| Week 2 | [week-2-tools-and-agent/](week-2-tools-and-agent/) | Tool & Agent | 🚧 待建 |
| Week 3 | [week-3-langgraph/](week-3-langgraph/) | StateGraph、条件边、HITL、多 Agent | 🚧 待建 |
| Week 4 | [week-4-langsmith-and-project/](week-4-langsmith-and-project/) | 追踪、评估、综合项目 | 🚧 待建 |

## 走法

1. 每周从最小编号开始（01_xxx.md）
2. 按"准备 → 任务卡 → 自检 → 卡点日志"顺序走
3. 卡 5 分钟以上 → 发 AI："给我一个再小一号的练习"
4. 通过自检 → 跳下一篇；不通过 → 留尾巴下次问，先继续

## 第一次跑这套？

去读 [HOW_TO_LEARN_WITH_AI.md](../HOW_TO_LEARN_WITH_AI.md)，再来这里。
```

- [ ] **Step 3: Commit**

```bash
git add tutorial/
git commit -m "feat: 建 tutorial/ 骨架 + week 索引（week-1 待填）"
```

---

## Task 7: 重写 README.md（30 秒入口）

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 备份原 README（可选，git 已有历史，跳过也行）**

跳过——`git log` 能找回。

- [ ] **Step 2: 用以下内容覆盖 README.md**

```markdown
# LangChain Tutorial Zero — 给零基础选手的 AI 辅助学习教程

> 不是另一份官方教程，是一份**学习脚手架**：你跟 Claude Code / Cursor 对话产出代码，再回头对照参考答案看自己写得如何。

## 这是什么

- 4 周从零学会 LangChain + LangGraph + LangSmith
- **核心方法**：每个知识点配「给 AI 的 prompt」+ 自检任务，让 AI 陪你拆代码
- **后端**：DashScope（通义千问）+ LangSmith Trace（都有免费层）

## 30 秒上手

```bash
git clone <fork 后你自己的 URL>
cd langchain-tutorial-zero

# 1. 跟着 SETUP.md 配 .env 和依赖
open SETUP.md

# 2. 读完元教程再开始（5 分钟）
open HOW_TO_LEARN_WITH_AI.md

# 3. 走第一篇 tutorial
open tutorial/week-1-langchain/01_hello_llm.md
```

## 仓库结构

```
├── HOW_TO_LEARN_WITH_AI.md   # 必读：怎么用 CC/Cursor 学陌生代码
├── SETUP.md                  # 必读：环境搭建
├── tutorial/                 # 学习剧本（你主要看这里）
│   └── week-1-langchain/  ... week-4-langsmith-and-project/
├── final/                    # 参考答案（任务卡指明何时偷看）
├── _scratch/                 # 你的主战场（自己代码放这）
└── docs/                     # 速查表 / 排错手册
```

## 适合你吗

✅ 适合：编程零基础或只懂语法、第一次碰 LangChain、想用 AI 工具学陌生代码、有 30+ 小时（4 周）
❌ 不适合：想找官方文档替代品、不想写代码只想读、英文资料够你看

## 这个仓库**不**做什么

- 不是另一份官方文档翻译
- 不教你 Python 基础（请先有 Python 入门，至少能看懂 `def foo():`）
- 不保证最新版 LangChain 兼容（pin 在 0.3.x，半年回头一次）

## 反馈与贡献

提 Issue 或 PR 都行。本项目基于 [MIT License](LICENSE)。
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: 重写 README，定位为零基础 + AI 工具学习脚手架"
```

---

## Task 8: 写 SETUP.md

**Files:**
- Create: `SETUP.md`

- [ ] **Step 1: 写 SETUP.md**

```markdown
# SETUP — 环境搭建

## 0. 你需要的

- Python 3.10+ （`python --version` 检查）
- 一个 [DashScope](https://dashscope.aliyuncs.com) 账号（阿里云通义千问，国内手机号注册即可，新号送免费额度）
- 一个 [LangSmith](https://smith.langchain.com) 账号（免费 5K runs/月，看 Trace 用）

## 1. 申请 API Key（10 分钟）

### DashScope（必填）
1. 登录 [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com)
2. 控制台 → API-KEY 管理 → 创建新 Key
3. 复制 Key（形如 `sk-xxxxxxxx`），放到下面 .env 里的 `DASHSCOPE_API_KEY`

### LangSmith（必填）
1. 登录 [smith.langchain.com](https://smith.langchain.com)
2. 右上头像 → Settings → API Keys → Create API Key
3. 复制（形如 `lsv2_xx...`），放到 .env 里的 `LANGCHAIN_API_KEY`

## 2. 配 .env

```bash
cp .env.example .env
# 用任何编辑器打开 .env，把两个 _here 占位符替换成真实 Key
```

最少要填的：
- `DASHSCOPE_API_KEY`
- `LANGCHAIN_API_KEY`

`LANGCHAIN_PROJECT=study` 是 LangSmith 项目名，所有 Trace 会聚在 `study` 项目里，可以保留默认。

## 3. 装依赖

```bash
# 推荐用虚拟环境（避免和系统 Python 包冲突）
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 4. 第一次烟雾测试

```bash
python final/01_langchain/01_hello_llm.py
```

期望看到：
- 三段输出（invoke / stream / batch）
- 末尾 `✅ 运行完毕！请前往 ...`

如果看到 LLM 真的回答了「什么是 LangChain？」，那环境就 OK 了。

## 5. 看一眼 Trace（验证 LangSmith 通了）

打开 [smith.langchain.com](https://smith.langchain.com)，左侧 Projects → `study`，应该看到刚才那次跑的 Trace 记录（含每次 LLM 调用、token 用量、耗时）。

## 排错

| 报错 | 多半原因 | 处理 |
|------|----------|------|
| `KeyError: 'DASHSCOPE_BASE_URL'` | .env 没加载 / 没填完 | 检查 .env 是不是放在仓库根目录，是不是写完整 |
| `401 Unauthorized` | DashScope key 错 | 重新去 DashScope 控制台拷一遍 |
| `ModuleNotFoundError: langchain_xxx` | 依赖没装全 | `pip install -r requirements.txt --upgrade` |
| `Connection error` | 网络问题 / 代理 | .env 取消注释 `HTTPS_PROXY` 或 ping `dashscope.aliyuncs.com` |
| LangSmith 没看到 Trace | `LANGCHAIN_TRACING_V2` 没设 / API key 错 | 检查 .env 里这两项 |

实在搞不定？发 Issue 时贴：你的 OS、Python 版本、完整报错（删掉 key）。
```

- [ ] **Step 2: Commit**

```bash
git add SETUP.md
git commit -m "docs: 写 SETUP.md（含申 key、装依赖、烟雾测试、排错表）"
```

---

## Task 9: 写 HOW_TO_LEARN_WITH_AI.md（元教程）

**Files:**
- Create: `HOW_TO_LEARN_WITH_AI.md`

- [ ] **Step 1: 写 HOW_TO_LEARN_WITH_AI.md**

```markdown
# HOW TO LEARN WITH AI — 怎么用 Claude Code / Cursor 学陌生代码

> 5 分钟读完。读完之前别开始走 tutorial。

## 这套教程的底层假设

零基础选手看陌生代码，会同时卡两件事：

1. **「这一行做什么」**——文档能查，慢但能解
2. **「为什么要这样写」**——文档不写，只有跟资深开发者对话才能解

AI 工具（Claude Code / Cursor）的本质优势是「陪问陪改」——但你不知道问什么。

所以这套 tutorial 给你预制问题清单：你只要按任务卡复制粘贴 prompt 给 AI，就能把代码问明白。

## 三个角色，分清

| 角色 | 是谁 | 干嘛的 |
|------|------|--------|
| 学习者 | 你 | 复制 prompt 给 AI、写代码、自检、记卡点日志 |
| AI 工具 | Claude Code / Cursor | 陪问、解释、纠错——**不许直接给完整代码** |
| `final/` | 参考答案 | 任务卡指明何时偷看；偷看是用来对比，不是用来抄 |

## 七条 prompt 心法

### 1. 不问"X 是什么"，问"X 跟我已知的 Y 有什么共同点"

❌ 不好：「ChatPromptTemplate 是什么？」
✅ 好：「ChatPromptTemplate 跟我熟悉的 Python f-string / 邮件模板有什么共同点和不同点？」

为什么：AI 用类比解释，你才能"挂"在已知概念上记住。

### 2. 不让 AI 直接给代码，先让它列大纲

❌ 不好：「帮我写一个 demo_invoke 函数」
✅ 好：「帮我列大纲：demo_invoke 函数大概有几步？每步干嘛？等我说 OK 再给代码」

为什么：你先看清结构，再看实现，能区分「设计问题」和「写法问题」。

### 3. 报错先描述"我以为会发生什么"，再贴报错

❌ 不好：「我跑代码报错了，[贴 traceback]」
✅ 好：「我跑 _scratch/my_01_hello.py，我以为会输出 `1+1=2` 之类，结果报错：[贴 traceback]。请别直接修——给我 3 个候选原因让我猜哪个最可能」

为什么：你的"以为"比报错更值钱——它暴露你的心智模型在哪卡住。

### 4. 卡 5 分钟以上必须开新 prompt

旧对话堆太多上下文，AI 会变笨；你也容易跟着 AI 跑偏。

发现卡住超 5 分钟 → 关掉当前对话 → 重写一句精简的 prompt → 从头开始。

### 5. 让 AI 用日常类比，不堆术语

每个 prompt 后面加一句：

```
请用日常类比解释。不要堆术语。回答 200 字内。
```

技术解释从日常类比开始——从微信群、邮件模板、流水线、菜谱这种零基础也懂的概念出发，再升级到术语。

### 6. 让 AI 单次只引一个问题

零基础选手最容易被 AI"一口气问 5 个问题"打懵。

每个 prompt 加：

```
每次只问我一个问题，等我回答你再继续。
```

### 7. 自己写完再让 AI 对比 final

✅ 好：「我自己写了 _scratch/my_01_hello.py（[贴代码]），请对比 final/01_langchain/01_hello_llm.py：哪里是『风格差异』，哪里是『真错』？真错的地方告诉我『为什么这样写更好』，**不要直接给修改后代码**——让我自己改」

为什么：自己改一次，记住率比抄代码高 5-10 倍。

## 怎么用 Claude Code

```bash
# 1. 安装（如果还没）
brew install anthropic/claude/claude        # macOS
# 或下载 desktop app from claude.ai/download

# 2. cd 到仓库根目录
cd langchain-tutorial-zero

# 3. 启动
claude

# 4. 把任务卡里的 prompt 复制粘贴进对话框
```

Claude Code 默认能读你仓库里所有文件，所以 prompt 里写「请看 final/01_langchain/01_hello_llm.py 第 14-25 行」是直接生效的。

## 怎么用 Cursor

```
1. 下载安装：cursor.com
2. File → Open Folder → 选 langchain-tutorial-zero
3. 关键快捷键：
   - Cmd+L (Mac) / Ctrl+L (Win)：打开右侧 chat 对话框（"陪问"模式）
   - Cmd+K：在编辑器里弹小输入框（"行内修改"模式）

主要用 Cmd+L——就是把任务卡的 prompt 粘进右侧 chat。
@文件名 可以让 Cursor 把文件内容一并喂给 AI（比如 `@final/01_hello_llm.py`）
```

## AI 给的代码跑不通时——3 步诊断

### Step 1: 别急着复制粘贴报错

先自己看 5 秒报错最后一行：是 `ImportError`、`KeyError`、`AttributeError` 还是别的？

### Step 2: 描述"我以为会发生什么"

```
我跑 _scratch/foo.py，我以为会打印 "hello"，结果看到 [报错最后一行]。
我猜可能是 [你的猜测]。这猜对了吗？如果不对，再给我 1 个候选原因。
```

### Step 3: 还是不行，最小复现

把代码砍到最少（5-10 行）能复现同一个错。砍掉无关后大概率你自己就能看出来。砍完还看不出来，把砍后的最小版贴给 AI。

## 每周末「卡点日志」复盘

走完一周后，打开 `_scratch/journal/`，回答：

1. 这周哪几个任务卡走得最顺？为什么？
2. 哪几个最卡？卡在哪？
3. 哪个 prompt 让 AI 给出了最好的解释？（贴下来当下周复用）
4. 我现在能用一句话讲明白「LCEL / LangChain Memory」吗？讲不出 = 没真懂，下周回头补

## 准备好了？

去 [tutorial/week-1-langchain/01_hello_llm.md](tutorial/week-1-langchain/01_hello_llm.md) 走第一篇。
```

- [ ] **Step 2: Commit**

```bash
git add HOW_TO_LEARN_WITH_AI.md
git commit -m "docs: 写 HOW_TO_LEARN_WITH_AI 元教程（七条 prompt 心法 + CC/Cursor 用法）"
```

---

## Task 10: 写 tutorial/week-1-langchain/01_hello_llm.md（gold standard）

**Files:**
- Create: `tutorial/week-1-langchain/01_hello_llm.md`

这是**标杆 tutorial**——之后所有 tutorial 都模仿它的结构。任务卡密度最高（6 个），prompt 最详细，挖空最多。

- [ ] **Step 1: 写 01_hello_llm.md**

完整内容：

````markdown
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

1. 打开 `final/01_langchain/01_hello_llm.py`，**只看创建 llm 的那段**（大概 14-21 行）
2. 把那段复制到 `_scratch/my_01_hello.py`
3. **关掉 final 文件**，别再偷看

**给 AI 的 prompt**：

```
我在 _scratch/my_01_hello.py 里只复制了 LLM 实例化那段代码：

[把你刚复制的那几行贴这里]

请引导我理解，不要直接给答案：
1. ChatOpenAI 的每个参数（model, base_url, api_key, temperature）分别是干嘛的？
   请逐个让我猜一遍，等我答完你再告诉我对不对。
2. 如果我不写 base_url，会发生什么？我能预测吗？

每次只问我一个问题，等我回答再继续。
```

**自检**：你能说清「为什么连 DashScope 必须写 base_url」（答案藏在 OpenAI 兼容协议里）。

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
````

- [ ] **Step 2: 自己走一遍当作 smoke test（10-15 分钟）**

按 tutorial 步骤实际走一遍：

1. 跑 `python final/01_langchain/01_hello_llm.py` 验证 final 能跑（之前已验，复核）
2. 把任务 1 的 prompt 复制出来，模拟一下：「这个 prompt 真的指向 final 文件第 14-21 行吗？」——核对行号是否准
3. 检查所有相对路径（`../../final/...`、`../../SETUP.md`）是否都能在 GitHub 渲染下点开
4. 自检通关条件每条是不是真的可机械验证

发现问题就改 tutorial 文件。

- [ ] **Step 3: Commit**

```bash
git add tutorial/week-1-langchain/01_hello_llm.md
git commit -m "docs: 写 week-1/01_hello_llm.md（gold standard 标杆 tutorial）"
```

---

## Task 11: 写 tutorial/week-1-langchain/02_prompt_template.md

**Files:**
- Create: `tutorial/week-1-langchain/02_prompt_template.md`

按 01 的标准模板写。具体内容指引：

**本步带回家的概念**：Prompt 模板把「模板字符串 + 变量」分开，方便复用；ChatPromptTemplate 在此基础上加了角色区分。

**配套代码**：`final/01_langchain/02_prompt_template.py`

**任务卡数量**：5-6 个微任务（密度仍然高，因为还在 week 1 早期）

**核心覆盖**：
1. 跑 final 看四段输出（PromptTemplate / ChatPromptTemplate / FewShot / partial）
2. 类比理解：模板像邮件模板？像 f-string？区别？
3. 挖空写 PromptTemplate 一段
4. 自己写 ChatPromptTemplate（多角色）
5. 玩 partial：固定一半变量
6. 自检对比

**关键 prompts**（必须用这些原文）：

prompt 1（任务 1，类比理解）：
```
我刚跑了 final/01_langchain/02_prompt_template.py，看到 4 段：
PromptTemplate、ChatPromptTemplate、FewShotChatMessagePromptTemplate、Partial。

请用日常类比帮我搞清 3 件事：
1. PromptTemplate 跟 Python 的 f-string 比，多了什么？为什么不直接用 f-string？
2. ChatPromptTemplate 在 PromptTemplate 基础上加了什么？什么时候非用它不可？
3. FewShot 是什么场景才需要？请给我一个生活里的"少样本"类比（比如教小孩做题先举几个例子？）

回答 300 字内，每问独立段落。不要堆术语。
```

prompt 2（任务 3，挖空写）：
```
我在 _scratch/my_02_prompt.py 里只写了 PromptTemplate.from_template(...) 那一行：

[贴你的代码]

请引导我（不要直接给答案）：
1. 把这个 template 渲染成字符串，要调什么方法？
2. 把它接到 llm 后面变成"链"，LCEL 用什么符号？为什么是这个符号？
3. 链的结尾通常加 StrOutputParser()，这一步是干嘛的？不加会怎样？

每次只引一个问题。
```

prompt 3（任务 6，自检）：
```
我写的 ChatPromptTemplate 版本：
[贴你的代码]

参考答案 final/01_langchain/02_prompt_template.py 的 demo_chat_prompt_template：
[贴 final]

请分清"风格差异"和"真问题"——真问题告诉我为什么这样写更好，但不给修改后代码。
```

**通关条件**：
- 自己版能跑通 PromptTemplate + ChatPromptTemplate 两条链
- 能讲清「为什么用 LCEL 的 `|` 而不是嵌套调用」
- 至少跑过一次带 `partial` 的链

- [ ] **Step 1: 按上述指引写 02_prompt_template.md**

按 01_hello_llm.md 完全相同的章节结构（准备 / 任务卡 / 通关条件 / 卡点日志 / 通往下一站），把上面 prompts 嵌入对应任务里，补足任务卡的"做什么"步骤说明。

- [ ] **Step 2: 自走 smoke test**

模拟"零基础按这个走一遍"，重点验证：
- 所有 prompts 能直接复制粘贴到 CC 用
- 所有相对路径都能点
- 通关条件可机械验证

- [ ] **Step 3: Commit**

```bash
git add tutorial/week-1-langchain/02_prompt_template.md
git commit -m "docs: 写 week-1/02_prompt_template.md"
```

---

## Task 12: 写 tutorial/week-1-langchain/03_chains.md

**Files:**
- Create: `tutorial/week-1-langchain/03_chains.md`

**本步带回家的概念**：LCEL（LangChain Expression Language）让你用 `|` 把组件串成链；每个组件都实现统一接口，所以可以自由拼。

**配套代码**：`final/01_langchain/03_chains.py`

**任务卡数量**：4-5 个（密度开始降，因为已经走完 2 篇）

**核心覆盖**：
1. 跑 final 看链路输出
2. 拆开看：`prompt | llm | parser` 三段每段是啥
3. 加 RunnableLambda 做自定义逻辑
4. 链路并行 / 分支
5. 自检对比

**关键 prompts**：

prompt 1（拆链）：
```
LCEL 的 `prompt | llm | parser` 用了 Python 的 `|` 运算符。请回答：
1. 这个 `|` 跟 Unix shell 的管道（cat | grep）有什么共同点？
2. 链里每个组件都得实现什么"接口"才能被 `|` 串起来？（提示：跟 Java interface / 鸭子类型有关）
3. 如果我想在链中间加一段"自定义逻辑"（比如把 LLM 输出转大写），用什么类？

每问 2-3 句话。日常类比开始。
```

prompt 2（自己拼链）：
```
我在 _scratch/my_03_chains.py 里要拼一条链：
- prompt：让 LLM 用 50 字介绍一个城市
- llm：调通义千问
- 自定义函数：把回答转大写
- parser：取最后字符串

请引导我（不直接给代码）：
1. 这个自定义函数应该用 RunnableLambda 包一下还是直接写函数？
2. 链的顺序怎么排？哪些可以颠倒？哪些不能？

每次只引一问。
```

**通关条件**：
- 自己写一条 ≥ 4 段的链能跑通
- 至少包含一个 RunnableLambda
- 能讲清「为什么 LCEL 比手写 invoke 链好维护」

- [ ] **Step 1: 写 03_chains.md（按 01 模板，套上面 prompts 和覆盖范围）**
- [ ] **Step 2: 自走 smoke test**
- [ ] **Step 3: Commit**

```bash
git add tutorial/week-1-langchain/03_chains.md
git commit -m "docs: 写 week-1/03_chains.md"
```

---

## Task 13: 写 tutorial/week-1-langchain/04_memory.md

**Files:**
- Create: `tutorial/week-1-langchain/04_memory.md`

**本步带回家的概念**：多轮对话需要"上下文记忆"——本质是把历史消息每次都塞进 prompt。LangChain 的 `RunnableWithMessageHistory` 帮你管这件事。

**配套代码**：`final/01_langchain/04_memory.py`

**任务卡数量**：4-5 个

**核心覆盖**：
1. 跑 final 看多轮对话效果
2. 反例先做：写一段**没有 memory** 的多轮对话，亲眼看 LLM"失忆"
3. 加 RunnableWithMessageHistory，对比效果
4. 玩不同的 history store（dict / file）
5. 自检

**关键 prompts**：

prompt 1（反例理解）：
```
请帮我用日常类比讲清两件事：
1. 为什么 LLM 默认是"无状态"的？跟一个客服每次接电话都不知道你是谁有什么共同点？
2. RunnableWithMessageHistory 帮你做了什么"中间人"工作？把它比喻成什么职业？

200 字内，类比开始。
```

prompt 2（亲眼看失忆）：
```
帮我设计一个最小代码：
- 跑两轮对话
- 第一轮："我叫小明"
- 第二轮："我叫什么？"
- 不加任何 memory，看 LLM 怎么答

约束：
- 先列大纲（这个代码 5 行还是 10 行？）
- 等我说 OK 再给代码
- 不要直接加 memory
```

跑完看 LLM 真的不记得，才有体感。

**通关条件**：
- 跑通"无 memory"和"有 memory"两版，对比效果
- 能讲清"history store 在多用户场景为什么必须按 session_id 区分"

- [ ] **Step 1: 写 04_memory.md**
- [ ] **Step 2: smoke test**
- [ ] **Step 3: Commit**

```bash
git add tutorial/week-1-langchain/04_memory.md
git commit -m "docs: 写 week-1/04_memory.md"
```

---

## Task 14: 写 tutorial/week-1-langchain/05_rag_basic.md

**Files:**
- Create: `tutorial/week-1-langchain/05_rag_basic.md`

**本步带回家的概念**：RAG（Retrieval-Augmented Generation）= 把外部文档切片 → 向量化 → 检索相关片段 → 喂给 LLM 当上下文。解决"LLM 不知道你公司私有数据"的问题。

**配套代码**：`final/01_langchain/05_rag_basic.py`

**任务卡数量**：4 个（week 1 末尾，密度再降一点）

**核心覆盖**：
1. 跑 final 看 RAG 效果（用一份文档问 LLM 答里面的内容）
2. 拆 RAG 4 步骤：切片 / 向量化 / 检索 / 增强
3. 改 chunk_size，看检索质量怎么变
4. 自检 + 思考"什么场景用 RAG / 什么场景不该用"

**关键 prompts**：

prompt 1（4 步骤拆解）：
```
RAG 有 4 个核心步骤：
1. 文档切片（splitter）
2. 向量化（embeddings）
3. 检索（retriever）
4. 增强生成（prompt 拼接 + LLM）

请用"找资料写报告"的日常类比，把这 4 步分别比喻成做报告时的什么动作。
比如切片可能像"把一摞书撕成一页页"？

每步 1-2 句话。
```

prompt 2（chunk_size 直觉）：
```
RecursiveCharacterTextSplitter 的 chunk_size 参数：
- 太小（100 字符）会怎样？
- 太大（10000 字符）会怎样？
- 默认值大概多少？为什么是这个量级？

请用"读书做笔记"的类比讲。每点 1-2 句。
```

**通关条件**：
- 自己改 chunk_size 跑两次，能对比说出差别
- 能讲清"RAG 适合什么 / 不适合什么"（提示：知识更新频繁的适合，需要推理跨多文档的不适合）

- [ ] **Step 1: 写 05_rag_basic.md**
- [ ] **Step 2: smoke test**
- [ ] **Step 3: Commit**

```bash
git add tutorial/week-1-langchain/05_rag_basic.md
git commit -m "docs: 写 week-1/05_rag_basic.md（week-1 五篇标杆完成）"
```

---

## Task 15: Phase 1 收尾——更新 tutorial/README.md 状态 + 全量自检

**Files:**
- Modify: `tutorial/README.md`

- [ ] **Step 1: 把 tutorial/README.md 的 week-1 状态从 ✅ Ready 改成 ✅ Done (5/5)**

```markdown
| Week 1 | [week-1-langchain/](week-1-langchain/) | LangChain 核心：LLM、Prompt、LCEL、Memory、RAG | ✅ Done (5/5) |
```

- [ ] **Step 2: 全量自检 — 跑过所有 5 个 final**

```bash
for f in final/01_langchain/{01,02,03,04,05}_*.py; do
  echo "=== $f ==="
  python "$f" 2>&1 | tail -2
done
```

Expected：每个都正常输出（最后一行能看到 ✅ 或正常 print）。

- [ ] **Step 3: 全量自检 — 检查所有 tutorial 链接**

```bash
python -c "
import re, pathlib
broken = []
for p in pathlib.Path('tutorial/week-1-langchain').glob('*.md'):
    text = p.read_text()
    # 找所有 markdown 链接
    for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
        link = m.group(2)
        if link.startswith(('http', '#', 'mailto')): continue
        target = (p.parent / link).resolve()
        if not target.exists():
            broken.append(f'{p.name}: {link} → 404')
print('\n'.join(broken) if broken else 'all links ok')
"
```

Expected：`all links ok`（如果有 broken，回去修对应 tutorial）

- [ ] **Step 4: Commit + push（如果用户授权）**

```bash
git add tutorial/README.md
git commit -m "chore: week-1 五篇 tutorial 完成，更新索引状态"
```

不主动 push（按 CLAUDE.md 行为底线，push 前要问）。

---

## Self-Review 完成标记

✓ Spec 覆盖：Phase 0（Task 1-9）+ Phase 1（Task 10-15）全覆盖；Phase 2-3 明确延后到下一计划
✓ 占位符扫描：全部任务都有具体代码 / 命令 / 文件路径
✓ 类型/命名一致性：`make_llm()` / `final/_common.py` / `qwen-plus` / `tutorial/week-1-langchain/01_hello_llm.md` 全文统一
✓ 风险有覆盖：Task 4 Step 2 处理了 sys.path 兜底；Task 5 Step 4 验证 .gitignore；Task 15 跑全量回归

---

## 下一计划（Phase 2 + 3 预告）

完成本计划后，开新计划覆盖：
- Phase 2：week-2 / week-3 / week-4 共 ~12 篇 tutorial（密度逐周递减，week-4 capstone 最少 prompt）
- Phase 3：docs/prompts-cheatsheet.md / docs/debug-recipes.md / 真实零基础走 week-1 自测 / push to GitHub
