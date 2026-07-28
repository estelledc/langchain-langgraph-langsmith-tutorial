# LangChain Tutorial Zero — 重构设计 spec

> 日期：2026-05-29
> 状态：等待评审
> 来源仓库：[estelledc/langchain-langgraph-langsmith-tutorial](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial)

## 1. 背景与问题

原仓库是一份「LangChain → LangGraph → LangSmith」4 周学习计划，17 个 `.py` 文件，每个文件采用「顶部注释知识点 + demo_xxx 函数 + main 跑」的官方教程文风。

零基础学习者跟着走，会同时卡两个点：
1. 「这一行做什么」——能查文档解决，慢但能解
2. 「为什么要这样写」——文档不写，只有跟资深开发者对话才能解

AI 工具（Claude Code / Cursor）的本质优势是「陪问陪改」——但学习者不知道问什么。所以重构的核心目标，是把「被动看代码」翻成「主动跟 AI 对话」，并**预制问题清单**让学习者不至于卡死。

## 2. 受众与约束

- **目标受众**：零基础选手，可发布给别人 fork 来走（不只 Jason 自用）
- **学习姿势**：渐进处理——前期任务卡 prompt 多 + 代码片段小（挖空式），后期任务卡 prompt 少 + 给完整代码（通读式）
- **AI 工具覆盖**：Claude Code + Cursor 同时支持
- **改造权限**：从底层重构，不必拘泥原仓库目录形态

## 3. 选定方案：双轨 + 任务卡

被排除的方案：
- A 增量分层包（拆 50+ 微 step）：极致渐进但文件数膨胀，与「后期读为主」姿势冲突
- C Notebook：交互流畅但与 CC/Cursor 集成差，环境门槛高，发布性差

选定 B 的理由：
- 跟「渐进姿势」对齐：用 prompt 密度而非文件数实现渐进
- 可发布门槛低：fork 后只要装 requirements + 跟着 tutorial/01 走
- 维护量可控：~17×2 文件 + 2 份元教程

## 4. 目录与角色分工

```
langchain-tutorial-zero/
├── README.md                  # 30 秒入口：你在哪 / 该读哪 / 跳到 SETUP
├── HOW_TO_LEARN_WITH_AI.md    # 元教程：怎么用 CC/Cursor 学这套
├── SETUP.md                   # 环境搭建（DashScope key、venv、依赖）
├── .env.example
├── requirements.txt
│
├── tutorial/                  # 学习剧本（按周组织，唯一学习入口）
│   ├── week-1-langchain/         01_hello_llm.md ... 05_rag_basic.md
│   ├── week-2-tools-and-agent/   06_tools_agent.md + 综合练习
│   ├── week-3-langgraph/         01_simple_graph.md ... 04_multi_agent.md
│   └── week-4-langsmith-and-project/  01_tracing.md ... 04_capstone.md
│
├── final/                     # 成品代码（参考答案，最后才偷看）
│   ├── 01_langchain/  ... 04_project/   # 沿用原结构，做兼容性微调
│   └── _common.py             # 共享 boilerplate（load_dotenv 等）
│
├── _scratch/                  # 学习者主战场（gitignore）
│   ├── README.md              # 提醒：在这里跟 AI 一起写自己的版本
│   └── journal/               # 卡点日志
│
└── docs/
    ├── prompts-cheatsheet.md  # 高频 prompt 速查
    ├── debug-recipes.md       # 报错时的诊断 prompt 套路
    ├── concepts/              # 难点小词典（按需新增）
    └── superpowers/specs/     # 本 spec 所在
```

## 5. tutorial/*.md 标准模板

每个 tutorial 文件都按这个结构写，前期版本（week-1 前几篇）任务卡密度最高，往后递减。

```markdown
# 0X · <概念名> — 跟 AI 一起 <动作>
> 本步带回家的概念：<一句话>

## 准备 (5 min)
- 上一步通关了吗？打开 final/<对应 .py> 扫一眼，先别细看
- 打开 Claude Code 或 Cursor，cd 到本仓库

## 任务卡（前期 6-8 个微任务，后期 2-3 个长任务）
### 任务 N：<动作>
**做什么**：...
**给 AI 的 prompt**（直接复制）：

   > [可粘贴的 prompt 片段，约束 AI 用日常类比、不堆术语、单次只引一问]

## 自检
**对照 prompt**：把你的代码贴给 AI，让它对比 final/，分清"风格差异" vs "真错"

## 卡点日志（必填，去 _scratch/journal/ 记）
- 卡了多久 / AI 哪句解释让我"原来如此" / 想留作复用的 prompt

## 通往下一站
- 通过 → 下一篇 ; 没通过 → 发 AI："给我一个再小一号的练习"
```

**前期 vs 后期的渐进体现**：
- Week 1 前 3 篇：任务卡 6-8 个微任务，每任务带完整 prompt，挖空多
- Week 1 后 + Week 2：任务卡 4-6 个，prompt 开始让学习者自己改写
- Week 3：任务卡 3-4 个，给完整代码先读再改
- Week 4 capstone：任务卡 2-3 个长任务，主要是「给完整需求自己想 prompt」

## 6. HOW_TO_LEARN_WITH_AI.md 骨架

1. **这不是教程，是脚手架**——你跟 AI 对话产出代码，不是抄代码
2. **三个角色你要分清**：学习者（你）/ AI 工具 / final 目录的代码（最后偷看的参考答案）
3. **七条 prompt 心法**：
   - 不问「X 是什么」，问「X 跟我已知的 Y 有什么共同点」
   - 不让 AI 直接给代码，先让它列大纲
   - 报错先描述「我以为会发生什么」再贴报错
   - 卡 5 分钟以上必须开新 prompt
   - 让 AI 用日常类比，不堆术语
   - 让 AI 单次只引一个问题
   - 自己写完再让 AI 对比 final
4. **怎么用 Claude Code**（30 行讲明白核心姿势）
5. **怎么用 Cursor**（30 行：Cmd+K vs Cmd+L 区别等）
6. **AI 给的代码跑不通时**：3 步诊断
7. **每周末复盘**：卡点日志怎么写

## 7. 学习者一周真实路径（验证设计能跑通）

周一晚：
1. 打开仓库 README → 读 SETUP → 读 HOW_TO_LEARN
2. 跳到 `tutorial/week-1-langchain/01_hello_llm.md`
3. 做任务 1：`python final/01_langchain/01_hello_llm.py` 跑起来
4. 做任务 2：把 prompt 1 复制给 CC，让它解释 ChatOpenAI 是什么角色
5. 做任务 3：在 `_scratch/my_01_hello.py` 跟 AI 对话写自己版的 invoke
6. 自检：贴自己的代码给 AI 对比 final
7. 通过 → `_scratch/journal/2026-XX-XX.md` 写卡点日志

周二-周四同模式做 02-04；周五做 05；周六小结；周日跳到 week-2。

## 8. 原代码改动范围（最小化）

**修**：
- `qwen3.5-plus` → `qwen-plus`（DashScope 实际可用的模型 ID；3.5 这个 ID 不存在）
- 每个 `.py` 顶部加锚点注释 `# 配套教程：tutorial/week-X/0Y_xxx.md`
- 把每个文件重复 5 行的 `load_dotenv` boilerplate 抽出 `final/_common.py`
- README 重写为 30 秒入口（不再是教程主体）

**不改**：
- 核心 LCEL / Graph 写法（这是要学的目标）
- 17 个文件的概念覆盖范围（已经够好）
- requirements.txt 主体（按需补 1-2 个）

## 9. 验收标准（每个 tutorial 必须满足）

1. 一个真正零基础的人，光复制粘贴 prompt 就能跑通对应 final
2. 至少 1 处挖空练习（前期）+ 1 处「对比 AI 版」练习
3. 末尾有卡点日志锚点
4. 全文不超 400 行，不堆术语
5. 中文为主；术语首次出现配英文

**仓库级验收**：
- README → HOW_TO_LEARN → SETUP → tutorial/week-1/01 这条路径不超过 30 分钟能走完
- 任意一篇 tutorial 都能独立工作（前置假设明确写在「准备」段）
- `final/` 全跑通（CI 或本地手动 smoke test）

## 10. 实施阶段切片

**Phase 0 — 骨架与元教程（先行）**
- 改 README、写 HOW_TO_LEARN_WITH_AI、写 SETUP、写 .gitignore（_scratch/journal/）
- 建 tutorial/ 和 final/ 目录骨架，把原 17 .py 移到 final/
- 抽 `_common.py`，改 `qwen3.5-plus` → `qwen-plus`，加锚点注释

**Phase 1 — 标杆 tutorial（week-1 全部 5 篇）**
- 写 01_hello_llm.md 作为模板标杆（最高密度任务卡）
- 复盘标杆，确认模板 OK 后批量写 02-05
- 此时回头校 HOW_TO_LEARN 是否需要补「七条心法」案例

**Phase 2 — week-2 / week-3 / week-4 批量推进**
- 按相同节奏推进；每周收尾时调一次密度（避免后期任务卡过密）
- week-4 capstone 给最少 prompt，最像真实开发场景

**Phase 3 — 文档配套与发布**
- 写 docs/prompts-cheatsheet.md
- 写 docs/debug-recipes.md
- 跑一遍真实「零基础走 week-1」自测
- 改原仓库 README，加新仓库链接，建议把原仓库改名 archive

## 11. 风险与应对

| 风险 | 触发条件 | 应对 |
|------|----------|------|
| tutorial 写得太啰嗦 | 单文件 > 400 行 | 强制砍至 400 行内；多余 prompt 移到 docs/cheatsheet |
| AI 给的版本和 final 差太远 | 学习者自检过不了 | 在自检 prompt 里加「分清风格差异 vs 真错」的明确措辞 |
| 模型 / API 改名 | DashScope 改模型 ID | `final/_common.py` 集中管模型名，改一处 |
| LangChain 版本破坏性变更 | requirements.txt pin 失效 | requirements.txt 用 `>=` 而不是 `==`；CI 跑 smoke test |

## 12. 不做的事（YAGNI）

- 不做 bilingual（中英双语）——中文为主，英文术语首次注一下就够
- 不做 video / GIF 录屏——文字够用，维护成本低
- 不做配套小程序 / 网页——markdown + GitHub 渲染足够
- 不做 LangChain 版本兼容矩阵——pin 一个能跑的版本，半年回头一次
- 不做对比官方教程的逐行差异表——HOW_TO_LEARN 提一句「这不是官方教程的替代，是补充入口」
