# langchain-tutorial-zero 长远定位与优化设计

> 状态：草稿（spec），待用户 review 后转 plan
> 创建：2026-05-30

## 核心目标

把这个项目从「个人 14⭐ 教程 repo」升级到「能写进简历 + 能被中文 awesome list 收录的可信中文 LangChain 教程」，不追求短期涨星，追求**简历可信度** + **长期可维护**。

## 现状盘点

- 14 ⭐ / 0 fork / 53 commits / 最新 5月活跃
- owner: `estelledc` (zx775004276@qq.com)；本地 git 用户 `estelledc` —— **不是 jason 的主 GitHub 账号**
- 内容：14 篇 tutorial（4 周）+ 5 个 docs（concepts/prompts-cheatsheet/debug-recipes/challenges/test-runs）+ HOW_TO_LEARN_WITH_AI.md 元教程 + 14 个实测 final/.py
- 已有：MIT LICENSE / CHANGELOG / CONTRIBUTING / issue templates / GitHub Pages
- 缺：作者背书段 / 可视化路线图 hero 图 / 差异化定位声明 / Quick Start 可跑代码块 / 每节预估时长 / Colab 入口

## 调研结论（2026-05-30 14 个对标项目）

**反直觉发现**：1k+ ⭐ 项目都靠 ① 机构品牌（microsoft / datawhale）② 明星作者+书/视频（mlabonne / gkamradt）③ first mover SEO（liaokongVFX 2023-04 卡时间窗口）。Jason 三条都没有 → **不追涨星，追"被收录"和"简历可信度"**。

**关键对标**：
- liaokongVFX (9k⭐)：单 README + GitBook，开篇一句话痛点钩子；缺 LCEL/Runnable，已停更
- datawhalechina/llm-universe (13k⭐)：双形态发布 + 国内 API 抹平 + 主线项目贯穿
- mlabonne/llm-course (79k⭐)：可视化 roadmap 图本身就是最大获客资产
- microsoft/*-for-beginners：「Learn + Build」双标签 + 编号路径
- iparesh18/Learn-LangChain (5⭐)：**反面教材** — 结构合理但作者无背书

**借鉴清单**（按 ROI 排）：
1. 路线图 PNG/SVG hero 图（mlabonne 模式）
2. 作者介绍段建立 trust（iparesh18 失败教训）
3. 每节预估时长（所有对标都缺，低成本差异化）
4. Quick Start 60 秒可跑代码块
5. Learn / Build 双标签（microsoft 模式）
6. 中文 awesome list 收录 PR
7. Colab 一键跑徽章（pinecone-io 模式）

**不抄**：microsoft 七件套（pre-quiz/post-quiz/...）—— 会冲淡 AI 辅助学习这个差异化卖点。

## 差异化定位（一句话）

> "唯一系统教 AI 辅助学习心法的中文 LangChain 教程，覆盖 1.3.x 实测，4 周 14 篇 learning-by-doing"

三个不可替代的卖点：
- **AI 辅助学习元教程**（HOW_TO_LEARN_WITH_AI.md 七条心法）—— 没有第二份中文教程做这个
- **LangChain 1.3.x 实测**（docs/test-runs.md 6 处破坏性变更修法）—— liaokongVFX 已停更，datawhale 偏 RAG 不深 LangChain
- **任务卡 + 给 AI 的 prompt 结构** —— learning-by-doing 而非文档罗列

## 三层路径设计

### Phase 1：可信度补建（4-6 小时，立刻做）

让 README 在第一屏就过 trust check。
- README 顶部加「为什么是我」段
- 加可视化路线图 hero 图（PNG 导出 mermaid）
- 加 60 秒 Quick Start 可跑代码块
- 顶部加差异化定位句

**产出**：README v2 + hero 图，commit 后 push。

### Phase 2：被收录（2-4 小时，跟在 Phase 1 后）

让外部 list 愿意收录。
- 准备 awesome-list 提交描述（一句话定位 + 14 ⭐ 现状 + 差异化）
- 提 PR 到至少 2 个中文 awesome list
- 候选：[awesome-chinese-llm](https://github.com/HqWu-HITCS/Awesome-Chinese-LLM)、[awesome-chatgpt-zh](https://github.com/yzfly/awesome-chatgpt-zh)

**产出**：≥1 个 list 收录或至少 PR 已开。

### Phase 3：长期维护节奏（持续，每月 ~2 小时）

让项目保持「活的」状态而非快速过期。
- LangChain 月度 release notes 监控（fork 版本兼容性）
- final/ smoke test 每季度跑一次
- CHANGELOG 诚实标注「已过时段落」
- 每篇 tutorial 加每节预估时长（一次补完，后续维护少）

**产出**：维护节奏文档化在 `docs/maintenance.md`。

## 不做什么（明确边界）

- **不做** 英文版翻译 —— 14⭐ 没底气推 awesome-langchain，等被中文 list 收录后再考虑
- **不做** B 站配套视频 —— 投入大效果不确定
- **不做** owner 迁移到 jason 账号 —— 反而需要一篇博客解释「以前用 estelledc 账号，现在迁过来」会更复杂；维持现状，简历段直接写「主要作者」即可
- **不做** 抄 microsoft 七件套 —— 保留 AI 辅助学习这个真差异化
- **不做** 短期内冲 50+ ⭐ —— 涨星没有快路径，能涨就涨，重点在被收录

## 简历段（成果模板）

按 Phase 1+2 完成后，简历可写：

> **LangChain Tutorial Zero**（开源教程项目，[GitHub](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial)）
> - 主要作者；4 周 14 篇 learning-by-doing 中文教程，覆盖 LangChain / LangGraph / LangSmith
> - 创新点：AI 辅助学习元教程（七条 prompt 心法 + 任务卡结构），CLAUDE.md 自动加载教学约束
> - 工程：14 个独立可执行 .py 全部经 LangChain 1.3.x 实测，覆盖 6 处破坏性变更修法
> - 收录：被 [list 名] 中文 LLM 学习资源 list 收录

## Owner 归属处理

git remote 是 `estelledc/...`，commit author 是 `estelledc <zx775004276@qq.com>`。简历表述用「主要作者 / 主要贡献者」即可（estelledc 是 Jason 的另一个账号，可在简历或面试中说明）。**不做** 仓库迁移 —— 迁移会丢 stars、丢 issue 历史，付出大于收益。

## 已确认的决策（2026-05-30）

1. estelledc 就是 Jason 自己的 GitHub 账号 → 简历归属直接写「作者」即可，无需迁移仓库
2. 不做英文版（保留为未来选项，等被中文 list 收录后再评估）
3. Phase 1 立刻启动，Phase 2 跟在后面

## 后续

spec 通过 review 后转为 implementation plan（superpowers:writing-plans），逐项推进。
