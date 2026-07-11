# Maintenance — 长期维护节奏

> SoT：本文件描述本仓库的维护周期、触发条件和 SOP。
> 作者：Jason Xun，最后更新 2026-05-30。

## 设计原则

LangChain / LangGraph / LangSmith 三件套**版本迭代极快**（破坏性变更频率约每季度一次）。所以这个仓库的维护节奏不是"按月写新内容"，而是"按版本对齐 + 每年一次结构性回顾"。

**承诺**：

- 任何被 awesome-list 收录的版本，**至少保证 90 天内 final/.py 全跑通**
- 撞到破坏性变更而暂时跑不通，**24 小时内** README 顶部加 banner 警示
- 永久维持 `docs/test-runs.md` 的实测档案准确

**不承诺**：

- 不承诺追平 LangChain 的每个新功能（保留"教零基础"这个定位，加新功能反而稀释）
- 不承诺 issue 24 小时响应（个人项目，请耐心）

---

## 维护节奏一览

| 周期 | 触发 | 做什么 | 投入 |
|---|---|---|---|
| 周级 | LangChain 发布 minor release（1.4 → 1.5） | 在 LangChain CHANGELOG 扫"BREAKING CHANGES"，标关注点 | 5 分钟 |
| 月级 | 月初 | 跑一次 `scripts/smoke-test.sh`，更新 `docs/test-runs.md` | 30-60 分钟 |
| 季度 | 每季度第 1 周 | 完整跑批 14 个 final + 抽查 3 篇 tutorial 跟现行 API 对应 | 2 小时 |
| 年度 | 每年 12 月 | 结构性回顾：内容是否还匹配定位、要不要加新章节、是否要写英文版 | 半天 |
| 触发式 | 撞破坏性变更 | banner + issue 自我开 + PR 修复 | 视严重程度 |

---

## 周级：LangChain release 监控

### 操作

订阅这三个 GitHub releases：
- https://github.com/langchain-ai/langchain/releases
- https://github.com/langchain-ai/langgraph/releases
- https://github.com/langchain-ai/langsmith-sdk/releases

每个新 release 的 changelog 扫一遍 "BREAKING CHANGES" / "Removed" 章节。

### 判断"是否需要立刻动手"

| 信号 | 处理 |
|---|---|
| 改的是 `langchain.agents` / `pydantic` / `langsmith.evaluation` 等本仓 final 用到的模块 | 立刻进月级流程 |
| 改的是 langchain-experimental / 没 final 用 | 记到 [docs/test-runs.md](test-runs.md) 末尾"待跟进"段，下个月级合并处理 |
| Deprecation warning（不是 removal） | 周级仅记录，不修 |

---

## 月级：smoke test + 实测档案更新

### 操作

```bash
# 进项目
cd langchain-tutorial-zero
source .venv/bin/activate

# 一键跑批
bash scripts/smoke-test.sh
```

脚本会：
1. 跑 14 个独立可执行 final/.py（不含 `__init__.py` 和 04_project 的支持模块）
2. 每个文件 timeout 10 分钟
3. 输出 PASS/FAIL/SKIP + 耗时
4. 把结果写到 `docs/test-runs.md` 末尾"## 历次 smoke test 记录"段

### 期望结果

参考 `docs/test-runs.md` 的总览表。最近一次（2026-05-29）是：12 PASS / 1 PARTIAL（本机 SSL）/ 1 SKIP（本机 embedding 凭证）。

### 如果 PASS 数下降

1. 看具体哪个文件失败 + traceback
2. **先判断是不是破坏性变更**：grep traceback 关键词到 LangChain release notes
3. 如果是：进 [docs/test-runs.md](test-runs.md) 第一节加新条目，按 1.1-1.6 同样格式
4. 如果是凭证 / 网络：标 SKIP 不动代码
5. 修完重跑确认全 PASS

---

## 季度：结构性体检

每季度第 1 周（1月、4月、7月、10月）做：

### 1. 完整 smoke test
跟月级一样，但本季度第一次跑要**完全清环境**：
```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/smoke-test.sh
```

### 2. 抽查 3 篇 tutorial 跟现行 API
随机选 3 篇 `tutorial/week-X/*.md`，从头照着任务卡跟 AI 对话写一遍代码。如果遇到 API 不一致，更新 tutorial。

### 3. 更新 README 顶部的"实测"声明
README 里 `LangChain 1.3.x 实测` 这句要跟上当前版本。
- 如果 final 已经在 1.4.x 跑通：改成 `1.4.x 实测`
- 如果 1.4.x 撞了破坏性变更但还没改：保留 1.3.x，但加 banner

### 4. 检查 PR 状态
- `AiHubCN/Awesome-Chinese-LLM#102` — 看是否合并
- `WangRongsheng/awesome-LLM-resources#127` — 看是否合并
- 合并后 README 顶部加"被 X⭐ list 收录"徽章

---

## 年度：结构性回顾（每年 12 月）

回答这 5 个问题：

1. **定位还匹配吗？** 「零基础 LangChain 1.x 中文教程」这个定位还是当前缺口吗？还是已经有更好的项目占位？
2. **该加新章节吗？** Agent 模式 / RAG 进阶 / 多模态等，年内有没有什么新东西重要到值得加一周？
3. **该写英文版吗？** 看 star 数、issue 国际占比、awesome-langchain 收录政策是否变化
4. **owner 归属要变吗？** 如果当时 estelledc 这个账号变成不方便用，考虑迁移
5. **下一年的承诺？** 续约 90 天兼容承诺，还是要降到 60 天 / 30 天

回答写到 `docs/superpowers/year-review-YYYY.md`。

---

## 触发式：撞破坏性变更的紧急流程

### 24 小时内

1. README 顶部加 banner：

```markdown
> ⚠️ **维护中**：LangChain X.Y 引入了破坏性变更（[详情](#)），final/X.py 当前跑不通，预计 N 天内修复。
```

2. 在自己仓库开 issue：[BREAKING] LangChain X.Y - what broke
3. label: `breaking-change`

### 修复时

1. 进 [docs/test-runs.md](test-runs.md) 一节加新破坏性变更条目（保持 1.1-1.6 风格）
2. 改对应 final/.py
3. 重跑 smoke test 全 PASS
4. 改 requirements.txt pin 到能跑通的版本
5. 移除 README banner
6. CHANGELOG.md 加 PATCH 版本（如 v1.0.1）

---

## 信号：什么时候考虑放弃维护

如果以下任何一条成立，**坦诚在 README 顶部声明"已停止主动维护"**而不是假装还在维护：

- 连续 2 个月没跑过 smoke test
- LangChain 进入下一个 major version（如 2.0），适配工作量超过 8 小时不愿投入
- 个人主线已经离开 LLM/AI 领域

停止维护后：
- README 顶部加"⚠️ 已停止维护，建议参考 [其他活跃中文 LangChain 教程]"
- 不删仓库，保留作为简历素材
- 把维护权 transfer 给愿意接的人（issue 征集）

诚实承认 > 慢性烂尾。

---

## 工具清单

| 工具 | 路径 | 用途 |
|---|---|---|
| smoke test | [scripts/smoke-test.sh](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/scripts/smoke-test.sh) | 一键跑 14 个直接执行入口 |
| 实测档案 | [docs/test-runs.md](test-runs.md) | 历次跑批结果 + 破坏性变更记录 |
| changelog | [CHANGELOG.md](../CHANGELOG.md) | 版本号 + 重要变更 |
| 长远 spec | [docs/superpowers/specs/2026-05-30-tutorial-positioning-design.md](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/superpowers/specs/2026-05-30-tutorial-positioning-design.md) | 项目定位与 Phase 1+2+3 总规划 |
| awesome-list PR | [docs/superpowers/awesome-list-pr-templates.md](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/docs/superpowers/awesome-list-pr-templates.md) | PR A/B 模板 + fallback list |

## 历次年度回顾

- 2026-12（待）

## 历次承诺更新

| 日期 | 承诺 | 备注 |
|---|---|---|
| 2026-05-30 | 90 天兼容 + 24 小时 banner | 初版承诺 |
