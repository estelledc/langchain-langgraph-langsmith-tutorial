# Changelog

本仓库重要变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

版本号语义：

- **MAJOR**：教程结构 / 学习路径破坏性变更（fork 者需要重新走一遍）
- **MINOR**：加新 tutorial / 新 docs 章节，向下兼容
- **PATCH**：修 bug / 改 prompt 措辞 / 加报错条目

---

## [Unreleased]

### Changed

- 重定位为 Agent Engineering Lab，课程从框架 API 路径改为 15 个核心实验与 10 个前沿实验。
- V1 冻结到 `v1-legacy`，旧教程保存在 `legacy/v1/`。
- 安装合同迁移到 `pyproject.toml` + `uv.lock`，默认离线且无需 API Key。
- 新增 framework-neutral `RunRequest / Evidence / Citation / RunResult` 合同。
- 新增 Workflow、LangChain `create_agent` 和 LangGraph runtime 边界。
- 删除模型可控 `eval()` 路径，使用受限 AST 计算器。
- 模拟搜索改为明确的版本化 fixture search。
- 新增 capability、regression、adversarial、tool-contract 数据集与四态 grader。
- 新增严格 fast suite、verification passport、课程生成器和统一 CI/Pages 门禁。

---

## [1.1.0] — 2026-05-30

围绕"长期可信度 + 被收录"的对外升级。

### Added
- **README 全面重写** —— 顶部加差异化定位段（三个不可替代卖点）、作者介绍、60 秒 Quick Start 可跑代码块；不再用旧的"30 秒上手"开场
- `docs/screenshots/00-roadmap.png` —— 4 周 16 篇路线图 PNG hero 图（`docs/screenshots/_roadmap.mmd` 是 mermaid 源文件，可用 `npx @mermaid-js/mermaid-cli` 重渲染）
- `tutorial/README.md` —— 学习路线表加每周总时长，前置概念依赖表加每篇预估时长（4 周 ~17 小时）
- `docs/maintenance.md` —— 长期维护节奏与 SOP（周/月/季/年/触发式）
- `scripts/smoke-test.sh` —— 一键跑批 14 个 final/.py，支持 `--quick`，结果自动 append 到 `docs/test-runs.md`
- `docs/superpowers/specs/2026-05-30-tutorial-positioning-design.md` —— 长远定位 spec（Phase 1+2+3 总规划）
- `docs/superpowers/awesome-list-pr-templates.md` —— 中文 awesome list PR 模板 + 已提交 PR 清单

### Changed
- 与官方教程对比表保留，但定位段移到 README 顶部之前
- 「30 秒上手」改名「完整环境（4 周教程）」，仅作 Quick Start 失败后的兜底入口

### Submitted (外部 PR)
- [`AiHubCN/Awesome-Chinese-LLM#102`](https://github.com/AiHubCN/Awesome-Chinese-LLM/pull/102) — open / mergeable
- [`WangRongsheng/awesome-LLM-resources#127`](https://github.com/WangRongsheng/awesome-LLM-resources/pull/127) — open / mergeable

### Maintenance commitment
- 至少 90 天内 final/.py 全跑通；撞破坏性变更 24 小时内 README 加 banner
- 月级跑 smoke test，季度做完整体检，年度做结构性回顾（详见 [docs/maintenance.md](docs/maintenance.md)）

---

## [1.0.x post-release] — 2026-05-29 ~ 05-30

### Added (P0-P1 系统化补建)
- `CLAUDE.md` + `.cursorrules` —— Claude Code / Cursor 自动加载教学约束（7 条心法转 AI 默认行为）
- `_scratch/journal/example-week1-01-hello-llm.md` —— 真实卡点日志样例
- `.github/ISSUE_TEMPLATE/` 三类（bug / 卡点反馈 / enhancement） + `pull_request_template.md`
- `CONTRIBUTING.md` —— 三种贡献方式 + 文件层约束
- `docs/screenshots/` —— Pages 渲染 4 张截图嵌 README
- `tutorial/README.md` —— mermaid 学习路径图 + 前置概念依赖表
- `LICENSE` —— 加 Attribution 段 + 学术引用 BibTeX

### Fixed
- README 30 秒上手段：clone URL 改成实际 repo 名（之前指向不存在的 `langchain-tutorial-zero.git`）

---

## [1.0.0] — 2026-05-29

第一个完整可发布版本。

### Added
- **教程主体**：4 周 14 篇 tutorial（week-1-langchain × 5 / week-2-tools-and-agent × 1 / week-3-langgraph × 4 / week-4-langsmith-and-project × 4）
- **参考代码**：14 个独立可执行 final/.py，全部经作者实测（langchain 1.3.2）
- **元教程**：`HOW_TO_LEARN_WITH_AI.md` 七条 prompt 心法
- **环境**：`SETUP.md` 含 DashScope + LangSmith API 申请、依赖安装、烟雾测试、排错表
- **共享 boilerplate**：`final/_common.py` 抽 `make_llm()` 工厂函数
- **速查 docs/**：
  - `concepts.md` —— 18 个零基础卡点术语词典
  - `prompts-cheatsheet.md` —— 7 场景 21 个 prompt 模板 + 反模式表
  - `debug-recipes.md` —— 6 类报错 16 个常见问题速查 + 万能诊断 prompt
  - `challenges.md` —— capstone 后 7 个真实小项目挑战
  - `test-runs.md` —— 14 个 final 实测档案（含 langchain 1.x 兼容性修复 6 处）
- **GitHub Pages**：`_config.yml` + `assets/css/style.scss` + `.github/workflows/pages.yml` 一键部署 jekyll-theme-cayman + relative-links 插件

### Fixed (langchain 1.x 兼容性，详见 docs/test-runs.md)
- `langchain_core.pydantic_v1` 移除 → 改用 `from pydantic import BaseModel, Field`
- `langchain.agents.AgentExecutor` 拆到 `langchain_classic.agents`
- `langsmith.evaluation.LangChainStringEvaluator` 移除 → 改用自定义 LLM-as-Judge
- `create_dataset(data_type="kv")` 不再支持字符串
- `input()` 在批跑触发 EOFError → 加 try/except
- 缺 `grandalf` 依赖（StateGraph ASCII 可视化）

---

## 维护节奏

- 每周扫一次 LangChain release notes，重大变更触发新 PATCH（更新 docs/test-runs.md + final/）
- 每月看一次 issue 列表，整理"高频卡点"开 MINOR 版本补 tutorial / docs
- 每半年回头跑一次全量 final smoke test（pin 版本不变，但跑通验证还在）

## 历史前期 commit（pre-1.0.0）

参见 git log，重点：

- `8cb037a` 实测报告 + tutorial/SETUP/README 同步实测发现
- `2958561` langchain 1.x 兼容性修复
- `0305339` 写 week-1/01_hello_llm.md（gold standard 标杆 tutorial）
