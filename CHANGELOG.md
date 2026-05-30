# Changelog

本仓库重要变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。

版本号语义：

- **MAJOR**：教程结构 / 学习路径破坏性变更（fork 者需要重新走一遍）
- **MINOR**：加新 tutorial / 新 docs 章节，向下兼容
- **PATCH**：修 bug / 改 prompt 措辞 / 加报错条目

---

## [Unreleased]

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
