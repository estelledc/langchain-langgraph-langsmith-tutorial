# Contributing — 怎么给本仓库贡献

> 你卡的地方就是本仓库下次该补的地方。Issue / PR 都欢迎，下面说清"该往哪贡献"。

---

## 三种贡献方式（按工作量从小到大）

### 1. 提 Issue（5 分钟）

最有价值的 issue 类型：

- **[卡点反馈](.github/ISSUE_TEMPLATE/learning_block.md)**：你走某篇 tutorial 卡住超 30 分钟（不是代码报错，是讲解不清）。模板会问你"你觉得怎么改"——这是仓库改进最重要的输入
- **[报错](.github/ISSUE_TEMPLATE/bug.md)**：跑 `final/` 或 `_scratch/` 撞到 [docs/debug-recipes.md](docs/debug-recipes.md) 没列的报错
- **[改进建议](.github/ISSUE_TEMPLATE/enhancement.md)**：教程结构 / 工具链 / 文档建议

### 2. 提 PR 加内容（30 分钟 - 2 小时）

最受欢迎的 PR 类型：

- **加新报错到 [docs/debug-recipes.md](docs/debug-recipes.md)**：你撞到 + 修好的报错——大概率别人也会撞
- **加新 prompt 到 [docs/prompts-cheatsheet.md](docs/prompts-cheatsheet.md)**：你沉淀的私房 prompt
- **加新概念到 [docs/concepts.md](docs/concepts.md)**：你被某个术语卡过，写了个好类比
- **完成 [docs/challenges.md](docs/challenges.md) 后写卡点日志**：PR 你的 `_scratch/journal/challenge-N-<日期>.md` 让别人参考

### 3. 提 PR 改 tutorial 主体（半天 +）

- **拆某篇 tutorial 的任务**：发现某篇任务卡密度太高 / 太低，提议重新切分
- **加新一篇 tutorial**：覆盖现有 14 篇没讲到的话题（structured output / streaming 进阶 / async 等）。**先开 issue 讨论**再写

---

## PR 流程

1. **fork 本仓库**到你的 GitHub
2. **建分支**：`git checkout -b add-debug-recipe-pydantic-error`
3. **改文件**：注意约束（见下方）
4. **本地校验**：
   ```bash
   # markdown 链接校验
   python3 -c "
   import re, pathlib
   broken = []
   for p in pathlib.Path('.').rglob('*.md'):
       if '_scratch' in str(p) or '.venv' in str(p): continue
       text = p.read_text()
       for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', text):
           link = m.group(2).split('#')[0]
           if link.startswith(('http', '#', 'mailto')) or not link: continue
           target = (p.parent / link).resolve()
           if not target.exists():
               broken.append(f'{p}: {link}')
   print('\n'.join(broken) if broken else 'all links ok')
   "
   ```
5. **提 PR**，按 [pull_request_template.md](.github/pull_request_template.md) 填

---

## 必须遵守的约束

### 内容层面

- ✅ **零基础视角**：术语首次出现配日常类比；不假设读者懂 LLM/Embedding/Agent 等
- ✅ **结论先行**：每段第一句给结论
- ✅ **列表 > 段落**：能列点不写段
- ❌ **不用 emoji**：保持仓库整体风格
- ❌ **不用装饰边框**：不要 `═══` 之类的 ASCII art

### 文件层面

- ✅ 改 `tutorial/*.md` / `docs/*.md` —— 教学主体
- ✅ 加 `_scratch/journal/example-*.md` —— 真实日志样例（白名单已加）
- ❌ **不改 `final/*.py`**——那是参考答案。除非是 langchain 升级兼容性修复，且必须更新 [docs/test-runs.md](docs/test-runs.md)
- ❌ **不引入新依赖**——除非有强理由，且要更新 `requirements.txt` + 跑过所有 final
- ❌ **不提交 `.env` / 任何 API key**——pre-commit hook 会拦但别测试它

### 代码层面（如果改 final）

- 必须真实跑过你改的 .py，把输出贴在 PR 描述里
- 必须更新 [docs/test-runs.md](docs/test-runs.md) 对应行
- 跨文件改动（影响多个 final）请先开 issue 讨论

---

## 怎么写 issue 才有用

❌ 不好的 issue：

> "tutorial 看不懂"

✅ 好的 issue：

> "tutorial/week-3-langgraph/02_conditional_edges.md 任务 3 我卡了 40 分钟。
> 卡点是 add_conditional_edges 的第三个参数（mapping dict），任务卡只说'跟条件函数返回值对应'，
> 但没解释为什么要分两层——条件函数返回 'use_tool'，mapping 把它翻成 'tool_node'。
> 我觉得加一句'你可以理解成路由表：条件函数返回'路由 key'，mapping 决定 key 跳哪个节点'会更清楚。
> 我的背景：6 个月 Python，第一次写 LangGraph"

---

## 联系方式

- 紧急问题：开 issue 加 `urgent` label
- 一般讨论：[Discussions](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/discussions) （如开启）
- 邮件：见 commit author email

感谢你的贡献——零基础视角对本仓库比工程师视角更值钱。
