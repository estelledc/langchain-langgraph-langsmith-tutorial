# Contributing

最有价值的贡献不是再加一个框架示例，而是把真实失败变成可复现合同。

## 适合提交的改动

- 新的 capability case：证明目标能力还缺什么。
- 新的 regression case：固定已经观察和修复的失败。
- 工具合同修复：input/output、错误、权限、副作用或幂等语义。
- Grader 修复：减少假通过、假失败、UNKNOWN 或 evaluator ERROR。
- 实验改进：让学习者能独立预测、打破、Trace 和解释行为。
- 兼容性修复：包含 fresh install 和受影响 runtime 的验证证据。

## 不直接接受

- 只增加 provider 示例，没有独立任务和验收。
- 把 fixture、单次模型试验或 CI 构建写成生产成功。
- 用 LLM Judge 检查 schema、权限、预算、工具次数或引用存在性。
- 将真实 API Key、未脱敏 Trace、个人数据或内部地址提交到仓库。
- 让核心领域模型依赖 MCP、Deep Agents 或具体 provider。

## 本地流程

```bash
uv sync --frozen
uv run agent-lab verify
bundle exec jekyll build
uv run python scripts/check_site.py --built _site
```

改课程元数据后运行：

```bash
uv run python scripts/generate_curriculum.py
uv run python scripts/check_curriculum.py
```

## PR 必须回答

1. 哪个可观察行为发生变化？
2. 根因在哪一层：合同、工具、上下文、状态、runtime、provider 还是 evaluator？
3. 哪个 test 或 dataset case 在修复前会失败？
4. 哪些结果仍然是 `UNKNOWN`？
5. 成本、延迟、权限或维护复杂度是否增加？

提交前不要宽泛暂存。只把本 PR 的文件加入 commit。
