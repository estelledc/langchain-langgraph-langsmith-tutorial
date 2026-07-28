## What changed

说明用户可见行为和涉及的合同/runtime/dataset/lab。

## Why

根因或已观察问题是什么？为什么更小方案不够？

## Evidence

- [ ] `uv sync --frozen`
- [ ] `uv run agent-lab verify`
- [ ] `bundle exec jekyll build`
- [ ] `uv run python scripts/check_site.py --built _site`
- [ ] `bundle exec htmlproofer _site --disable-external --no-enforce-https --swap-urls '^/langchain-langgraph-langsmith-tutorial:'`

列出新增/变化的 case_id，以及修复前后的结果。不要只写“测试通过”。

## Behavior delta

| Dataset / case | Before | After | Evidence |
|---|---|---|---|
| | | | |

## Risk and rollback

权限、数据、成本、延迟、兼容性和回滚路径。

## UNKNOWN

明确写出未执行的 live provider、online eval、Agent Server 或 production verification。
