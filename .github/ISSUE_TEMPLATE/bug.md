---
name: Bug / Regression
about: 报告可复现的合同、runtime、grader、课程或部署问题
title: "[Bug] "
labels: bug
assignees: ''
---

## 失败位于哪一层

- [ ] install / lock
- [ ] domain contract
- [ ] tool / retrieval / memory
- [ ] Workflow / LangChain / LangGraph / Deep Agents
- [ ] dataset / grader / gate
- [ ] curriculum / site
- [ ] deployment

## 最小复现

```bash
# 不含凭证和本机私密路径
```

涉及 eval 时请填写 suite、case_id、runtime、trial 和 dataset version。

## 预期行为

说明你依据的合同、测试或文档。不要只写“应该成功”。

## 实际行为

贴最小错误片段、`RunStatus`、`termination_reason` 或 grader 四态结果。删除 token、真实用户数据和内部地址。

## 环境

- source commit：
- Python：
- `uv.lock` 是否未修改：
- OS：
- 是否需要外部 provider：

## 回归资产建议

这个问题应进入 unit test、contract test、capability case、regression case 还是 adversarial case？
