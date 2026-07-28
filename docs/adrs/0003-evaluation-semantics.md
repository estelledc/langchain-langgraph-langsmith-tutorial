---
layout: default
title: ADR 0003 · Evaluation semantics
---

# ADR 0003：评测语义

状态：Accepted

## 决定

单个 grader 只返回：

- `PASS`：证据足以确认满足合同。
- `FAIL`：证据足以确认违反合同。
- `UNKNOWN`：输入或证据不足，暂时不能判断。
- `ERROR`：grader 或基础设施自身失败。

只有 `PASS/FAIL` 可以带 `score`。`UNKNOWN/ERROR` 不参加平均分，并单独统计 unknown rate 与 evaluator error rate。

## 优先顺序

```text
schema / policy / budget
→ tool step
→ trajectory
→ attribution
→ semantic judge
→ human review
```

确定性事实不得交给 LLM Judge，包括禁止工具、参数 schema、调用次数、状态转换、引用存在性、审批和预算。

## 数据集边界

- capability：目标能力。
- regression：已知缺陷。
- adversarial：注入、越权、恶意工具输出和预算攻击。
- contracts：工具与状态合同。

数据集、grader 和 prompt 独立版本化。一次模型运行不等于稳定通过；在线候选需支持多 trial 报告。
