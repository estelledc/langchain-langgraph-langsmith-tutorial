---
layout: default
title: Verification
description: 测试、Eval、部署与真实外部结果的证据分层。
---

# Verification

## 证据不是同一种东西

| 层级 | 能证明 | 不能证明 |
|---|---|---|
| Fixture | 离线输入固定、回归可重复 | 当前互联网事实 |
| Unit / contract test | 确定性代码合同 | 模型概率性质量 |
| Offline eval | 指定 dataset、runtime、grader 下的行为 | 真实流量分布 |
| Live smoke | 某次外部请求成功 | 稳定性、用户价值、生产状态 |
| Deployment receipt | 某 commit 已部署 | 业务目标已达成 |
| Production observation | 真实流量中的行为 | 因果收益，除非有对照设计 |

## 本地门禁

```bash
uv sync --frozen
uv run agent-lab verify
bundle exec jekyll build
uv run python scripts/check_site.py --built _site
```

fast suite 由 9 个 case × 2 个 runtime 组成，要求：

- pass rate = 1.0。
- unknown rate = 0.0。
- evaluator error rate = 0.0。

这里的 unknown rate 指 grader 无法判断，不是 Agent 正确返回 `RunStatus.UNKNOWN`。无证据拒答本身可以是一个通过的 capability case。

## 四态语义

- `PASS`：有足够证据确认满足合同，score 必须存在。
- `FAIL`：有足够证据确认违反合同，score 必须存在。
- `UNKNOWN`：证据不足，不带 score。
- `ERROR`：grader 或基础设施失败，不带 score，必须保留 error。

任何 `UNKNOWN/ERROR → 0.5` 的转换都会制造假质量信号，因此被 schema 拒绝。

## Verification Passport

```bash
uv run agent-lab passport --suite fast --output verification-passport.json
```

护照包含：

- source commit 与 worktree dirty 状态。
- `uv.lock` 哈希和关键包版本。
- dataset 文件哈希与 suite 版本。
- pass、unknown、evaluator error rate。
- 未运行的 live provider 与 online eval。

护照是运行产物，由 CI 上传；它不与源码 commit 自引用地混在一起。

## 生产回流

真实失败 Trace 只有经过筛选、脱敏、最小复现和 owner review 后，才能进入 `datasets/production/`。Fixture、合成样例和模型实验不得改名冒充生产证据。
