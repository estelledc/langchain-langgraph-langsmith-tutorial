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

离线发布门禁由三类 suite 组成：

| suite | 数据合同 | 当前规模 | 证明范围 |
|---|---|---:|---|
| `fast` | `RunRequest → RunResult` | 9 case × 2 runtime = 18 case-run | Workflow/LangGraph 行为与证据合同 |
| `security` | 结构化 tool input | 4 case | 安全计算器拒绝注入、属性访问和资源超限 |
| `contracts` | 版本化 ToolSpec | 2 case | 实现声明与 capability、副作用、错误集合一致 |

三者都要求：

- pass rate = 1.0。
- unknown rate = 0.0。
- evaluator error rate = 0.0。
- runtime error rate = 0.0。

这里的 unknown rate 指 grader 无法判断，不是 Agent 正确返回 `RunStatus.UNKNOWN`。无证据拒答本身可以是一个通过的 capability case。

## 四态语义

- `PASS`：有足够证据确认满足合同，score 必须存在。
- `FAIL`：有足够证据确认违反合同，score 必须存在。
- `UNKNOWN`：证据不足，不带 score。
- `ERROR`：grader 或基础设施失败，不带 score，必须保留 error。

任何 `UNKNOWN/ERROR → 0.5` 的转换都会制造假质量信号，因此被 schema 拒绝。

## Verification Passport

Passport 不能由调用者手填 `test_status=PASS`。推荐一次完成构建、完整门禁和护照：

```bash
uv build --clear
uv run agent-lab verify \
  --passport-output verification-passport.json \
  --artifact-dir dist
```

`uv run agent-lab passport` 是同一完整门禁的便利入口，不是跳过测试的报告生成器。

护照包含：

- source commit 与 worktree dirty 状态。
- `uv.lock` 哈希和关键包版本。
- dataset、suite 配置和 wheel/sdist 文件哈希。
- Ruff、mypy、pytest、课程、站点和三类 suite 的派生 gate 状态。
- 每套 suite 的 pass、unknown、evaluator error、runtime error rate。
- 未运行的 live provider 与 online eval。

护照是运行产物，由 CI 上传；它不与源码 commit 自引用地混在一起。

CI 还会在 Python 3.11、3.12、3.13 运行离线门禁，并把 wheel 安装到干净环境，从源码树外执行 `agent-lab run/eval`、`xcodefix task` 和全部 optional extras 的 import smoke。这个检查证明发行物可安装，不证明外部 provider 或 Xcode live run 可用。

## XcodeFixBench 证据边界

`keyboard-layout-001` 当前是 `synthetic-seeded` 开发任务。Gold Patch、Negative Patch、审批绑定、XCTest 和 Simulator runtime oracle 已形成完整本机证据链，但任务尚未达到 RFC 要求的 20 次稳定性晋级门禁，也没有真机第三方输入法或 held-out Agent 排行榜。因此它属于 executable dev slice，不属于正式 benchmark 成绩。

## 生产回流

真实失败 Trace 只有经过筛选、脱敏、最小复现和 owner review 后，才能进入 `datasets/production/`。Fixture、合成样例和模型实验不得改名冒充生产证据。
