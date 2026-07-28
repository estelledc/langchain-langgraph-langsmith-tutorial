# Agent Engineering Lab 工作约定

- 中文环境，结论先行；先解释任务合同，再谈框架 API。
- `src/agent_lab/domain/` 不得依赖 LangChain、LangGraph、LangSmith 或模型提供商。
- 默认路径必须离线、无 API Key 可运行。在线模型、LangSmith、Deep Agents 和持久化后端都是可选扩展。
- 工具输入、输出、错误、权限、副作用和幂等语义必须显式建模。
- 不可信内容只能进入 `Evidence`，不得直接升级为事实或长期记忆。
- 评测结果只允许 `PASS / FAIL / UNKNOWN / ERROR`；基础设施错误不得折算成质量分。
- 修改行为时同步补充确定性测试或回归样例。先跑最小测试，交付前跑 `uv run agent-lab verify`。
- 不提交凭证、内部地址、本机绝对路径、真实用户数据或未经筛选的生产 Trace。
- V1 已冻结在 `v1-legacy`；只修安全或迁移入口，不在 `legacy/v1/` 继续扩课。
