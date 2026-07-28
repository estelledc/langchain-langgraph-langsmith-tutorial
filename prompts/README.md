# Prompts

模型运行时使用的 Prompt 必须版本化并随包发布。当前源真相在：

- `src/agent_lab/data/prompts/trusted_research_system_v1.txt`

修改 Prompt 时必须新建版本、运行相同 dataset，并报告 capability、regression、成本和 UNKNOWN 的变化；不要直接覆盖既有版本后沿用旧名字。
