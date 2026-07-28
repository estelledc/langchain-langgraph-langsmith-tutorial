#!/usr/bin/env python3
"""Generate learner-facing lab pages from curriculum/labs.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

STATUS_NOTES = {
    "executable": "离线实现与自动化验收均已接入。",
    "implemented": "实现已存在；涉及外部模型的行为仍需单独 live 验证。",
    "configured": "部署或集成合同已配置；外部结果以实际 receipt 为准。",
    "scaffold": "当前只有实验合同，不代表能力已实现或验证。",
}


def load_curriculum(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "curriculum/labs.yaml").read_text(encoding="utf-8"))


def render_index(curriculum: dict[str, Any]) -> str:
    labs = sorted(curriculum["labs"], key=lambda item: item["order"])
    lines = [
        "# Agent Engineering Labs",
        "",
        "课程不是按框架 API 排序，而是按任务合同、工具、上下文、状态、证据和评测能力推进。状态字段是证据边界，不是进度装饰。",
        "",
    ]
    for track_id in ("core", "frontier"):
        track = curriculum["tracks"][track_id]
        selected = [item for item in labs if item["track"] == track_id]
        lines.extend(
            [
                f"## {track['title']}",
                "",
                track["description"],
                "",
                "| 实验 | 状态 | 产物 | 核心问题 |",
                "|---|---|---|---|",
            ]
        )
        for lab in selected:
            lines.append(
                f"| [{lab['id']} · {lab['title']}]({lab['id']}/) | "
                f"`{lab['status']}` | {lab['artifact']} | {lab['question']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 状态语义",
            "",
            "- `executable`：离线实现与自动化验收均已接入。",
            "- `implemented`：代码已存在，但 live provider 或外部系统行为未在默认门禁中验证。",
            "- `configured`：集成与部署配置已存在，不能替代真实部署 receipt。",
            "- `scaffold`：只有实验合同，不能描述为已完成能力。",
            "",
            "V1 教程保存在 [`v1-legacy`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/tree/v1-legacy)。",
            "",
        ]
    )
    return "\n".join(lines)


def render_lab(lab: dict[str, Any]) -> str:
    prerequisite_text = ", ".join(f"[{item}](../{item}/)" for item in lab["prerequisites"]) or "无"
    reference_lines = [
        f"- [`{path}`](https://github.com/estelledc/langchain-langgraph-langsmith-tutorial/blob/master/{path})"
        for path in lab["references"]
    ]
    lines = [
        f"# {lab['id']} · {lab['title']}",
        "",
        f"> 状态：`{lab['status']}`。{STATUS_NOTES[lab['status']]}",
        "",
        lab["summary"],
        "",
        f"- 前置：{prerequisite_text}",
        f"- 产物：`{lab['artifact']}`",
        f"- 能力：{', '.join(f'`{item}`' for item in lab['capabilities'])}",
        "",
        "## 1. Frame",
        "",
        f"先回答：**{lab['question']}**",
        "",
        "不要先选框架。先写清任务输入、成功条件、风险和不需要 Agent 的最小基线。",
        "",
        "## 2. Predict",
        "",
        "在运行前预测可观察轨迹：",
        "",
        "```text",
        " → ".join(lab["expected_trace"]),
        "```",
        "",
        "## 3. Build",
        "",
        f"完成或审查 `{lab['artifact']}`。实现入口：",
        "",
        *reference_lines,
        "",
        "## 4. Break",
        "",
        "主动制造这些失败，不要只跑 happy path：",
        "",
        *(f"- `{item}`" for item in lab["failure_cases"]),
        "",
        "## 5. Trace",
        "",
        "只记录节点、工具、状态、证据和终止原因。不要记录或要求模型暴露隐藏推理。",
        "",
        "## 6. Evaluate",
        "",
        "验收合同：",
        "",
        *(f"- `{item}`" for item in lab["acceptance"]),
        "",
        "关联 suite：" + ", ".join(f"`{item}`" for item in lab["eval_suites"]),
        "",
        "## 7. Reflect",
        "",
        "解释额外复杂度解决了哪个已观察问题，以及它新增了什么维护和失败成本。",
        "",
        "## 8. Promote",
        "",
        "把失败提升为 dataset case、确定性 policy、测试或版本化 learning。没有新证据时，不提升状态。",
        "",
        "[返回实验目录](../)",
        "",
    ]
    return "\n".join(lines)


def expected_outputs(root: Path) -> dict[Path, str]:
    curriculum = load_curriculum(root)
    outputs = {root / "labs/README.md": render_index(curriculum)}
    for lab in curriculum["labs"]:
        outputs[root / "labs" / lab["id"] / "README.md"] = render_lab(lab)
    return outputs


def generate(root: Path) -> None:
    for path, content in expected_outputs(root).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    generate(args.root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
