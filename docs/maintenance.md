---
layout: default
title: Maintenance
description: 依赖、数据集、课程、外部验证与发布的维护合同。
---

# Maintenance

## 每个 PR

- `uv sync --frozen`。
- `uv run agent-lab verify`。
- Jekyll build + rendered site + HTML-Proofer。
- 行为变化必须对应 test 或 dataset case。

## 依赖更新

只在独立分支更新 lock。先验证最低/最高支持 Python，再运行离线门禁。Provider、LangSmith 或 Deep Agents 的外部行为必须单独记录 live receipt，不能由 import smoke 替代。

## 数据集

- capability 描述目标，不混入已知 bug。
- regression 只接受可复现失败。
- adversarial 记录攻击面，不承诺覆盖全部攻击。
- production case 必须筛选、脱敏和 owner review。

## 发布

PR build 成功后才合并 master。只有 master 的 Pages deployment 成功并核验公开 URL，才能写“站点已部署”。Agent Server、online eval 和真实 provider 仍保持独立状态。
