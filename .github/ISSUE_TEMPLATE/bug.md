---
name: 报错 / Bug
about: 跑 final/ 或 _scratch/ 时遇到代码报错，且 docs/debug-recipes.md 里没列
title: "[Bug] "
labels: bug
assignees: ''
---

## 报错来自哪个文件

- 文件：`final/01_langchain/0X_xxx.py` 或 `_scratch/my_xxx.py`
- 当时跑的命令：`python final/...` / `python -c "..."`

## 我以为会发生什么

（1-2 句话讲你的预期。这一步比 traceback 更重要——暴露你的心智模型在哪卡住）

## 实际报错

```
[贴报错最后 5-10 行 traceback，删掉 path 里你不想公开的部分]
```

## 已经试过什么

- [ ] 检查了 [docs/debug-recipes.md](../../docs/debug-recipes.md) 没有匹配条目
- [ ] 跑了 `pip show langchain` 看版本（贴版本号在下方）
- [ ] 用了 [万能诊断 prompt](../../docs/debug-recipes.md#万能诊断-prompt) 问 AI 但没解决

## 环境

- Python：`python --version` 输出
- LangChain：`pip show langchain` 的 Version 行
- OS：Mac / Win / Linux
- 网络：公网 / 公司 / VPN
