# Trusted Research

V2 的首个纵向切片。入口：

```bash
uv run agent-lab run --goal "LangGraph 的 checkpointer 和 Store 有什么区别？"
uv run agent-lab eval --suite fast
```

它支持 fixture search、typed Evidence、引用校验、无证据拒答、注入隔离、预算和结构化 Trace。默认实现不联网、不调用模型。
