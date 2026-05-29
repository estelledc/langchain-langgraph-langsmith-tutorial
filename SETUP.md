# SETUP — 环境搭建

## 0. 你需要的

- Python 3.10+ （`python --version` 检查）
- 一个 [DashScope](https://dashscope.aliyuncs.com) 账号（阿里云通义千问，国内手机号注册即可，新号送免费额度）
- 一个 [LangSmith](https://smith.langchain.com) 账号（免费 5K runs/月，看 Trace 用）

## 1. 申请 API Key（10 分钟）

### DashScope（必填）
1. 登录 [dashscope.aliyuncs.com](https://dashscope.aliyuncs.com)
2. 控制台 → API-KEY 管理 → 创建新 Key
3. 复制 Key（形如 `sk-xxxxxxxx`），放到下面 .env 里的 `DASHSCOPE_API_KEY`

### LangSmith（必填）
1. 登录 [smith.langchain.com](https://smith.langchain.com)
2. 右上头像 → Settings → API Keys → Create API Key
3. 复制（形如 `lsv2_xx...`），放到 .env 里的 `LANGCHAIN_API_KEY`

## 2. 配 .env

```bash
cp .env.example .env
# 用任何编辑器打开 .env，把两个 _here 占位符替换成真实 Key
```

最少要填的：
- `DASHSCOPE_API_KEY`
- `LANGCHAIN_API_KEY`

`LANGCHAIN_PROJECT=study` 是 LangSmith 项目名，所有 Trace 会聚在 `study` 项目里，可以保留默认。

## 3. 装依赖

```bash
# 推荐用虚拟环境（避免和系统 Python 包冲突）
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## 4. 第一次烟雾测试

```bash
python final/01_langchain/01_hello_llm.py
```

期望看到：
- 三段输出（invoke / stream / batch）
- 末尾 `✅ 运行完毕！请前往 ...`

如果看到 LLM 真的回答了「什么是 LangChain？」，那环境就 OK 了。

## 5. 看一眼 Trace（验证 LangSmith 通了）

打开 [smith.langchain.com](https://smith.langchain.com)，左侧 Projects → `study`，应该看到刚才那次跑的 Trace 记录（含每次 LLM 调用、token 用量、耗时）。

## 排错

| 报错 | 多半原因 | 处理 |
|------|----------|------|
| `KeyError: 'DASHSCOPE_BASE_URL'` | .env 没加载 / 没填完 | 检查 .env 是不是放在仓库根目录，是不是写完整 |
| `401 Unauthorized` | DashScope key 错 | 重新去 DashScope 控制台拷一遍 |
| `403 ... text-embedding-v3` | DashScope key 没开通 embedding 模型 | 控制台 → 模型广场 → 开通"通用文本向量"。仅 `final/01_langchain/05_rag_basic.py` 这篇用得到，其他文件不影响 |
| `ModuleNotFoundError: langchain_xxx` | 依赖没装全 | `pip install -r requirements.txt --upgrade` |
| `ImportError: cannot import name 'X' from 'langchain.Y'` | langchain 版本破坏性变更（如 1.x 把 0.x 的 agents / pydantic_v1 移走了） | 看 [docs/test-runs.md](docs/test-runs.md) 第一节"需要修的代码"——已记录已知的 6 处迁移；新版变更跟着 [LangChain 升级指南](https://python.langchain.com/docs/versions/v1_0/) 改 import 路径 |
| `Connection error` | 网络问题 / 代理 | .env 取消注释 `HTTPS_PROXY` 或 ping `dashscope.aliyuncs.com` |
| `SSLCertVerificationError` | 公司 MDM 网络拦了 TLS | 普通家用网络不会撞；公司网络下需要 export 公司 CA bundle 到 `SSL_CERT_FILE` 和 `REQUESTS_CA_BUNDLE` |
| LangSmith 没看到 Trace | `LANGCHAIN_TRACING_V2` 没设 / API key 错 | 检查 .env 里这两项 |

实在搞不定？发 Issue 时贴：你的 OS、Python 版本、完整报错（删掉 key）。

> **想验证仓库代码是否真能跑？** 看 [docs/test-runs.md](docs/test-runs.md)——作者实测过所有 14 个独立可执行脚本的真实输出和耗时。
