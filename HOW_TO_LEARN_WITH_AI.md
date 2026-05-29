# HOW TO LEARN WITH AI — 怎么用 Claude Code / Cursor 学陌生代码

> 5 分钟读完。读完之前别开始走 tutorial。

## 这套教程的底层假设

零基础选手看陌生代码，会同时卡两件事：

1. **「这一行做什么」**——文档能查，慢但能解
2. **「为什么要这样写」**——文档不写，只有跟资深开发者对话才能解

AI 工具（Claude Code / Cursor）的本质优势是「陪问陪改」——但你不知道问什么。

所以这套 tutorial 给你预制问题清单：你只要按任务卡复制粘贴 prompt 给 AI，就能把代码问明白。

## 三个角色，分清

| 角色 | 是谁 | 干嘛的 |
|------|------|--------|
| 学习者 | 你 | 复制 prompt 给 AI、写代码、自检、记卡点日志 |
| AI 工具 | Claude Code / Cursor | 陪问、解释、纠错——**不许直接给完整代码** |
| `final/` | 参考答案 | 任务卡指明何时偷看；偷看是用来对比，不是用来抄 |

## 七条 prompt 心法

### 1. 不问"X 是什么"，问"X 跟我已知的 Y 有什么共同点"

❌ 不好：「ChatPromptTemplate 是什么？」
✅ 好：「ChatPromptTemplate 跟我熟悉的 Python f-string / 邮件模板有什么共同点和不同点？」

为什么：AI 用类比解释，你才能"挂"在已知概念上记住。

### 2. 不让 AI 直接给代码，先让它列大纲

❌ 不好：「帮我写一个 demo_invoke 函数」
✅ 好：「帮我列大纲：demo_invoke 函数大概有几步？每步干嘛？等我说 OK 再给代码」

为什么：你先看清结构，再看实现，能区分「设计问题」和「写法问题」。

### 3. 报错先描述"我以为会发生什么"，再贴报错

❌ 不好：「我跑代码报错了，[贴 traceback]」
✅ 好：「我跑 _scratch/my_01_hello.py，我以为会输出 `1+1=2` 之类，结果报错：[贴 traceback]。请别直接修——给我 3 个候选原因让我猜哪个最可能」

为什么：你的"以为"比报错更值钱——它暴露你的心智模型在哪卡住。

### 4. 卡 5 分钟以上必须开新 prompt

旧对话堆太多上下文，AI 会变笨；你也容易跟着 AI 跑偏。

发现卡住超 5 分钟 → 关掉当前对话 → 重写一句精简的 prompt → 从头开始。

### 5. 让 AI 用日常类比，不堆术语

每个 prompt 后面加一句：

```
请用日常类比解释。不要堆术语。回答 200 字内。
```

技术解释从日常类比开始——从微信群、邮件模板、流水线、菜谱这种零基础也懂的概念出发，再升级到术语。

### 6. 让 AI 单次只引一个问题

零基础选手最容易被 AI"一口气问 5 个问题"打懵。

每个 prompt 加：

```
每次只问我一个问题，等我回答你再继续。
```

### 7. 自己写完再让 AI 对比 final

✅ 好：「我自己写了 _scratch/my_01_hello.py（[贴代码]），请对比 final/01_langchain/01_hello_llm.py：哪里是『风格差异』，哪里是『真错』？真错的地方告诉我『为什么这样写更好』，**不要直接给修改后代码**——让我自己改」

为什么：自己改一次，记住率比抄代码高 5-10 倍。

## 怎么用 Claude Code

```bash
# 1. 安装（如果还没）
brew install anthropic/claude/claude        # macOS
# 或下载 desktop app from claude.ai/download

# 2. cd 到仓库根目录
cd langchain-tutorial-zero

# 3. 启动
claude

# 4. 把任务卡里的 prompt 复制粘贴进对话框
```

Claude Code 默认能读你仓库里所有文件，所以 prompt 里写「请看 final/01_langchain/01_hello_llm.py 第 14-25 行」是直接生效的。

## 怎么用 Cursor

```
1. 下载安装：cursor.com
2. File → Open Folder → 选 langchain-tutorial-zero
3. 关键快捷键：
   - Cmd+L (Mac) / Ctrl+L (Win)：打开右侧 chat 对话框（"陪问"模式）
   - Cmd+K：在编辑器里弹小输入框（"行内修改"模式）

主要用 Cmd+L——就是把任务卡的 prompt 粘进右侧 chat。
@文件名 可以让 Cursor 把文件内容一并喂给 AI（比如 `@final/01_hello_llm.py`）
```

## AI 给的代码跑不通时——3 步诊断

### Step 1: 别急着复制粘贴报错

先自己看 5 秒报错最后一行：是 `ImportError`、`KeyError`、`AttributeError` 还是别的？

### Step 2: 描述"我以为会发生什么"

```
我跑 _scratch/foo.py，我以为会打印 "hello"，结果看到 [报错最后一行]。
我猜可能是 [你的猜测]。这猜对了吗？如果不对，再给我 1 个候选原因。
```

### Step 3: 还是不行，最小复现

把代码砍到最少（5-10 行）能复现同一个错。砍掉无关后大概率你自己就能看出来。砍完还看不出来，把砍后的最小版贴给 AI。

## 每周末「卡点日志」复盘

走完一周后，打开 `_scratch/journal/`，回答：

1. 这周哪几个任务卡走得最顺？为什么？
2. 哪几个最卡？卡在哪？
3. 哪个 prompt 让 AI 给出了最好的解释？（贴下来当下周复用）
4. 我现在能用一句话讲明白「LCEL / LangChain Memory」吗？讲不出 = 没真懂，下周回头补

## 准备好了？

去 [tutorial/week-1-langchain/01_hello_llm.md](tutorial/week-1-langchain/01_hello_llm.md) 走第一篇。
