[Uploading README.md…]()
# 基于RAG的智能知识库系统demo

一个用于学习 RAG（检索增强生成）的文档问答项目：上传文档后，在本地完成文本提取、切分、向量化和检索，再按需调用 DeepSeek 官方 API，根据检索到的原文生成回答。

本项目使用 **Python + uv + Streamlit + Sentence Transformers + FAISS + DeepSeek API**。PyCharm 是开发工具，uv 管理项目依赖和 `.venv` 虚拟环境。

## 已实现功能

- 同时上传 PDF、TXT、Markdown，展示已索引文档与片段数量。
- 按 token 切分文本，可调整片段长度和相邻片段重叠长度。
- 使用本地 `sentence-transformers/all-MiniLM-L6-v2` 模型在 CPU 上生成 384 维向量。
- 使用 FAISS 检索相关片段，支持 Top-K 和可选相似度阈值。
- 提供“仅检索”和“RAG 问答”两种模式。
- 展示原文、文件名、PDF 实际页码和相似度，检查回答中的来源编号。
- 将知识库保存到 `data/knowledge_base.npz`，启动新会话时尝试恢复。
- 查看、清空当前会话问答记录，并导出为 Markdown。
- 上传校验、内容去重、损坏文件跳过、PDF 无文字页提示。

## 两种工作模式

| 模式 | 执行内容 | 是否调用 DeepSeek |
| --- | --- | --- |
| 仅检索 | 问题向量化，查找并展示相关原文 | 否 |
| RAG 问答 | 先检索，再把问题和命中片段发给 DeepSeek 生成回答 | 有检索结果时调用 |

文档向量化使用本地模型，不调用 DeepSeek 嵌入接口。DeepSeek 在本项目中负责生成回答。RAG 问答会向官方接口发送问题、命中片段及相应来源信息。

## 环境与依赖

当前开发环境使用 Python 3.11。直接依赖如下，具体版本以 `pyproject.toml` 和 `uv.lock` 为准。

| 依赖 | 用途 |
| --- | --- |
| streamlit | 上传、参数设置、问答页面 |
| pypdf | 提取 PDF 文字和页码 |
| sentence-transformers | 加载本地嵌入模型 |
| numpy | 向量数组和知识库文件读写 |
| faiss-cpu | 向量索引与相似度检索 |
| openai | 通过兼容接口调用 DeepSeek，不是调用 OpenAI 模型 |
| python-dotenv | 读取项目根目录的 `.env` 配置 |

## 安装与启动

### 1. 安装项目依赖

在包含 `pyproject.toml` 和 `uv.lock` 的项目根目录打开 PowerShell：

```powershell
uv sync --locked
uv run python --version
uv run python -c "import sys; print(sys.executable)"
```

`uv sync` 会创建或同步 `.venv`；使用 `uv run` 无需先手动激活环境。已有项目不需要再次执行 `uv init`。

在 PyCharm 中将项目解释器设置为项目目录下的 `.venv\Scripts\python.exe`。

### 2. 准备本地嵌入模型

当前代码设置了 `local_files_only=True`，运行应用时不会自动下载模型，需要当前用户的模型缓存中已经存在：

```text
sentence-transformers/all-MiniLM-L6-v2
```

如果尚未缓存，可在联网环境中主动运行一次以下命令下载模型，然后再启动应用：

```powershell
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"
```

### 3. 配置 DeepSeek

仅检索模式不需要 API 密钥。使用 RAG 问答时，在项目根目录创建或编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=替换为你的真实密钥
DEEPSEEK_MODEL=替换为官方平台当前支持的模型标识
```

上面是配置格式，不能将占位文字原样用作配置。不要将包含真实密钥的 `.env` 提交到 Git。

模型选择顺序：页面非空输入 → 配置中的 `DEEPSEEK_MODEL` → 代码默认值 `deepseek-flash`。同名系统环境变量优先于 `.env`。默认值只是当前代码的回退设置，本次未验证其在线可用性，建议显式填写账户可调用的模型标识。

页面的“DeepSeek 模型名称”只影响回答生成，不会修改本地嵌入模型。接口地址在代码中固定为 `https://api.deepseek.com`。

### 4. 启动页面

```powershell
uv run streamlit run app.py --server.fileWatcherType none
```

通常访问 `http://localhost:8501`，以终端显示的地址为准。上述命令关闭文件监听，修改代码后需要按 `Ctrl+C` 停止，再重新启动。

## 使用步骤

1. 一起选中需要纳入知识库的文档。
2. 按需设置切块参数，默认每片段 180 token，重叠 30 token；重叠必须小于片段长度。
3. 点击“建立知识库”，检查成功数量及跳过文件的提示。
4. 输入问题，先用“仅检索”检查命中的原文是否相关。
5. 选择“RAG 问答”，配置可用模型后提交问题，查看回答及来源。
6. 按需导出问答与来源。

Top-K 默认 3，可选 1～10。只有勾选“过滤低相似度片段”时阈值才生效。相似度不是回答正确率；没有达到条件的片段时不会调用 DeepSeek。

**建立新知识库会替换现有内容，不是追加文档。** 成功后清空当前会话问答记录；构建或保存失败时保留原知识库与记录。知识库保存在磁盘，问答历史仅保存在当前 Streamlit 会话中，需要留存时请导出。

## 上传规则

| 情况 | 行为 |
| --- | --- |
| 超过 20 个文件，或合计超过 100 MiB | 拒绝本次建库 |
| 单文件超过 20 MiB | 跳过该文件并提示 |
| 文件内容完全相同 | 按 SHA-256 去重，保留第一个成功解析的文件 |
| 文件名相同但内容不同 | 跳过后一个文件，提示重命名 |
| 空文件、不支持的格式、解析失败 | 跳过并提示，其他有效文件继续处理 |
| PDF 页面没有可提取文字 | 跳过该页并提示，后续页面保留实际页码 |
| 全部文件不可用 | 建库失败，保留旧知识库 |

容量按 1024 进制计算。TXT/MD 使用 UTF-8，可带 BOM；需要密码的 PDF 请先解密。扫描 PDF 需要先做 OCR，本项目没有实现 OCR。

## 项目结构

```text
rag-document-assistant-demo/
├── app.py                    # Streamlit 页面与流程协调
├── upload_loader.py          # 多文件校验、去重、读取及提示汇总
├── document_loader.py        # PDF/TXT/MD 文本读取
├── text_splitter.py          # 按 token 切块并保留来源
├── embedding_model.py        # 本地模型加载和向量化
├── vector_store.py           # FAISS 检索及知识库保存、恢复
├── rag_answer.py             # DeepSeek 配置、提示词和回答生成
├── citation_checker.py       # 引用编号检查
├── chat_export.py            # 问答及来源导出
├── tests/
│   ├── test_regressions.py    # 自动回归测试，模拟 API
│   └── check_local.py         # 真实本地嵌入模型测试
├── data/                     # 首次保存知识库时创建
│   └── knowledge_base.npz
├── .env                      # 用户自行配置，不应提交
├── pyproject.toml
├── uv.lock
├── README.md
└── 项目流程总结.md
```

## 测试与验收

自动回归测试：

```powershell
uv run python -m unittest discover -s tests -p test_regressions.py -v
```

本地模型测试（需要已有模型缓存）：

```powershell
uv run python tests/check_local.py
```

2026-09-21 的验证结果：18 项回归测试通过，真实本地模型的向量化、临时知识库保存恢复和检索通过。这两种测试不调用收费 API；真实 DeepSeek 在线回答尚未在本次修复中验证。

手动验收清单：

- 上传有文字的 PDF，确认文档数、片段数、页码和原文正常。
- 提问文档明确包含的事实，检查回答与引用对应。
- 提问文档没有的信息，观察是否说明资料不足；这需要实际验收，提示词不能保证绝不编造。
- 提问需要多个片段的问题，检查信息完整性。
- 重启应用，确认知识库恢复后仍可检索。
- 尝试重复、空白或损坏文件，检查提示及旧知识库保留行为。
- 配置真实 DeepSeek 密钥与可用模型，验证一次在线问答。

## 当前限制与后续方向

- 无 OCR、图片理解、重排序或关键词与向量混合检索。
- 当前模型对中文资料的检索效果需要结合实际文档评估。
- 问答记录用于展示和导出；历史对话没有作为上下文传给模型，每次提问独立处理。
- 引用检查只验证编号范围并提示缺失引用，不验证答案事实是否被原文支持。
- 当前保存格式未记录嵌入模型身份；更换嵌入模型后必须重建知识库，即使维度相同也不能混用。
- 当前是单个本地知识库 Demo，没有用户隔离，不适合直接作为多人共享服务部署。

优先完成真实 API 验收；后续可先补充嵌入模型身份校验，再按实际需求增加功能。

原理、模块关系及复刻顺序见 [项目流程总结](项目流程总结.md)。修复细节见 [检查与修复记录](REVIEW_2026-09-21.md)。
#   r a g - d o c u m e n t - a s s i s t a n t - d e m o 
 
 
