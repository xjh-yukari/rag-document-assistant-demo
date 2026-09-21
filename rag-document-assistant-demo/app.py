from pathlib import Path

import streamlit as st
from openai import (
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIStatusError,
)

from text_splitter import split_documents
from embedding_model import TextEmbedder
from vector_store import VectorStore
from rag_answer import generate_answer
from citation_checker import check_citations
from upload_loader import load_uploaded_documents
from chat_export import export_history

st.set_page_config(
    page_title="我的 RAG 知识库",
    page_icon="📚",
    layout="wide",
)


KNOWLEDGE_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "knowledge_base.npz"
)

# 缓存模型，避免每次页面操作都重新加载
@st.cache_resource
def get_embedder():
    return TextEmbedder()


# 初始化当前会话的数据
if "store" not in st.session_state:
    st.session_state.store = None


    if KNOWLEDGE_PATH.is_file():
        try:
            restored_store = VectorStore.load(KNOWLEDGE_PATH)

            st.session_state.store = restored_store


        except Exception:
            st.warning(
                "已有知识库加载失败，请重新上传文档并建立知识库。"
            )

if "history" not in st.session_state:
    st.session_state.history = []

if "upload_warnings" not in st.session_state:
    st.session_state.upload_warnings = []


st.title("📚 我的 RAG 知识库")
st.caption("上传文档，检索原文，再根据来源回答问题。")


# ---------- 文档上传与建库 ----------

uploaded_files = st.file_uploader(
    "选择一份或多份文档",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

st.caption(
    "建立知识库会替换当前内容。需要保留的文档，请一起选中后建立。"
)
st.caption(
    "最多20个文件，单文件20 MB，总计100 MB（按1024进制计算）。"
    "重复内容会跳过，扫描PDF需要先进行OCR。"
)

with st.expander("切块设置"):
    chunk_size = st.number_input(
        "每个片段的 token 数",
        min_value=32,
        max_value=224,
        value=180,
        step=1,
    )

    overlap = st.number_input(
        "相邻片段重叠的 token 数",
        min_value=0,
        max_value=128,
        value=30,
        step=1,
    )

    st.caption(
        "重叠长度必须小于片段长度。修改后需要重新建立知识库。"
    )

if st.button(
    "建立知识库",
    disabled=not uploaded_files,
):
    st.session_state.upload_warnings = []
    try:
        with st.spinner("正在读取文档并建立索引……"):
            if not 0 <= int(overlap) < int(chunk_size):
                raise ValueError("重叠长度必须小于片段长度")

            documents, load_warnings = load_uploaded_documents(
                uploaded_files
            )

            st.session_state.upload_warnings = load_warnings
            embedder = get_embedder()

            chunks = split_documents(
                documents,
                tokenizer=embedder.model.tokenizer,
                chunk_size=int(chunk_size),
                overlap=int(overlap),
            )

            if not chunks:
                raise ValueError("文档没有可用片段")

            texts = [chunk["text"] for chunk in chunks]

            # 检查实际输入长度
            for number, text in enumerate(texts, start=1):
                token_ids = embedder.model.tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]

                if len(token_ids) > embedder.model.max_seq_length:
                    raise ValueError(
                        f"片段 {number} 超过模型长度限制"
                    )

            vectors = embedder.encode(texts)
            new_store = VectorStore(chunks, vectors)
            new_store.save(KNOWLEDGE_PATH)

            # 全部成功后，再替换当前知识库
            st.session_state.store = new_store
            st.session_state.history = []

        document_count = len({chunk["filename"] for chunk in chunks})
        st.success(f"知识库建立成功：{document_count} 份文档，{len(chunks)} 个片段。")

    except Exception as error:
        # 页面边界：显示错误，避免整个页面中断
        st.error(f"建立知识库失败：{error}")

for warning in st.session_state.upload_warnings:
    st.warning(warning)


# ---------- 展示当前知识库 ----------

store = st.session_state.store

if store is None:
    st.info("请先上传文档，并点击“建立知识库”。")

else:
    # 去重，同时保留文件出现的顺序
    filenames = list(
        dict.fromkeys(
            chunk["filename"]
            for chunk in store.chunks
        )
    )

    st.write("已索引文档数：", len(filenames))
    st.write("可检索片段数：", store.index.ntotal)

    with st.expander("查看文档列表", expanded=True):
        for filename in filenames:
            st.text(filename)


# ---------- 提问区域 ----------

with st.form("question_form"):
    question = st.text_input("请输入关于文档的问题")

    mode = st.radio(
        "工作模式",
        ["仅检索", "RAG 问答"],
        horizontal=True,
    )

    top_k = st.slider(
        "检索片段数",
        min_value=1,
        max_value=10,
        value=3,
    )
    filter_enabled = st.checkbox(
        "过滤低相似度片段",
        value=False,
    )

    threshold = st.slider(
        "最低相似度",
        min_value=-1.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
    )

    st.caption(
        "只有勾选过滤后，阈值才会生效。相似度不是正确率。"
    )
    model_name = st.text_input(
        "DeepSeek 模型名称",
        value="",
        help="留空使用 .env 中的 DEEPSEEK_MODEL；填写账户支持的官方模型名称。",
    )

    submitted = st.form_submit_button("提交问题")

st.caption(
    "仅检索在本地运行；RAG 问答会将问题和命中片段发送到 DeepSeek。"
)


# ---------- 执行检索与问答 ----------

if submitted:
    question = question.strip()

    if store is None:
        st.warning("请先建立知识库。")

    elif not question:
        st.warning("请输入问题。")

    else:
        results = []

        try:
            with st.spinner("正在检索……"):
                embedder = get_embedder()

                question_ids = embedder.model.tokenizer(
                    question,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]

                if len(question_ids) > embedder.model.max_seq_length:
                    raise ValueError("问题过长，请缩短后重试")

                question_vectors = embedder.encode([question])

                results = store.search(
                    question_vectors,
                    top_k=top_k,
                    min_score=threshold if filter_enabled else -1.0,
                )

            warnings = []

            if not results:
                answer = (
                    "没有找到满足当前条件的片段。"
                    "可以尝试降低相似度阈值，或换一种方式提问。"
                )

            elif mode == "仅检索":
                answer = (
                    f"找到 {len(results)} 个相关片段，"
                    "请查看下方原文。此次没有调用 DeepSeek。"
                )

            else:
                with st.spinner("DeepSeek 正在生成回答……"):
                    answer = generate_answer(
                        question,
                        results,
                        model_name=model_name,
                    )

                answer, warnings = check_citations(
                    answer,
                    source_count=len(results),
                )

            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "warnings": warnings,
                "sources": results,
            })

        except AuthenticationError:
            st.error("身份验证失败，请检查本地 API Key。")

        except RateLimitError:
            st.error("请求受到限制，请稍后重试并检查账户状态。")

        except APIConnectionError:
            # 也包含 SDK 的超时异常
            st.error("连接失败或请求超时，请检查网络后重试。")

        except APIStatusError as error:
            st.error(
                f"DeepSeek 请求失败，HTTP 状态码：{error.status_code}"
            )

        except ValueError as error:
            st.error(str(error))

        except Exception:
            st.error("处理失败，请检查本地模型和运行环境。")


# ---------- 显示问答记录 ----------

st.subheader("问答记录")

if not st.session_state.history:
    st.info("还没有问答记录，请提交一个问题。")

else:
    if st.button("清空问答记录"):
        st.session_state.history = []
        st.rerun()

    st.download_button(
        label="导出问答与来源",
        data=export_history(st.session_state.history),
        file_name="rag-chat-history.md",
        mime="text/markdown",
    )

    for record in st.session_state.history:
        with st.chat_message("user"):
            st.write(record["question"])

        with st.chat_message("assistant"):
            st.markdown(record["answer"])

            for warning in record["warnings"]:
                st.warning(warning)

            for number, source in enumerate(
                record["sources"],
                start=1,
            ):
                page = source["page"]

                location = (
                    f"PDF 第 {page} 页"
                    if page is not None
                    else "文本文件"
                )

                title = (
                    f"[{number}] "
                    f"{source['filename']} · {location}"
                )

                with st.expander(title):
                    st.write(
                        "相似度：",
                        round(source["score"], 4),
                    )
                    st.text(source["text"])
