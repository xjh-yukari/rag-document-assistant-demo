"""Run: uv run python -m unittest discover -s tests -p test_regressions.py -v

All files are synthetic. Model/API calls are mocked; user data and keys are not used.
"""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import re
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from document_loader import load_document
from rag_answer import generate_answer
from upload_loader import load_uploaded_documents
from vector_store import VectorStore


class Upload:
    def __init__(self, name, content, size=None):
        self.name, self.content = name, content
        self.size = len(content) if size is None else size

    def getvalue(self):
        return self.content


def pdf_bytes(texts, password=None):
    writer = PdfWriter()
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    font_ref = writer._add_object(font)
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        if text:
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})
            })
            stream = DecodedStreamObject()
            stream.set_data(f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode())
            page[NameObject("/Contents")] = writer._add_object(stream)
    if password:
        writer.encrypt(password)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def fixture_store():
    return VectorStore([
        {"chunk_id": 1, "filename": "old.md", "page": None, "text": "hotel policy"},
        {"chunk_id": 2, "filename": "old.pdf", "page": 3, "text": "learning budget"},
    ], np.array([[1, 0], [0, 1]], dtype="float32"))


class UploadTests(unittest.TestCase):
    def test_empty_selection(self):
        with self.assertRaisesRegex(ValueError, "至少"):
            load_uploaded_documents([])

    def test_file_count_limit(self):
        with self.assertRaisesRegex(ValueError, "20个"):
            load_uploaded_documents([Upload(f"{i}.txt", b"ok") for i in range(21)])

    def test_total_size_limit_before_reading(self):
        file = Upload("a.txt", b"ok", size=101 * 1024 * 1024)
        with patch.object(file, "getvalue", side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError, "总大小"):
                load_uploaded_documents([file])

    def test_single_oversize_skipped(self):
        docs, warnings = load_uploaded_documents([
            Upload("big.txt", b"large", size=21 * 1024 * 1024), Upload("ok.md", b"valid")
        ])
        self.assertEqual([d["filename"] for d in docs], ["ok.md"])
        self.assertIn("20 MB", warnings[0])

    def test_actual_size_also_checked(self):
        with patch("upload_loader.MAX_FILE_BYTES", 4):
            docs, warnings = load_uploaded_documents([
                Upload("big.txt", b"12345", size=1), Upload("ok.md", b"ok")
            ])
        self.assertEqual(len(docs), 1)
        self.assertIn("big.txt", warnings[0])

    def test_duplicates_different_names(self):
        docs, warnings = load_uploaded_documents([
            Upload("first.txt", b"same"), Upload("renamed.md", b"same")
        ])
        self.assertEqual(len(docs), 1)
        self.assertIn("内容完全相同", warnings[0])

    def test_conflicting_names_and_failed_names(self):
        docs, warnings = load_uploaded_documents([
            Upload("a.txt", b""), Upload("a.txt", b"first"), Upload("A.TXT", b"second")
        ])
        self.assertEqual(docs[0]["text"], "first")
        self.assertEqual(len(docs), 1)
        self.assertTrue(any("重名" in w for w in warnings))

    def test_invalid_files_do_not_block_good_file(self):
        docs, warnings = load_uploaded_documents([
            Upload("bad.pdf", b"broken"), Upload("empty.txt", b""),
            Upload("binary.exe", b"binary"), Upload("good.md", "有效内容".encode()),
        ])
        self.assertEqual(docs[0]["text"], "有效内容")
        self.assertEqual(len(warnings), 3)

    def test_all_invalid(self):
        with self.assertRaisesRegex(ValueError, "没有读取到任何有效"):
            load_uploaded_documents([Upload("empty.md", b"   ")])

    def test_pdf_page_warning_and_physical_number(self):
        docs, warnings = load_uploaded_documents([
            Upload("policy.pdf", pdf_bytes(["First page", "", "Third page"]))
        ])
        self.assertEqual([d["page"] for d in docs], [1, 3])
        self.assertTrue(all(d["filename"] == "policy.pdf" for d in docs))
        self.assertIn("policy.pdf：第 2 页", warnings[0])
        self.assertIn("OCR", warnings[0])

    def test_encrypted_pdf(self):
        docs, warnings = load_uploaded_documents([
            Upload("secret.pdf", pdf_bytes(["private"], password="password")),
            Upload("public.md", b"public"),
        ])
        self.assertEqual(docs[0]["filename"], "public.md")
        self.assertIn("加密", warnings[0])

    def test_direct_loader_empty_warning_list_and_utf8_bom(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.pdf"
            path.write_bytes(pdf_bytes(["", "Text"]))
            warnings = []
            docs = load_document(path, warnings=warnings)
            self.assertEqual(docs[0]["page"], 2)
            self.assertEqual(len(warnings), 1)
            path = Path(tmp) / "a.md"
            path.write_bytes("中文".encode("utf-8-sig"))
            self.assertEqual(load_document(path)[0]["text"], "中文")


class AnswerTests(unittest.TestCase):
    def test_one_call_and_custom_model(self):
        result = [{"filename": "policy.pdf", "page": 1, "text": "600 yuan"}]
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="600元。[1]"))])
        with patch("rag_answer.dotenv_values", return_value={"DEEPSEEK_API_KEY": "dummy-test"}), \
             patch.dict("os.environ", {}, clear=True), patch("rag_answer.OpenAI") as factory:
            create = factory.return_value.__enter__.return_value.chat.completions.create
            # A second request fails immediately instead of recursing indefinitely.
            create.side_effect = [response, AssertionError("Unexpected second API request")]
            self.assertEqual(generate_answer("limit?", result, " chosen-model "), "600元。[1]")
            self.assertEqual(create.call_count, 1)
            self.assertEqual(create.call_args.kwargs["model"], "chosen-model")

    def test_config_reloads_without_environment_pollution(self):
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        result = [{"filename": "a.md", "page": None, "text": "text"}]
        with patch.dict("os.environ", {}, clear=True), patch("rag_answer.OpenAI") as factory, \
             patch("rag_answer.dotenv_values", side_effect=[
                 {"DEEPSEEK_API_KEY": "first", "DEEPSEEK_MODEL": "first-model"},
                 {"DEEPSEEK_API_KEY": "second", "DEEPSEEK_MODEL": "second-model"},
             ]):
            factory.return_value.__enter__.return_value.chat.completions.create.return_value = response
            generate_answer("q", result)
            generate_answer("q", result)
            self.assertEqual([c.kwargs["api_key"] for c in factory.call_args_list], ["first", "second"])

    def test_empty_sources_do_not_call_api(self):
        with patch("rag_answer.OpenAI") as factory:
            self.assertIn("没有检索", generate_answer("q", []))
            factory.assert_not_called()


class PersistenceTests(unittest.TestCase):
    def test_roundtrip_and_failed_replace(self):
        store = fixture_store()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "中文目录" / "knowledge.npz"
            store.save(path)
            original = path.read_bytes()
            restored = VectorStore.load(path)
            self.assertEqual(store.chunks, restored.chunks)
            self.assertEqual(restored.search(np.array([[0, 1]]), min_score=0.5)[0]["page"], 3)
            with patch("vector_store.os.replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    store.save(path)
            self.assertEqual(path.read_bytes(), original)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class FakeEmbedder:
    def __init__(self):
        self.model = SimpleNamespace(tokenizer=FakeTokenizer(), max_seq_length=256)

    def encode(self, texts):
        return np.array([[1, 0] for _ in texts], dtype="float32")


class PageTests(unittest.TestCase):
    def test_page_build_query_model_forwarding_and_history(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        uploads = [Upload("one.md", b"hotel limit 600"), Upload("copy.md", b"hotel limit 600")]
        with patch("streamlit.file_uploader", return_value=uploads), \
             patch("embedding_model.TextEmbedder", FakeEmbedder), \
             patch("vector_store.VectorStore.save") as save, \
             patch("rag_answer.generate_answer", return_value="600元。[1]") as answer:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
            app.session_state["store"] = fixture_store()
            app.session_state["history"] = []
            app.run()
            self.assertFalse(app.exception)
            next(b for b in app.button if b.label == "建立知识库").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["store"].index.ntotal, 1)
            self.assertTrue(any("重复文件" in w.value for w in app.warning))
            save.assert_called_once()
            next(t for t in app.text_input if t.label == "请输入关于文档的问题").set_value("hotel limit?")
            next(t for t in app.text_input if t.label == "DeepSeek 模型名称").set_value("chosen-model")
            app.radio[0].set_value("RAG 问答")
            next(b for b in app.button if b.label == "提交问题").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(answer.call_args.kwargs["model_name"], "chosen-model")
            self.assertEqual(len(app.session_state["history"]), 1)
            self.assertTrue(any("重复文件" in w.value for w in app.warning))
            next(b for b in app.button if b.label == "清空问答记录").click().run()
            self.assertFalse(app.session_state["history"])
            self.assertEqual(app.session_state["store"].index.ntotal, 1)
        st.cache_resource.clear()

    def test_invalid_upload_keeps_old_store_without_loading_model(self):
        from streamlit.testing.v1 import AppTest
        with patch("streamlit.file_uploader", return_value=[Upload("empty.txt", b"")]), \
             patch("embedding_model.TextEmbedder", side_effect=AssertionError("must not load")), \
             patch("vector_store.VectorStore.save") as save:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20)
            app.session_state["store"] = fixture_store()
            app.session_state["history"] = [{"question": "old", "answer": "old", "sources": [], "warnings": []}]
            app.run()
            next(b for b in app.button if b.label == "建立知识库").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertEqual(app.session_state["store"].index.ntotal, 2)
            self.assertEqual(len(app.session_state["history"]), 1)
            save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
