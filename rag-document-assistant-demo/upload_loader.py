from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from document_loader import load_document


MAX_FILES = 20
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 100 * 1024 * 1024


def load_uploaded_documents(uploaded_files):
    if not uploaded_files:
        raise ValueError("请至少选择一个文件")
    if len(uploaded_files) > MAX_FILES:
        raise ValueError("一次最多上传20个文件")
    if sum(file.size for file in uploaded_files) > MAX_TOTAL_BYTES:
        raise ValueError("本次上传文件的总大小不能超过100 MB")

    all_documents = []
    warnings = []
    seen_contents = {}
    seen_names = set()

    with TemporaryDirectory() as temp_dir:
        for number, uploaded_file in enumerate(uploaded_files):
            filename = Path(uploaded_file.name.replace("\\", "/")).name
            if uploaded_file.size > MAX_FILE_BYTES:
                warnings.append(f"{filename} 超过单文件20 MB限制，已跳过。")
                continue

            content = uploaded_file.getvalue()
            if len(content) > MAX_FILE_BYTES:
                warnings.append(f"{filename} 超过单文件20 MB限制，已跳过。")
                continue
            if not content:
                warnings.append(f"{filename} 是空文件，已跳过。")
                continue

            suffix = Path(filename).suffix.lower()
            if suffix not in {".pdf", ".txt", ".md"}:
                warnings.append(f"{filename} 格式不支持，已跳过。")
                continue

            content_id = sha256(content).hexdigest()
            if content_id in seen_contents:
                original_name = seen_contents[content_id]
                warnings.append(f"{filename} 与 {original_name} 内容完全相同，已跳过重复文件。")
                continue
            name_key = filename.casefold()
            if name_key in seen_names:
                warnings.append(
                    f"{filename} 与已读取文件重名，但内容不同。已跳过；请重命名后重新上传。"
                )
                continue

            temp_path = Path(temp_dir) / f"upload_{number}{suffix}"
            file_warnings = []
            try:
                temp_path.write_bytes(content)
                documents = load_document(temp_path, warnings=file_warnings)
                for document in documents:
                    document["filename"] = filename
                all_documents.extend(documents)
                seen_contents[content_id] = filename
                seen_names.add(name_key)
            except Exception as error:
                warnings.append(f"{filename} 读取失败：{error}")

            warnings.extend(f"{filename}：{message}" for message in file_warnings)

    if not all_documents:
        raise ValueError("没有读取到任何有效文档。" + "；".join(warnings))
    return all_documents, warnings
