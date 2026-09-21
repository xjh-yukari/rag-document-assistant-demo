from pathlib import Path

from pypdf import PdfReader


def load_document(
    file_path: str | Path,
    warnings: list[str] | None = None,
) -> list[dict]:
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")

    suffix = path.suffix.lower()
    documents = []

    if suffix == ".pdf":
        reader = PdfReader(path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ValueError("PDF 已加密，请先解密后再上传")

        # 页码从 1 开始，保持与 PDF 的物理页序号一致
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                message = (
                    f"第 {page_number} 页没有提取到文字，已跳过。"
                    "该页可能为空白页或扫描页；当前版本不支持 OCR。"
                )
                if warnings is not None:
                    warnings.append(message)
                else:
                    print(message)
                continue

            documents.append({
                "filename": path.name,
                "page": page_number,
                "text": text,
            })

    elif suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8-sig").strip()

        if text:
            documents.append({
                "filename": path.name,
                "page": None,
                "text": text,
            })

    else:
        raise ValueError("目前只支持 PDF、TXT 和 MD 文件")

    if not documents:
        raise ValueError("没有提取到有效文字，请检查文件是否为空或属于扫描件")

    return documents
