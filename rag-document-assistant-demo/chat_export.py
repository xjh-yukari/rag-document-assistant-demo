def export_history(history: list[dict]) -> str:
    lines = [
        "# RAG 问答记录",
        "",
        "每轮问题独立检索，来源编号仅在该轮内有效。",
        "",
    ]

    for turn, record in enumerate(history, start=1):
        lines.extend([
            f"## 第 {turn} 轮",
            "",
            "### 问题",
            "",
            record["question"],
            "",
            "### 回答",
            "",
            record["answer"],
            "",
        ])

        if record["warnings"]:
            lines.extend([
                "### 检查提示",
                "",
            ])

            for warning in record["warnings"]:
                lines.append(f"- {warning}")

            lines.append("")

        if record["sources"]:
            lines.extend([
                "### 本轮检索来源",
                "",
            ])

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

            lines.extend([
                f"#### [{number}] {source['filename']} · {location}",
                "",
                f"相似度：{source['score']:.4f}",
                "",
            ])

            # 将原文作为 Markdown 引用块保存
            for text_line in source["text"].splitlines():
                lines.append(f"> {text_line}")

            lines.append("")

    return "\n".join(lines)