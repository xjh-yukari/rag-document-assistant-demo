def split_documents(
    documents: list[dict],
    tokenizer,
    chunk_size: int = 180,
    overlap: int = 30,
) -> list[dict]:

    if not 0 <= overlap < chunk_size:
        raise ValueError("重叠长度必须大于等于0，且小于切块长度")

    chunks = []

    for document in documents:
        text = document["text"]

        # 获取每个 token 在原文中的字符起止位置
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        offsets = encoded["offset_mapping"]

        # 每次向前移动的 token 数量
        step = chunk_size - overlap

        for start in range(0, len(offsets), step):
            end = min(start + chunk_size, len(offsets))

            # 把 token 位置转换成原文字符位置
            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            chunk_text = text[char_start:char_end].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": len(chunks) + 1,
                    "filename": document["filename"],
                    "page": document["page"],
                    "text": chunk_text,
                })

            # 已经处理到原文末尾，不再生成多余的重叠片段
            if end == len(offsets):
                break

    return chunks
