

class TextEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        # 创建对象时加载一次模型
        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            local_files_only=True,
        )

    def encode(self, texts: list[str]):
        if not texts:
            raise ValueError("需要至少一段文字才能生成向量")

        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return vectors
