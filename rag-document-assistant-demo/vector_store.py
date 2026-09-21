import faiss
import numpy as np
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

class VectorStore:
    def __init__(self, chunks: list[dict], vectors: np.ndarray):
        # 转换成 FAISS 使用的连续 float32 数组
        vectors = np.array(
            vectors,
            dtype="float32",
            order="C",
            copy=True,
        )

        if vectors.ndim != 2:
            raise ValueError("文档向量必须是二维数组")

        if len(chunks) == 0:
            raise ValueError("没有可建立索引的文档片段")

        if len(chunks) != vectors.shape[0]:
            raise ValueError("片段数量与向量数量不一致")

        self.chunks = chunks

        # 例如 (6, 384) 中的 384
        dimension = vectors.shape[1]

        # 归一化后，内积可以用来计算余弦相似度
        faiss.normalize_L2(vectors)

        # 创建精确内积检索索引
        self.index = faiss.IndexFlatIP(dimension)

        # 按当前顺序加入文档向量
        self.index.add(vectors)

    def search(
            self,
            question_vectors: np.ndarray,
            top_k: int = 3,
            min_score: float = -1.0,
    ) -> list[dict]:

        if top_k < 1:
            raise ValueError("top_k 必须大于等于1")
        if not -1.0 <= min_score <= 1.0:
            raise ValueError("相似度阈值必须在 -1 到 1 之间")

        query = np.array(
            question_vectors,
            dtype="float32",
            order="C",
            copy=True,
        )

        # 这个方法一次处理一个问题
        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("问题向量形状应为 (1, 向量维度)")

        if query.shape[1] != self.index.d:
            raise ValueError("问题向量与文档向量维度不一致")

        faiss.normalize_L2(query)

        # 返回数量不能超过已有片段数量
        k = min(top_k, self.index.ntotal)

        scores, positions = self.index.search(query, k)

        results = []

        for score, position in zip(scores[0], positions[0]):
            if float(score) < min_score:
                continue

            chunk = self.chunks[int(position)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "text": chunk["text"],
                "score": float(score),
            })

        return results

    def save(self, file_path: str | Path):
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 将 FAISS 索引转换成字节数组
        index_bytes = faiss.serialize_index(self.index)

        # 将片段和基本信息转换成 JSON 字符串
        metadata = json.dumps(
            {
                "version": 1,
                "dimension": self.index.d,
                "chunks": self.chunks,
            },
            ensure_ascii=False,
        )

        # 先写入同一目录中的临时文件
        with NamedTemporaryFile(
                dir=path.parent,
                suffix=".tmp",
                delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with temp_path.open("wb") as file:
                np.savez_compressed(
                    file,
                    index=index_bytes,
                    metadata=np.array(metadata),
                )

            # 完整写入后，再替换正式文件
            os.replace(temp_path, path)

        finally:
            # 写入或替换失败时，清理临时文件
            if temp_path.exists():
                temp_path.unlink()

    @classmethod
    def load(cls, file_path: str | Path):
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"找不到知识库文件：{path}")

        # 禁止使用 pickle 反序列化
        with np.load(path, allow_pickle=False) as saved:
            index_bytes = np.array(
                saved["index"],
                dtype="uint8",
                copy=True,
            )

            metadata = json.loads(
                saved["metadata"].item()
            )

        if metadata["version"] != 1:
            raise ValueError("不支持这个知识库文件版本")

        index = faiss.deserialize_index(index_bytes)
        chunks = metadata["chunks"]

        if index.ntotal != len(chunks):
            raise ValueError("索引数量与原文片段数量不一致")

        if index.d != metadata["dimension"]:
            raise ValueError("索引维度与保存的信息不一致")

        # 创建对象，但不重新执行建库初始化
        store = cls.__new__(cls)
        store.index = index
        store.chunks = chunks

        return store
