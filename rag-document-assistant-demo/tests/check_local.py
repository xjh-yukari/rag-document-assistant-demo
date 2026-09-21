"""Offline smoke test: cached model + synthetic uploads + FAISS round-trip."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding_model import TextEmbedder
from text_splitter import split_documents
from upload_loader import load_uploaded_documents
from vector_store import VectorStore


class Upload:
    def __init__(self, name, content):
        self.name = name
        self.content = content
        self.size = len(content)

    def getvalue(self):
        return self.content


def main():
    policy = b"The hotel reimbursement limit is CNY 600 per person per night."
    uploads = [
        Upload("hotel.md", policy), Upload("duplicate.txt", policy),
        Upload("parking.txt", b"Visitors receive two hours of free parking."),
    ]
    documents, warnings = load_uploaded_documents(uploads)
    assert len(documents) == 2 and len(warnings) == 1
    embedder = TextEmbedder()
    chunks = split_documents(documents, embedder.model.tokenizer)
    texts = [c["text"] for c in chunks]
    assert all(len(embedder.model.tokenizer(t)["input_ids"]) <= embedder.model.max_seq_length for t in texts)
    vectors = embedder.encode(texts)
    store = VectorStore(chunks, vectors)
    question = embedder.encode(["What is the hotel reimbursement limit?"])
    with TemporaryDirectory(prefix="rag-demo-check-") as tmp:
        path = Path(tmp) / "knowledge.npz"
        store.save(path)
        restored = VectorStore.load(path)
        hits = restored.search(question, top_k=2)
        assert hits[0]["filename"] == "hotel.md"
        assert "600" in hits[0]["text"]
    print(f"PASS: duplicate skipped; {len(chunks)} chunks; {vectors.shape[1]} dimensions")
    print("PASS: saved/reloaded index retrieves hotel.md with the 600 CNY fact")
    print("No API requests, model downloads or changes to the user's knowledge base.")


if __name__ == "__main__":
    main()
