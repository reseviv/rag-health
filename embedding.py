from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  # biomedical model trained on PubMed + MS MARCO retrieval task

def load_chunks(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f: 
        return json.load(f)


def build_embeddings(texts: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product works well with normalized embeddings
    index.add(embeddings)
    return index


def build_metadata(chunks: list[dict]) -> list[dict]:
    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "faiss_id": i,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk.get("document_id"),
            "source_file": chunk.get("source_file"),
            "source": chunk.get("source"),
            "topic": chunk.get("topic"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "chunk_index": chunk.get("chunk_index"),
            "text": chunk["text"]
        })
    return metadata


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)

    if not chunks:
        raise ValueError("No chunks found in chunks.json")

    texts = [c["text"] for c in chunks]

    print(f"Creating embeddings with model: {MODEL_NAME}")
    embeddings = build_embeddings(texts)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings)

    print("Preparing metadata...")
    metadata = build_metadata(chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index -> {INDEX_FILE}")
    print(f"Saved metadata -> {METADATA_FILE}")
    print(f"Indexed {len(metadata)} chunks")
    print(f"Embedding dimension: {embeddings.shape[1]}")


if __name__ == "__main__":
    main()