"""
embedding.py — Build FAISS index using MedCPT-Article-Encoder (run once after chunk.py)

Changes from v1:
  - Encoder: ncats/MedCPT-Article-Encoder (was pritamdeka/S-PubMedBert-MS-MARCO)
  - Asymmetric: articles encoded here with Article-Encoder;
    queries encoded at retrieval time with MedCPT-Query-Encoder
  - Uses HuggingFace transformers directly (not sentence-transformers)

Usage:
    pip install transformers torch faiss-cpu
    python embedding.py
"""

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(1)

CHUNKS_FILE = Path("data/processed/chunks.json")
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faiss.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MAX_LENGTH = 512
BATCH_SIZE = 32


def load_chunks(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_embeddings(texts: list[str]) -> np.ndarray:
    print(f"Loading {ARTICLE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL)
    model.eval()

    all_embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
        )

        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

        all_embeddings.append(embeddings.cpu().numpy())

        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  Encoded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    return np.concatenate(all_embeddings, axis=0).astype("float32")


def build_metadata(chunks: list[dict]) -> list[dict]:
    return [
        {
            "faiss_id": i,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk.get("document_id"),
            "source_file": chunk.get("source_file"),
            "source": chunk.get("source"),
            "topic": chunk.get("topic"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "chunk_index": chunk.get("chunk_index"),
            "text": chunk["text"],
        }
        for i, chunk in enumerate(chunks)
    ]


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks(CHUNKS_FILE)
    if not chunks:
        raise ValueError("No chunks found.")

    texts = [c["text"] for c in chunks]

    print(f"Building embeddings for {len(texts)} chunks...")
    embeddings = build_embeddings(texts)

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))

    metadata = build_metadata(chunks)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved FAISS index  -> {INDEX_FILE}  ({len(texts)} vectors, dim={dim})")
    print(f"Saved metadata     -> {METADATA_FILE}")


if __name__ == "__main__":
    main()
