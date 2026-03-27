from pathlib import Path

import faiss  
import numpy as np
INDEX_DIR = Path("index")
INDEX_FILE = INDEX_DIR / "faiss.index"

index = faiss.read_index(str(INDEX_FILE))
print("Total vectors:", index.ntotal)
print("Dimension:", index.d)

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# load faiss index
index = faiss.read_index("index/faiss.index")

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load metadata that maps FAISS row IDs -> text chunk
with open("index/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

query = "What are the WHO guidelines for infertility?"
q_emb = model.encode([query], normalize_embeddings=True).astype("float32")

# search top 5
scores, ids = index.search(q_emb, 5)

for score, idx in zip(scores[0], ids[0]):
    if idx == -1:
        continue
    print("ID:", idx)
    print("Score:", score)
    print("Text:", metadata[idx]["text"])
    print("Source:", metadata[idx].get("source_file"))
    print("-" * 50)