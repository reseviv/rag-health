# rag-health

Retrieval-Augmented Generation (RAG) project for health PDFs using FAISS indexing and chunk metadata.

## Project structure

```text
rag-health/
  data/
    raw/            # input PDFs (ignored by git by default)
    processed/      # generated chunks/embeddings (ignored)
  index/            # FAISS index + metadata (ignored)
  src/              # core library code (chunking, embedding, retrieval)
  app/              # API/UI (e.g., FastAPI/Streamlit/React)
