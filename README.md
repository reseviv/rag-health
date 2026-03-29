# RAG Health

A healthcare-focused Retrieval-Augmented Generation (RAG) project designed to help users find, understand, and navigate trusted clinical guideline content more effectively. The project uses document chunking, text normalisation, embeddings, vector search with FAISS, and answer generation to support evidence-grounded responses from health documents.

> Current focus: building a reliable RAG pipeline for health guideline documents, with future extension toward GraphRAG for richer knowledge relationships and multi-hop retrieval.

---

## Project Overview

RAG Health is designed to retrieve relevant information from medical and health guideline documents and use those results to support grounded answers. The system is aimed at reducing hallucinations and improving traceability by linking outputs back to source material.

This project is especially useful for:
- clinical reading support
- structured exploration of healthcare guidelines
- evidence-grounded question answering
- future GraphRAG experimentation in the health domain

At the current stage, the project focuses on:
- PDF ingestion and preprocessing
- text chunking and metadata generation
- sentence embedding creation
- FAISS-based similarity search
- retrieval pipeline for grounded response generation

---

## Key Features

- **Health document ingestion** from PDF files
- **Text normalisation** to clean Unicode issues and improve consistency
- **Chunking pipeline** with overlap for better retrieval quality
- **Metadata tracking** such as source file, section, and chunk index
- **Embedding generation** using SentenceTransformers
- **Vector search** with FAISS for fast retrieval
- **Grounded answer generation** based on retrieved passages
- **Designed for extension to GraphRAG**

---

## Repository Structure

```text
rag-health/
├── data/
│   ├── raw/                  # original PDFs
│   │   ├── abortion.pdf
│   │   ├── hiv.pdf
│   │   └── infertility.pdf
│   └── processed/
│       └── chunks.json       # processed chunks with metadata
├── index/
│   ├── faiss.index           # FAISS vector index
│   └── metadata.json         # metadata aligned to indexed chunks
├── src/
│   ├── chunk.py              # chunking + metadata creation
│   ├── preprocess.py         # text cleaning / normalisation
│   ├── embed.py              # embedding generation
│   ├── retrieve.py           # FAISS retrieval logic
│   └── rag.py                # answer generation pipeline
├── app/                      # optional frontend / interface
├── requirements.txt
└── README.md
```

---

## System Workflow

```text
PDF Documents
   ↓
Text Extraction
   ↓
Normalisation & Cleaning
   ↓
Chunking + Metadata
   ↓
Embedding Generation
   ↓
FAISS Indexing
   ↓
User Query
   ↓
Query Embedding
   ↓
Top-k Retrieval
   ↓
LLM / Response Generator
   ↓
Grounded Answer
```

---

## Tech Stack

- **Python**
- **SentenceTransformers** (`all-MiniLM-L6-v2`)
- **FAISS** for vector similarity search
- **JSON** for processed chunk storage and metadata
- **PDF parsing tools** such as `pypdf` or similar
- **Optional LLM integration** for final answer generation
