# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-Health is a research project comparing orchestration strategies and retrieval methods for health-domain question answering, grounded in WHO clinical guidelines (HIV, Infertility, Abortion Care PDFs).

**Core research question**: Which combination of LLM orchestration mode and retrieval method performs best on clinical QA tasks?

**Team**: 2 people · **Repo**: https://github.com/reseviv/rag-health
**Stack**: Python · Dynamiq · FAISS · LightRAG · GraphRAG-Benchmark

---

## Experiment Design

The project runs a matrix of conditions evaluated against GraphRAG-Benchmark's medical dataset:

```
4 Orchestration Modes (Dynamiq)
  × 2 Retrieval Methods
  = 8 experimental conditions

Orchestration modes: Single | Linear | Adaptive | Graph
Retrieval methods:   Vanilla RAG (FAISS) | GraphRAG (LightRAG)

Evaluated on 4 task types (GraphRAG-Benchmark medical):
  - Fact Retrieval
  - Complex Reasoning
  - Contextual Summarization
  - Creative Generation
```

---

## Architecture

```
┌──────────────────────────────────────────────┐
│  ORCHESTRATION LAYER                         │
│  Dynamiq                                     │
│  Single / Linear / Adaptive / Graph          │
└──────────────────┬───────────────────────────┘
                   │ agents call tools
┌──────────────────▼───────────────────────────┐
│  RETRIEVAL LAYER                             │
│  Option A: Vanilla RAG (FAISS)               │
│  Option B: GraphRAG (LightRAG)               │
└──────────────────┬───────────────────────────┘
                   │ answers evaluated against
┌──────────────────▼───────────────────────────┐
│  TEST DATA + EVALUATION                      │
│  GraphRAG-Benchmark (medical dataset)        │
│  ├── Datasets/Questions/  → test questions   │
│  └── Evaluation/          → scoring scripts  │
└──────────────────────────────────────────────┘
```

### Retrieval Layer Detail

**Option A — Vanilla RAG (FAISS)**
```
PDF files → chunk.py → chunks.json → embedding.py → faiss.index + metadata.json
                                                   → top-k cosine retrieval
```

**Option B — GraphRAG (LightRAG)**
```
PDF files → LightRAG ingestion → knowledge graph (entities + relations)
                               → hybrid graph+vector retrieval (mix/hybrid/global/local modes)
```

LightRAG is directly supported by GraphRAG-Benchmark (`Examples/run_lightrag.py`), enabling consistent cross-method evaluation.

---

## Current Codebase State

### Implemented

- **chunk.py** — PDF extraction (`pypdf`), fixes OCR artifacts (spaced letters, page numbers), paragraph-based chunking (~500 chars). Output: `data/processed/chunks.json` with schema `{chunk_id, document_id, source_file, page, section, text}`.

- **embedding.py** — Encodes chunks with `sentence-transformers` (`all-MiniLM-L6-v2`), builds `IndexFlatIP` (L2-normalised vectors = cosine similarity). Outputs `index/faiss.index` + `index/metadata.json`.

- **graph_extract.py** — MVP entity/relation extraction using dictionary + pattern matching. Outputs `data/interim/entities.json` and `data/interim/relations.json`. **This is a prototype; LightRAG replaces it for the GraphRAG experimental condition**, but it can be kept as a lightweight alternative or for offline analysis.

- **faiss_test.py** — Standalone retrieval demo: loads index, encodes a query, prints top-5 results with scores.

- **prompt/clinical_assistant.txt** — System prompt template: cite with `[1][2]`, use only provided context, say "no info" if unsupported.

### Still Needed

| Component | Notes |
|---|---|
| Dynamiq orchestration wrappers | 4 modes: Single, Linear, Adaptive, Graph |
| LightRAG integration | Use `run_lightrag.py` from GraphRAG-Benchmark as reference |
| `src/pipeline.py` | Connects retriever → prompt → LLM → cited response |
| `src/llm.py` | `BaseLLM` abstraction; default `gpt-4o-mini` |
| GraphRAG-Benchmark setup | Clone repo, use medical dataset + evaluation scripts |
| `eval/run_experiments.py` | Loop over all 8 conditions, collect results |
| Streamlit UI (optional) | Demo interface |

---

## Running the Pipeline

```bash
# Install dependencies
pip install faiss-cpu sentence-transformers pypdf numpy dynamiq openai \
            langchain langchain_openai ragas rouge_score lightrag-hku

# --- Vanilla RAG pipeline ---

# Step 1: Chunk PDFs
python chunk.py
# Reads: data/raw/*.pdf → Outputs: data/processed/chunks.json

# Step 2: Build FAISS index
python embedding.py
# Reads: data/processed/chunks.json → Outputs: index/faiss.index, index/metadata.json

# Step 3: Test retrieval
python faiss_test.py

# --- GraphRAG pipeline (LightRAG) ---

# Use GraphRAG-Benchmark's run_lightrag.py as reference:
# python Examples/run_lightrag.py --subset medical --mode API --base_dir ./lightrag_workspace

# --- Evaluation (GraphRAG-Benchmark) ---

python Evaluation/generation_eval.py   # answer quality (ROUGE-L, correctness)
python Evaluation/retrieval_eval.py    # context relevancy, evidence recall
python Evaluation/indexing_eval.py     # KG structure (GraphRAG only)
```

---

## GraphRAG-Benchmark Integration

**Repo**: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark

The benchmark (ICLR'26) was built to answer exactly this project's research question. Key details:

- **Medical dataset** is included — directly applicable to WHO health guidelines
- **4 task types**: Fact Retrieval, Complex Reasoning, Contextual Summarization, Creative Generation
- **LightRAG already supported** via `Examples/run_lightrag.py` (use v1.2.5)
- Evaluation uses ROUGE-L + Answer Correctness + Coverage + Faithfulness depending on task type
- Requires `OPENAI_API_KEY` and embedding model (`BAAI/bge-large-en-v1.5` recommended)
- Use separate Conda environments per framework to avoid dependency conflicts

```bash
# Recommended setup for LightRAG condition
conda create -n lightrag python=3.10 -y
conda activate lightrag
pip install lightrag-hku
```

---

## Key Design Decisions

**graph_extract.py vs LightRAG**: `graph_extract.py` is a dictionary-based MVP. LightRAG uses LLM-extracted triples and a proper graph index. Use LightRAG for the GraphRAG experimental condition; keep `graph_extract.py` as a reference or lightweight offline tool.

**Dynamiq vs alternatives**: Dynamiq has built-in Single/Linear/Adaptive/Graph agent modes, making it the right choice for systematically testing orchestration. LangGraph is an alternative if Dynamiq proves limiting.

**FAISS stays**: `IndexFlatIP` with L2-normalised vectors = cosine similarity. Wrap behind `BaseRetriever` so it's swappable.

**LLM**: `gpt-4o-mini` as default. Abstract behind `BaseLLM`; swap via config.

**ARM / Dynamic Memory**: Skip — 4 static PDFs don't benefit. Mention as future work in report.

---

## Development Workflow

- **Person A**: Vanilla RAG pipeline (`src/chunker.py`, `src/embedder.py`, Dynamiq integration for Option A)
- **Person B**: GraphRAG pipeline (LightRAG integration, `graph_extract.py` reference, Dynamiq integration for Option B)
- Both: evaluation harness, experiment runner, report

Branch naming:
```
main
├── feat/dynamiq-orchestration
├── feat/vanilla-rag
├── feat/lightrag-integration
├── feat/evaluation-harness
└── feat/ui  (stretch)
```

- Keep notebooks for exploration; migrate stable code to `src/`
- Use `.env` for API keys — never commit
- PR reviews required even in a 2-person team (catches prompt regressions)

---

## Key Papers

- **Lewis et al. (2020)** — Foundational RAG. https://arxiv.org/abs/2005.11401
- **Gao et al. (2023)** — RAG survey: Naive → Advanced → Modular. https://arxiv.org/abs/2312.10997
- **Li et al. (2025)** — Contrastive ICL RAG: query expansion, top-k ablation, contrastive prompting.
- **Edge et al. (2024)** — Microsoft GraphRAG. https://arxiv.org/abs/2404.16130
- **Xiang et al. (2026)** — GraphRAG-Benchmark (ICLR'26): when to use graphs in RAG. https://arxiv.org/abs/2506.05690
- **Es et al. (2023)** — RAGAS evaluation framework. https://arxiv.org/abs/2309.15217
- **Zakka et al. (2024)** — Almanac: RAG for clinical medicine. https://arxiv.org/abs/2303.01229
