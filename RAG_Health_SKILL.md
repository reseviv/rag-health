---
name: rag-health
description: >
  RAG system development guide for a two-person team building a health-domain
  Retrieval-Augmented Generation chatbot. Covers the full pipeline: PDF
  ingestion → chunking → FAISS embedding → retrieval → LLM generation → eval.
  Use this skill when working on rag-health, discussing chunking strategies,
  vector DB choices, hybrid retrieval, GraphRAG, prompt engineering for clinical
  guidelines, evaluation metrics, MCP integration, or any task related to this
  project's codebase (github.com/reseviv/rag-health).
---

# RAG-Health Project Skill

## Project Overview

**Goal**: Build a RAG-powered chatbot that answers questions grounded in
health-domain PDFs (HIV, Infertility, Abortion Care guidelines).

**Team size**: 2 people
**Repo**: https://github.com/reseviv/rag-health
**Stack (current)**: Python · FAISS · Jupyter notebooks
**Stack (target)**: FastAPI or Streamlit backend · React or Streamlit UI

---

## Project Layout

```
rag-health/
  data/
    raw/          # Source PDFs (HIVService, AdvancedHIV, Infertility, AbortionCare)
    processed/    # chunks.json output from chunk.ipynb
  index/
    faiss.index   # FAISS flat index
    metadata.json # chunk <-> source/page mapping
  src/            # core library (to be built)
    chunker.py    # PDF -> chunks
    embedder.py   # chunks -> FAISS index
    retriever.py  # query -> top-k + hybrid BM25 fusion
    pipeline.py   # retriever -> LLM -> cited response
    llm.py        # BaseLLM abstraction (multi-model)
    mcp_server.py # (stretch) MCP tool exposure
    graph.py      # (stretch) NetworkX KG for GraphRAG
  app/
    main.py       # Streamlit or FastAPI entry point
  eval/
    ragas_eval.py # RAGAS evaluation harness
    test_set.json # hand-labelled Q&A pairs (~50)
  chunk.ipynb     # PDF -> chunks (prototype)
  embedding.ipynb # chunks -> FAISS index (prototype)
  retrieve.ipynb  # query -> top-k retrieval demo
  prompt.txt      # system prompt template (citation-aware, "education only")
```

---

## TODO Tracker

### Research (w1-w2)
- [x] RAG system architecture decision
- [x] Text preprocessing strategy
- [ ] Quantitative & qualitative evaluation metrics
      -> RAGAS · Recall@k · MRR · citation accuracy · graceful failure rate
- [ ] Token API & open-source LLM selection
      -> gpt-4o-mini (primary) · gemini-flash (fallback) · Ollama llama-3 (local)
- [ ] Prompt optimisation
      -> few-shot examples · chain-of-thought · faithfulness gate · HyDE query rewriting
- [ ] Vector DB option finalised
      -> FAISS now -> Qdrant if metadata filtering needed · hybrid BM25 wrapper

### RAG Pipeline (w3-6)
- [ ] Preprocessing — src/chunker.py [CORE]
      -> RecursiveCharacterTextSplitter · 400 tok · 50 overlap · store source + page
- [ ] Embedding — src/embedder.py [CORE]
      -> sentence-transformers or text-embedding-3-small · build FAISS index
- [ ] Retrieval — src/retriever.py [CORE]
      -> top-k search · MMR re-ranking · abstract behind BaseRetriever class
- [ ] Hybrid retrieval — BM25 + dense fusion [CORE]
      -> rank_bm25 + RRF score fusion · key for clinical acronyms & exact terms
- [ ] Top-k ablation study [CORE]
      -> test k=2,4,6,8 against eval set · pick optimal k per query type
- [ ] Query expansion / rewriting [CORE]
      -> LLM rewrites query -> 2-3 variants -> merge results (Li et al. 2025)
- [ ] GraphRAG — lightweight KG on 4 PDFs [STRETCH]
      -> LLM extracts triples · NetworkX graph · 1-2 hop traversal at query time

### LLM Chatbot (w3-6)
- [ ] API integration — multi-model [CORE]
      -> BaseLLM abstraction · swap model via config · gpt-4o-mini default
- [ ] RAG <-> LLM connection — src/pipeline.py [CORE]
      -> retriever -> prompt template -> LLM -> cited response
- [ ] Prompt engineering — prompt.txt v2 [CORE]
      -> few-shot citations · faithfulness gate · contrastive ICL (Li et al. 2025)
- [ ] UI — Streamlit MVP [CORE]
      -> chat input · cited sources panel · graceful "not in docs" response
- [ ] MCP server — src/mcp_server.py [STRETCH]
      -> expose search_guidelines() + get_page() as MCP tools
- [ ] ARM / dynamic memory [SKIP — future work only]
      -> overkill for 4 static PDFs · mention in report as future direction

### Evaluation / Testing / Finalising (w7-8)
- [ ] Retrieval quality
      -> Recall@k · MRR · NDCG · context precision & recall
- [ ] Generation quality
      -> faithfulness · hallucination rate · citation accuracy (page-level check)
- [ ] End-to-end answer quality
      -> RAGAS framework · answer relevancy · LLM-as-judge scoring
- [ ] Domain-specific eval
      -> out-of-scope refusal rate · guideline grounding · cross-doc multi-hop Qs
- [ ] QA testing
      -> adversarial queries · edge cases · questions not in any PDF
- [ ] Report writing
      -> methodology · ablation results · future work: ARM, full GraphRAG, MCP

---

## Pipeline Architecture

```
PDF files
   |
   v
[Chunker]      -- RecursiveCharacterTextSplitter, ~400 tok, 50 overlap
   |               store: {id, text, source, page, section}
   v
[Embedder]     -- sentence-transformers or text-embedding-3-small
   |               store in FAISS (flat L2)
   |-------------------------------------.
   v                                     v
[Dense retriever]                  [BM25 retriever]
   |    top-k cosine similarity          |    keyword match (rank_bm25)
   '--------------.---------------------'
                  v
            [RRF fusion]           -- Reciprocal Rank Fusion
                  |
                  v (optional)
           [Query rewriting]       -- LLM rewrites query -> 2-3 variants
                  |
                  v (optional stretch)
           [Graph traversal]       -- NetworkX KG, 1-2 hop entity lookup
                  |
                  v
              [LLM]                -- inject chunks into prompt.txt v2
                  |                   model: gpt-4o-mini / gemini-flash / llama-3
                  v
           [Response]              -- citation-grounded answer [1][2] with page numbers
```

---

## Research Directions & Scope Decisions

Three active research areas, assessed for feasibility in a 2-person prototype:

### 1. Best Practices / Contrastive ICL RAG — DO THIS (w3-4)
Source: Li et al. (2025) — Contrastive In-Context Learning RAG

Key techniques to implement:
- Query expansion: rewrite query into 2-3 clinical synonyms, retrieve for each,
  merge results with RRF
- Top-k ablation: systematically test k=2,4,6,8 — don't guess
- Contrastive prompting: show model a good and bad answer as few-shot examples
  so it learns citation-grounded vs hallucinated behaviour

This is 1-2 weeks of work and directly strengthens both system and report.

### 2. GraphRAG / KG Robustness — STRETCH GOAL (w5-6)
Source: "Towards Robust RAG Based on Knowledge Graph" (March 2026)

GraphRAG significantly improves robustness against noise, negative rejection,
and counterfactuals — directly relevant for clinical guidelines with
contradictory or overlapping content across 4 PDFs.

Scoped implementation (no Neo4j needed at this scale):

```python
import networkx as nx

# Step 1: LLM extracts triples from each chunk
triples = llm.extract("List (entity, relation, entity) triples: ...")

# Step 2: build in-memory graph
G = nx.DiGraph()
for head, rel, tail in triples:
    G.add_edge(head, tail, label=rel)

# Step 3: at query time, walk 1-2 hops from matched entity nodes
def graph_retrieve(query_entities):
    nodes = set()
    for entity in query_entities:
        if entity in G:
            nodes.update(nx.ego_graph(G, entity, radius=2).nodes())
    return nodes
```

Use LightRAG if you want a pre-built alternative.

### 3. ARM / Dynamic Memory — SKIP (mention in future work)
Source: "A Dynamic RAG System with Selective Memory and Remembrance" (Jan 2026)

ARM uses selective remembrance and decay for growing knowledge bases. This
project has 4 static PDFs that do not change. The motivation for ARM does not
apply. Include as a "future work" paragraph in the report.

---

## Vector DB Options

| Option | When to use |
|---|---|
| FAISS (current) | Prototyping, <= 100k chunks, no persistence needed |
| ChromaDB | Easy local persistent store, good for dev |
| Qdrant | If you need filtering by metadata (filter by PDF source) |
| Weaviate | Hybrid BM25 + dense search built-in |
| NetworkX | Stretch: in-memory graph for lightweight GraphRAG |

Recommendation: stay on FAISS, wrap behind BaseRetriever so swapping is
one line. Add BM25 via rank_bm25 alongside FAISS; fuse with RRF.

Vector DB generation landscape for context:
- Gen 1 (current): dense-only FAISS — cosine similarity, no persistence
- Gen 2 (target): hybrid BM25 + dense + RRF fusion — best for clinical terms
- Gen 3 (stretch): GraphRAG with NetworkX — multi-hop cross-doc reasoning
- Gen 4 (future): tensor/multimodal, agentic RAG, ARM dynamic memory

---

## LLM Options

| Model | Pros | Cons |
|---|---|---|
| gpt-4o-mini | Cheap, fast, good citations | API cost |
| gemini-1.5-flash | Very long context window | Less citation-following |
| mistral-7b (local) | Free, private | Needs GPU or slower |
| llama-3-8b (Ollama) | Free local, easy Mac setup | Quality varies |

Multi-model pattern: abstract behind a BaseLLM class; pass model name
as config. Lets you swap without rewriting RAG logic.

---

## Chunking Strategy

Migrate chunk.ipynb -> src/chunker.py:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,       # tokens -- tune per PDF density
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)
```

Store per chunk:
```json
{
  "id": "uuid",
  "text": "...",
  "source": "HIVService.pdf",
  "page": 12,
  "section": "Treatment"
}
```

---

## Prompting

Current prompt.txt is a solid baseline. Version 2 improvements:

1. Few-shot examples — add 2-3 Q&A pairs showing correct vs wrong citation style
2. Chain-of-thought — ask model to reason before answering
3. Faithfulness gate — "If not fully supported by context, say so explicitly"
4. Query rewriting (HyDE) — LLM generates hypothetical answer first, embed
   that to retrieve real chunks (closes query-document semantic gap)
5. Contrastive ICL — show one good answer and one bad answer as contrast
   (Li et al. 2025 key technique)

---

## Evaluation Framework

### RAGAS (primary)
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,   # HuggingFace Dataset: question/answer/contexts/ground_truth
    metrics=[faithfulness, answer_relevancy, context_recall]
)
```

### Full metric matrix

| Layer | Metric | Tool |
|---|---|---|
| Retrieval | Recall@k, Precision@k | manual / RAGAS |
| Retrieval | MRR, NDCG | manual |
| Retrieval | Context precision, Context recall | RAGAS |
| Generation | Faithfulness | RAGAS |
| Generation | Hallucination rate | LLM-as-judge |
| Generation | Citation accuracy | deterministic page check |
| End-to-end | Answer relevancy | RAGAS |
| End-to-end | Answer correctness | RAGAS |
| Domain | Out-of-scope refusal rate | manual test set |
| Domain | Guideline grounding | LLM-as-judge |
| Domain | Cross-doc multi-hop accuracy | manual test set |
| Operational | Latency (target < 3s E2E) | time.perf_counter |
| Operational | Cost per query | token count x price |

### Test set construction (~50 questions)
- Simple single-PDF factual questions (easy baseline)
- Cross-document questions requiring 2+ PDFs (GraphRAG payoff)
- Out-of-scope questions not answerable from PDFs (graceful failure)
- Adversarial / ambiguous queries (robustness)

---

## MCP Integration (Stretch)

Expose the RAG pipeline as MCP tools so any MCP-compatible LLM client
can use your retriever without knowing its internals:

```python
# src/mcp_server.py
from mcp.server import Server

server = Server("rag-health")

@server.tool()
async def search_guidelines(query: str, top_k: int = 5) -> str:
    """Search clinical guidelines and return relevant chunks with citations."""
    chunks = retriever.search(query, k=top_k)
    return format_chunks_with_citations(chunks)

@server.tool()
async def get_page(filename: str, page: int) -> str:
    """Return raw text from a specific PDF page."""
    return extract_page(filename, page)
```

MCP vs RAG distinction:
- RAG is a retrieval technique (embed -> retrieve -> inject)
- MCP is a communication protocol (standardised tool interface for LLMs)
- You build RAG inside an MCP tool — they are complementary, not competing

---

## Development Workflow (2-person team)

```
main --- research/eval-metrics
     |-- feat/chunker
     |-- feat/embedder
     |-- feat/retriever-hybrid
     |-- feat/query-expansion
     |-- feat/llm-integration
     |-- feat/ui
     |-- feat/graphrag        (stretch)
     '-- feat/mcp-server      (stretch)
```

- PR reviews required even in 2-person team (catches prompt regressions)
- Keep notebooks for exploration; migrate stable code to src/
- Use .env for API keys — never commit
- Abstract retriever and LLM behind base classes from day 1

---

## Key Implementation Files

| File | Owner | Priority | Notes |
|---|---|---|---|
| src/chunker.py | Person A | CORE | migrate from chunk.ipynb |
| src/embedder.py | Person A | CORE | migrate from embedding.ipynb |
| src/retriever.py | Person B | CORE | dense + BM25 + RRF fusion |
| src/llm.py | Person B | CORE | BaseLLM multi-model abstraction |
| src/pipeline.py | Person B | CORE | retriever -> LLM -> response |
| app/main.py | Person A | CORE | Streamlit UI entry point |
| eval/ragas_eval.py | Both | CORE | evaluation harness |
| eval/test_set.json | Both | CORE | ~50 hand-labelled Q&A pairs |
| src/graph.py | Person A | STRETCH | NetworkX GraphRAG |
| src/mcp_server.py | Person B | STRETCH | MCP tool server |

---

## Quick Commands

```bash
# Setup
pip install faiss-cpu sentence-transformers langchain openai ragas streamlit \
            rank_bm25 networkx mcp

# Run notebooks in order (prototype phase)
jupyter nbconvert --to notebook --execute chunk.ipynb
jupyter nbconvert --to notebook --execute embedding.ipynb
jupyter nbconvert --to notebook --execute retrieve.ipynb

# Convert notebooks -> scripts (implementation phase)
jupyter nbconvert --to script chunk.ipynb --output src/chunker
jupyter nbconvert --to script embedding.ipynb --output src/embedder
jupyter nbconvert --to script retrieve.ipynb --output src/retriever

# Run evaluation
python eval/ragas_eval.py --test-set eval/test_set.json --k 4

# Start UI
streamlit run app/main.py
```

---

## Paper References

### Core RAG
1. Lewis et al. (2020) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
   NeurIPS 2020. Foundational RAG paper. DPR retriever + Seq2Seq generator.
   https://arxiv.org/abs/2005.11401

2. Gao et al. (2023) — RAG for LLMs: A Survey
   Taxonomy: Naive -> Advanced -> Modular RAG.
   https://arxiv.org/abs/2312.10997

### Best Practices (implement this — w3-4)
3. Li et al. (2025) — Contrastive In-Context Learning RAG
   Query expansion, top-k ablation, contrastive prompting. Actionable
   best practices for contextual richness vs generation efficiency balance.

### Advanced Retrieval
4. Ma et al. (2023) — Query Rewriting for RAG LLMs
   Smaller LLM rewrites query before retrieval — easy win for health domain.
   https://arxiv.org/abs/2305.14283

5. Asai et al. (2023) — Self-RAG
   Model decides when to retrieve; critiques its own output.
   https://arxiv.org/abs/2310.11511

6. Chen et al. (2023) — Dense X Retrieval: What Retrieval Granularity?
   Proposition-level chunking (atomic fact units) vs sentence/passage.
   https://arxiv.org/abs/2312.06648

### GraphRAG (stretch goal — w5-6)
7. Edge et al. (2024) — From Local to Global: A GraphRAG Approach
   Microsoft's GraphRAG. Community detection with Leiden algorithm.
   https://arxiv.org/abs/2404.16130

8. Anon (March 2026) — Towards Robust RAG Based on Knowledge Graph
   GraphRAG improves robustness against noise, negative rejection,
   counterfactuals. Directly validates the stretch goal for this project.

### Dynamic Memory (future work only — do not implement)
9. Anon (Jan 2026) — A Dynamic RAG System with Selective Memory (ARM)
   Adaptive memory with decay for growing corpora. Not applicable to
   4 static PDFs — cite as future work in report.

### Evaluation
10. Es et al. (2023) — RAGAS: Automated Evaluation of RAG
    Reference-free evaluation using LLM-as-judge.
    https://arxiv.org/abs/2309.15217

11. Saad-Falcon et al. (2023) — ARES: Automated Evaluation Framework for RAG
    Alternative to RAGAS; lightweight judge models.
    https://arxiv.org/abs/2311.09476

### Health / Clinical NLP
12. Singhal et al. (2023) — Large Language Models Encode Clinical Knowledge (Med-PaLM)
    Foundational LLMs-for-medical-QA work; frames your eval design.
    https://arxiv.org/abs/2212.13138

13. Zakka et al. (2024) — Almanac: RAG for Clinical Medicine
    RAG applied specifically to clinical guidelines — most directly relevant.
    https://arxiv.org/abs/2303.01229
