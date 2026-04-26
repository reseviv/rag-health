# RAG Health — New Pipeline Design & Construction Plan

## Overview

The new pipeline replaces the fusion-score (α·graph + β·vector) approach with a graph-enriched
query strategy: retrieve chunks via FAISS, collect entities from both the query and retrieved
chunks, look up their graph neighbors, then pass chunk texts + entity hints together to the LLM.

---

## New Pipeline Flow

```
Original Query
    │
    ├─► [MedCPT-Query-Encoder] ──► FAISS search ──► top-K chunks
    │                                                      │
    │                                           chunk_to_entities lookup
    │                                                      │
    ├─► [spaCy en_core_sci_lg NER] ──► query entities     │
    │                                                      │
    └─────────────────────────────────────────────────────►│
                                                     Union + deduplicate
                                                     entity pool
                                                           │
                                              Graph lookup (top-N neighbors
                                              by edge weight per entity)
                                                           │
                                         Global deduplication of entity-neighbor pairs
                                                           │
                              ┌────────────────────────────┴──────────────────────────┐
                              │ context: chunk texts [1][2]...                        │
                              │ query:   original query +                             │
                              │          "think with: {entity: [neighbors]} (JSON)"   │
                              └────────────────────────────┬──────────────────────────┘
                                                           │
                                                    [LLM] generate
                                                           │
                                                     cited answer
```

---

## Component Changes

### Phase 1 — Rebuild Index (offline, run once)

#### 1A. `graph_build.py` — two changes

**Change 1: Fix NER model mismatch**
```python
# current (wrong)
nlp = spacy.load("en_ner_bc5cdr_md")   # diseases + chemicals only

# fix
nlp = spacy.load("en_core_sci_lg")     # broader biomedical, matches query-time NER
```

**Change 2: Replace co_occurs_with with typed relations via spaCy dep parser**

Instead of "two entities in the same chunk → co_occurs_with", parse dependency tree to extract
subject–verb–object triples between detected entity pairs:

```
"Antiretroviral therapy is used to treat HIV"
→ (antiretroviral therapy, treats, HIV)

"CD4 count is used to monitor disease progression"
→ (CD4 count, monitors, disease progression)
```

Edge weight = frequency of relation across all chunks (more occurrences = stronger edge).
Fallback: if dep parser finds no typed relation between two co-occurring entities, keep
co_occurs_with as a weak-confidence edge (so graph stays connected).

#### 1B. `embedding.py` — switch to MedCPT asymmetric encoders

```python
# current (symmetric)
model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
embeddings = model.encode(chunk_texts)

# new (asymmetric — separate encoder for documents)
from transformers import AutoTokenizer, AutoModel
article_encoder = AutoModel.from_pretrained("ncats/MedCPT-Article-Encoder")
# encode all chunks with article encoder → save to faiss.index
```

Outputs: rebuilt `index/faiss.index`, `index/metadata.json`, `index/graph.pkl`

---

### Phase 2 — New Retrieval Layer

**File: `graph_retrieve.py`** — major rewrite

#### Responsibilities

1. Encode query with MedCPT-Query-Encoder
2. FAISS search → top-K chunks
3. Collect entities:
   - Run `en_core_sci_lg` NER on original query
   - Look up `chunk_to_entities[chunk_id]` for each retrieved chunk
   - Union → deduplicate → entity pool
   - Optional filter: keep only entities appearing in ≥ 2 chunks (reduces noise)
4. Graph neighbor lookup:
   - For each entity in pool: get neighbors from graph
   - Rank by edge weight, take top-N per entity (N TBD, start with 5)
5. Global deduplication: collect unique (entity, neighbor) pairs as a set
6. Return: `{ chunks: [...], entity_neighbors: { entity: [neighbor, ...] } }`

#### Removed
- `alpha`, `beta`, `fused_score` — no longer used
- Personalized PageRank — replaced by direct neighbor lookup
- `_graph_scores`, `_vector_scores` fusion logic

#### Open Decisions
- Entity frequency threshold: all entities, or ≥ 2 chunks only?
- Neighbor cap N per entity (start with 5, tune later)

---

### Phase 3 — New LLM Client

**File: new file, not `llm.py`** (teammate's file — don't overwrite)

#### Methods needed

```python
def generate(query: str, chunks: list[dict], entity_hints: dict) -> str:
    # builds prompt: chunk context + enriched query
    # entity_hints format: { "hiv": ["antiretroviral therapy", "cd4 count"], ... }

def _format_context(chunks: list[dict]) -> str:
    # same as current: "[1] text (Source: file, page)"

def _format_entity_hints(entity_hints: dict) -> str:
    # produces JSON-style string:
    # "think about the question with these relevant concepts: 
    #  {"hiv": ["antiretroviral therapy", "cd4 count"], ...}"

def _call(messages, temperature=0.1, timeout=60) -> str:
    # adds timeout + simple retry (max 2 attempts)
```

#### New System Prompt

```
You are a clinical assistant answering questions based on WHO health guidelines.

Rules:
- Base your answer ONLY on the numbered source passages [1], [2], etc.
- Cite every claim with [1], [2], etc.
- The "relevant concepts" section provides reasoning hints about related medical
  concepts — use them to inform your thinking but do not treat them as source citations.
- If the answer is not in the source passages, say exactly:
  "This information is not available in the provided guidelines."
- Be concise and factual.
```

#### Fixes vs current `llm.py`
- `generate()` accepts `entity_hints` as third parameter
- Context length guard: warn if estimated tokens > model limit (8k for 7B models)
- Timeout + retry on `_call`
- Temperature as parameter with 0.1 default
- `rewrite_query` kept but optional (still used in linear/adaptive modes if kept)

---

### Phase 4 — Update Experiment Runner

**File: `pipeline.py`**

#### Agent modes

| Mode | Status | Notes |
|---|---|---|
| single | keep, update | retrieve → generate with entity hints |
| linear | keep, update | rewrite → retrieve → generate with hints → verify |
| adaptive | needs rework | lost quality signal (no fused score); needs new signal |
| graph | keep, update | decompose → retrieve each → collect entities → generate with hints |

**Adaptive mode replacement signal (options):**
- Count of matched entities in graph (more matches = better retrieval)
- Number of unique entity-neighbor pairs retrieved (richer context = retry not needed)
- Decision: TBD

---

## Build Order

```
1. Fix graph_build.py  (NER + dep parser typed relations)
2. Fix embedding.py    (MedCPT encoders)
3. Rebuild index       (graph.pkl, faiss.index, metadata.json)
4. Rewrite graph_retrieve.py
5. Write new LLM client file
6. Update pipeline.py
7. Build QA dataset    (still needed for evaluation — no ground truth yet)
8. Run experiments + eval
```

---

## Open Decisions (unresolved)

| # | Decision | Options | Status |
|---|---|---|---|
| 1 | Entity frequency threshold | ≥ 2 chunks | **agreed** |
| 2 | Neighbor cap N | start with 5 | agreed, refine later |
| 3 | Adaptive mode quality signal | entity match count / hint richness | undecided |
| 4 | New LLM client filename | `llm_v2.py` | **agreed** |
| 5 | QA dataset generation | generate_qa.py with LLM | not built yet |

---

## Deferred Improvements (don't build now)

- **REBEL** — HuggingFace `Babelscape/rebel-large` for typed relation extraction.
  Upgrade path if spaCy dep parser produces poor-quality relations in eval.
- **Neighbor ranking upgrade** — weight by entity type relevance + cross-chunk frequency,
  instead of edge weight alone.
