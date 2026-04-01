"""
graph_retrieve.py — Query phase

Combines:
  - SpaCy NER on query  →  Personalized PageRank on NetworkX graph  →  graph score per chunk
  - SentenceTransformer embedding  →  FAISS cosine search  →  vector score per chunk
  - Score fusion: alpha * graph_score + beta * vector_score

Usage:
    python graph_retrieve.py
    or import GraphRetriever and call .retrieve(query)
"""

import json
import logging
import pickle
from pathlib import Path

import faiss
import networkx as nx
import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

GRAPH_FILE = Path("index/graph.pkl")
FAISS_FILE = Path("index/faiss.index")
METADATA_FILE = Path("index/metadata.json")
MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"  # biomedical model trained on PubMed + MS MARCO retrieval task


class GraphRetriever:
    def __init__(self, alpha: float = 0.4, beta: float = 0.6, top_k: int = 5):
        """
        alpha  — weight for graph (PageRank) score
        beta   — weight for vector (FAISS cosine) score
        alpha + beta should equal 1.0
        """
        self.alpha = alpha
        self.beta = beta
        self.top_k = top_k

        print("Loading scispaCy general science NER model (en_core_sci_lg)...")
        self.nlp = spacy.load("en_core_sci_lg")

        print("Loading graph...")
        with open(GRAPH_FILE, "rb") as f:
            data = pickle.load(f)
        self.G: nx.Graph = data["graph"]
        self.entity_to_chunks: dict = data["entity_to_chunks"]
        self.chunk_to_entities: dict = data["chunk_to_entities"]

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_FILE))

        print("Loading metadata...")
        with open(METADATA_FILE, encoding="utf-8") as f:
            metadata_list = json.load(f)
        # Fast lookup: chunk_id -> metadata dict
        self.meta_by_chunk: dict = {m["chunk_id"]: m for m in metadata_list}
        # Ordered list indexed by faiss_id (same order as index)
        self.metadata: list = metadata_list

        print(f"Loading embedding model ({MODEL_NAME})...")
        self.model = SentenceTransformer(MODEL_NAME)

        print("GraphRetriever ready.\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_query_entities(self, query: str) -> list[str]:
        doc = self.nlp(query)
        return [ent.text.lower().strip() for ent in doc.ents if len(ent.text.strip()) > 2]

    def _vector_scores(self, query: str) -> dict[str, float]:
        """FAISS cosine similarity for every chunk. Returns chunk_id -> score in [0, 1]."""
        vec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        # Search all chunks so we can fuse across the full set
        scores, indices = self.index.search(vec, len(self.metadata))

        raw: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self.metadata[idx]["chunk_id"]
            raw[chunk_id] = float(score)

        # Cosine scores from IndexFlatIP on normalised vecs are already in [-1, 1].
        # Shift and scale to [0, 1] so they're comparable with graph scores.
        if not raw:
            return raw
        min_s, max_s = min(raw.values()), max(raw.values())
        rng = max_s - min_s or 1.0
        return {k: (v - min_s) / rng for k, v in raw.items()}

    def _graph_scores(self, query_entities: list[str], all_chunk_ids: list[str]) -> tuple[dict[str, float], str]:
        """Personalized PageRank → per-chunk importance. Returns (chunk_id -> score in [0, 1], retrieval_mode)."""
        zero = {cid: 0.0 for cid in all_chunk_ids}

        if not query_entities:
            logger.warning("GRAPH FALLBACK: NER found no entities in query — using vector-only retrieval.")
            return zero, "vector_only:no_entities"

        if self.G.number_of_nodes() == 0:
            logger.warning("GRAPH FALLBACK: graph has no nodes — using vector-only retrieval.")
            return zero, "vector_only:empty_graph"

        # Match query entities to graph nodes: exact first, then substring fallback
        personalization: dict[str, float] = {}
        for e in query_entities:
            if e in self.G:
                personalization[e] = 1.0
            else:
                # expand to all graph nodes that contain this entity as a substring
                substring_matches = [n for n in self.G.nodes if e in n]
                for n in substring_matches:
                    personalization[n] = 1.0

        if not personalization:
            logger.warning(
                "GRAPH FALLBACK: entities %s found by NER but none matched graph nodes — "
                "using vector-only retrieval.",
                query_entities,
            )
            return zero, "vector_only:no_graph_match"

        matched = list(personalization.keys())
        unmatched = [e for e in query_entities if e not in self.G and not any(e in n for n in self.G.nodes)]
        if unmatched:
            logger.info("GRAPH PARTIAL: entities %s matched graph; %s did not.", matched, unmatched)

        pr = nx.pagerank(self.G, personalization=personalization, weight="weight")

        # Accumulate entity PageRank scores into their associated chunks
        chunk_scores: dict[str, float] = {cid: 0.0 for cid in all_chunk_ids}
        for entity, score in pr.items():
            for chunk_id in self.entity_to_chunks.get(entity, []):
                if chunk_id in chunk_scores:
                    chunk_scores[chunk_id] += score

        # Normalize to [0, 1]
        max_score = max(chunk_scores.values()) or 1.0
        return {k: v / max_score for k, v in chunk_scores.items()}, "graph+vector"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict]:
        """
        Returns top-K chunks ranked by fused score.
        Each result dict contains: chunk_id, text, source_file, page,
        graph_score, vector_score, fused_score, query_entities.
        """
        # Step 1 — vector scores (FAISS)
        v_scores = self._vector_scores(query)
        all_chunk_ids = list(v_scores.keys())

        # Step 2 — graph scores (SpaCy NER + PageRank)
        query_entities = self._extract_query_entities(query)
        g_scores, retrieval_mode = self._graph_scores(query_entities, all_chunk_ids)

        # Step 3 — fuse
        fused = {
            cid: self.alpha * g_scores.get(cid, 0.0) + self.beta * v_scores.get(cid, 0.0)
            for cid in all_chunk_ids
        }

        # Step 4 — top-K
        top_ids = sorted(fused, key=fused.__getitem__, reverse=True)[: self.top_k]

        results = []
        for cid in top_ids:
            meta = self.meta_by_chunk.get(cid, {})
            results.append({
                "chunk_id": cid,
                "text": meta.get("text", ""),
                "source_file": meta.get("source_file", ""),
                "page": meta.get("page", ""),
                "graph_score": round(g_scores.get(cid, 0.0), 4),
                "vector_score": round(v_scores.get(cid, 0.0), 4),
                "fused_score": round(fused[cid], 4),
                "query_entities": query_entities,
                "retrieval_mode": retrieval_mode,
            })

        return results


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    retriever = GraphRetriever(alpha=0.4, beta=0.6, top_k=5)

    queries = [
        "What are the WHO guidelines for HIV treatment?",
        "How is infertility diagnosed?",
        "What are the risks of abortion care?",
    ]

    for query in queries:
        print(f"Query : {query}")
        results = retriever.retrieve(query)
        print(f"Entities detected: {results[0]['query_entities'] if results else []}")
        print()
        for i, r in enumerate(results, 1):
            print(f"  [{i}] {r['source_file']}  p.{r['page']}"
                  f"  graph={r['graph_score']}  vec={r['vector_score']}  fused={r['fused_score']}")
            print(f"       {r['text'][:120]}...")
        print("-" * 70)
