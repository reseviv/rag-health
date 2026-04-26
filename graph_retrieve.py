"""
graph_retrieve.py — Query phase

Changes from v1:
  - Query encoder: ncats/MedCPT-Query-Encoder (asymmetric pair with Article-Encoder)
  - No fusion score: alpha/beta/fused_score removed
  - Entity collection: query NER + chunk_to_entities from retrieved chunks
  - Frequency filter: entity must appear in >= 2 retrieved chunks (query NER entities bypass)
  - Entity cap: top 10 entities by retrieved-chunk frequency
  - Graph neighbors: top 5 per entity by edge weight
  - Returns: chunks list + entity_neighbors dict

Usage:
    python graph_retrieve.py
    or import GraphRetriever and call .retrieve(query)
"""

import json
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import spacy
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(1)

logger = logging.getLogger(__name__)

GRAPH_FILE = Path("index/graph.pkl")
FAISS_FILE = Path("index/faiss.index")
METADATA_FILE = Path("index/metadata.json")
QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
NER_MODEL = "en_core_sci_lg"

ENTITY_MIN_CHUNK_FREQ = 2   # entity must appear in >= this many retrieved chunks
ENTITY_MAX_COUNT = 10       # max entities to send to graph after filtering
NEIGHBOR_TOP_N = 5          # top N neighbors per entity by edge weight


class GraphRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

        print(f"Loading NER model ({NER_MODEL})...")
        self.nlp = spacy.load(NER_MODEL)

        print("Loading graph...")
        with open(GRAPH_FILE, "rb") as f:
            data = pickle.load(f)
        self.G = data["graph"]
        self.entity_to_chunks: dict = data["entity_to_chunks"]
        self.chunk_to_entities: dict = data["chunk_to_entities"]

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(FAISS_FILE))

        print("Loading metadata...")
        with open(METADATA_FILE, encoding="utf-8") as f:
            metadata_list = json.load(f)
        self.meta_by_chunk: dict = {m["chunk_id"]: m for m in metadata_list}
        self.metadata: list = metadata_list

        print(f"Loading query encoder ({QUERY_MODEL})...")
        self.query_tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
        self.query_model = AutoModel.from_pretrained(QUERY_MODEL)
        self.query_model.eval()

        print("GraphRetriever ready.\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_query(self, query: str) -> np.ndarray:
        inputs = self.query_tokenizer(
            [query],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = self.query_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.numpy().astype("float32")

    def _extract_query_entities(self, query: str) -> list[str]:
        doc = self.nlp(query)
        return [ent.text.lower().strip() for ent in doc.ents if len(ent.text.strip()) > 2]

    def _collect_entities(
        self, query_entities: list[str], chunks: list[dict]
    ) -> list[str]:
        """
        Combine query NER entities with entities from retrieved chunks.
        Chunk entities are filtered to those appearing in >= ENTITY_MIN_CHUNK_FREQ chunks.
        Query entities bypass the frequency filter.
        Returns up to ENTITY_MAX_COUNT entities total.
        """
        # Count entity frequency across retrieved chunks
        freq: dict[str, int] = {}
        for chunk in chunks:
            cid = chunk["chunk_id"]
            for ent in self.chunk_to_entities.get(cid, []):
                freq[ent] = freq.get(ent, 0) + 1

        # Chunk entities that pass frequency filter
        chunk_entities = {e for e, f in freq.items() if f >= ENTITY_MIN_CHUNK_FREQ}

        # Query entities always included
        query_set = set(query_entities)

        # Combine: query entities first (priority), then chunk entities by freq
        combined = list(query_set)
        remaining = sorted(
            chunk_entities - query_set,
            key=lambda e: freq.get(e, 0),
            reverse=True,
        )
        combined.extend(remaining)

        return combined[:ENTITY_MAX_COUNT]

    def _get_neighbors(self, entity: str) -> list[tuple[str, str, float]]:
        """
        Return top-N neighbors of entity sorted by edge weight.
        Each result: (neighbor_name, predicate, weight)
        Only returns neighbors that exist in graph.
        """
        if entity not in self.G:
            return []

        neighbors = []
        for neighbor in self.G.neighbors(entity):
            edge = self.G[entity][neighbor]
            neighbors.append((
                neighbor,
                edge.get("predicate", "co_occurs_with"),
                edge.get("weight", 1),
            ))

        neighbors.sort(key=lambda x: x[2], reverse=True)
        return neighbors[:NEIGHBOR_TOP_N]

    def _build_entity_neighbors(
        self, entities: list[str]
    ) -> dict[str, list[tuple[str, str]]]:
        """
        For each entity, get top-N neighbors.
        Returns {entity: [(neighbor, predicate), ...]}
        Globally deduplicates (entity, neighbor) pairs to avoid redundancy.
        """
        seen_pairs: set[frozenset] = set()
        result: dict[str, list[tuple[str, str]]] = {}

        for entity in entities:
            neighbors = self._get_neighbors(entity)
            unique = []
            for neighbor, predicate, _ in neighbors:
                pair = frozenset({entity, neighbor})
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    unique.append((neighbor, predicate))
            if unique:
                result[entity] = unique

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> dict:
        """
        Returns:
          {
            "chunks":          list of top-K chunk dicts (text, source, page, vector_score),
            "entity_neighbors": {entity: [(neighbor, predicate), ...]},
            "query_entities":  list of NER entities from query,
          }
        """
        # Step 1 — FAISS retrieval
        vec = self._encode_query(query)
        scores, indices = self.index.search(vec, self.top_k)

        chunks = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            chunks.append({
                "chunk_id":    meta["chunk_id"],
                "text":        meta.get("text", ""),
                "source_file": meta.get("source_file", ""),
                "page":        meta.get("page", ""),
                "vector_score": round(float(score), 4),
            })

        # Step 2 — Entity collection
        query_entities = self._extract_query_entities(query)
        entities = self._collect_entities(query_entities, chunks)

        # Step 3 — Graph neighbor lookup
        entity_neighbors = self._build_entity_neighbors(entities)

        return {
            "chunks": chunks,
            "entity_neighbors": entity_neighbors,
            "query_entities": query_entities,
        }


# ------------------------------------------------------------------
# Quick demo
# ------------------------------------------------------------------

if __name__ == "__main__":
    retriever = GraphRetriever(top_k=5)

    queries = [
        "What are the WHO guidelines for HIV treatment in adolescents?",
        "How is infertility diagnosed?",
        "What are the risks of abortion care?",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = retriever.retrieve(query)
        print(f"Query entities: {result['query_entities']}")
        print(f"Chunks retrieved: {len(result['chunks'])}")
        for c in result["chunks"]:
            print(f"  [{c['vector_score']}] {c['source_file']} p.{c['page']} — {c['text'][:80]}...")
        print(f"Entity neighbors ({len(result['entity_neighbors'])} entities):")
        for ent, neighbors in list(result["entity_neighbors"].items())[:3]:
            print(f"  {ent}: {neighbors}")
        print("-" * 70)
