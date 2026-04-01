"""
graph_build.py — Indexing phase (run once after chunk.py)

Reads chunks.json, extracts entities with scispaCy biomedical NER
(en_ner_bc5cdr_md), builds a co-occurrence graph with NetworkX,
and saves it to index/graph.pkl.

Usage:
    pip install scispacy
    pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
    python graph_build.py
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import spacy

CHUNKS_FILE = Path("data/processed/chunks.json")
GRAPH_FILE = Path("index/graph.pkl")


def build_graph(chunks_path: Path = CHUNKS_FILE, graph_path: Path = GRAPH_FILE) -> None:
    print("Loading scispaCy biomedical NER model (en_ner_bc5cdr_md)...")
    nlp = spacy.load("en_ner_bc5cdr_md")

    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    G = nx.Graph()
    entity_to_chunks: dict[str, list[str]] = defaultdict(list)   # entity -> [chunk_id, ...]
    chunk_to_entities: dict[str, list[str]] = defaultdict(list)  # chunk_id -> [entity, ...]

    print(f"Extracting entities from {len(chunks)} chunks...")
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        doc = nlp(chunk["text"])

        # SpaCy named entities — lowercase, deduplicated per chunk
        entities = {
            ent.text.lower().strip()
            for ent in doc.ents
            if len(ent.text.strip()) > 2
        }

        for entity in entities:
            G.add_node(entity)
            entity_to_chunks[entity].append(chunk_id)
            chunk_to_entities[chunk_id].append(entity)

        # Co-occurrence edges (weighted by how often two entities share a chunk)
        entity_list = list(entities)
        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                e1, e2 = entity_list[i], entity_list[j]
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                else:
                    G.add_edge(e1, e2, weight=1)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump({
            "graph": G,
            "entity_to_chunks": dict(entity_to_chunks),
            "chunk_to_entities": dict(chunk_to_entities),
        }, f)

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Saved -> {graph_path}")


if __name__ == "__main__":
    build_graph()
