"""
graph_build.py — Indexing phase (run once after chunk.py)

Changes from v1:
  - NER model: en_core_sci_lg (was en_ner_bc5cdr_md) — fixes mismatch with query-time NER
  - Relations: spaCy dep parser extracts typed predicates (treat, recommend, etc.)
    Fallback: co_occurs_with at confidence=0.5 when no typed relation found
  - Edge attrs: predicate (str), weight (int, co-occurrence frequency), confidence (float)

Usage:
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
NER_MODEL = "en_core_sci_lg"


# ------------------------------------------------------------------
# Relation extraction via dependency parser
# ------------------------------------------------------------------

def _find_typed_relation(span1, span2, sent) -> str | None:
    """
    Look for a verb that syntactically connects two entity spans in a sentence.
    Returns the verb lemma if found, else None.
    """
    span1_indices = set(range(span1.start, span1.end))
    span2_indices = set(range(span2.start, span2.end))

    for token in sent:
        if token.pos_ != "VERB":
            continue

        subj_indices: set[int] = set()
        obj_indices: set[int] = set()

        for child in token.children:
            if child.dep_ in ("nsubj", "nsubjpass", "csubj", "csubjpass"):
                for t in child.subtree:
                    subj_indices.add(t.i)
            if child.dep_ in ("dobj", "attr", "pobj", "oprd"):
                for t in child.subtree:
                    obj_indices.add(t.i)

        connected = (
            (span1_indices & subj_indices and span2_indices & obj_indices) or
            (span2_indices & subj_indices and span1_indices & obj_indices)
        )
        if connected:
            return token.lemma_

    return None


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_graph(chunks_path: Path = CHUNKS_FILE, graph_path: Path = GRAPH_FILE) -> None:
    print(f"Loading scispaCy model ({NER_MODEL})...")
    nlp = spacy.load(NER_MODEL)

    print(f"Loading chunks from {chunks_path}...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    G = nx.Graph()
    entity_to_chunks: dict[str, list[str]] = defaultdict(list)
    chunk_to_entities: dict[str, list[str]] = defaultdict(list)

    print(f"Extracting entities and relations from {len(chunks)} chunks...")
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        doc = nlp(chunk["text"])

        # -- Entity extraction --
        # deduplicate by canonical name, keep last span for dep parsing
        entity_spans: dict[str, any] = {}
        for ent in doc.ents:
            name = ent.text.lower().strip()
            if len(name) > 2:
                entity_spans[name] = ent

        entity_names = list(entity_spans.keys())

        for name in entity_names:
            G.add_node(name)
            entity_to_chunks[name].append(chunk_id)
            chunk_to_entities[chunk_id].append(name)

        # -- Relation extraction --
        # Map entity name → sentence index
        sent_list = list(doc.sents)
        entity_to_sent_idx: dict[str, int] = {}
        for name, span in entity_spans.items():
            for si, sent in enumerate(sent_list):
                if sent.start <= span.start < sent.end:
                    entity_to_sent_idx[name] = si
                    break

        for i in range(len(entity_names)):
            for j in range(i + 1, len(entity_names)):
                e1, e2 = entity_names[i], entity_names[j]

                predicate = "co_occurs_with"
                confidence = 0.5

                # Try typed relation if both entities are in the same sentence
                si1 = entity_to_sent_idx.get(e1)
                si2 = entity_to_sent_idx.get(e2)
                if si1 is not None and si1 == si2:
                    typed = _find_typed_relation(
                        entity_spans[e1], entity_spans[e2], sent_list[si1]
                    )
                    if typed:
                        predicate = typed
                        confidence = 1.0

                # Add or update edge
                if G.has_edge(e1, e2):
                    G[e1][e2]["weight"] += 1
                    # Upgrade from co_occurs_with to typed if we find one
                    if predicate != "co_occurs_with" and G[e1][e2]["predicate"] == "co_occurs_with":
                        G[e1][e2]["predicate"] = predicate
                        G[e1][e2]["confidence"] = confidence
                else:
                    G.add_edge(e1, e2, predicate=predicate, weight=1, confidence=confidence)

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump({
            "graph": G,
            "entity_to_chunks": dict(entity_to_chunks),
            "chunk_to_entities": dict(chunk_to_entities),
            "ner_model": NER_MODEL,
        }, f)

    typed_edges = sum(1 for _, _, d in G.edges(data=True) if d["predicate"] != "co_occurs_with")
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"({typed_edges} typed, {G.number_of_edges() - typed_edges} co_occurs_with)")
    print(f"Saved -> {graph_path}")


if __name__ == "__main__":
    build_graph()
