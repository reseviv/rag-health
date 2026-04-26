"""
pipeline.py — Experiment runner for the graph-enriched RAG pipeline

Tests all combinations of: LLMs × agent modes
  - LLMs:        qwen, qwen-small, deepseek-local, deepseek-api
  - Agent modes: single, linear, adaptive, graph

Retrieval is now a single GraphRetriever (no alpha/beta variants).
Each retrieval returns chunks + entity_neighbors from the graph.

Results saved to results/experiment_<timestamp>.json

Usage:
    python pipeline.py                             # run all combinations
    python pipeline.py --llm qwen                  # single LLM only
    python pipeline.py --mode single linear        # specific modes only
    python pipeline.py --questions path/to/qa.json # custom question file
    python pipeline.py --split simple              # only simple (non-multihop) questions
    python pipeline.py --split multihop            # only multihop questions
"""

import argparse
import json
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path

from graph_retrieve import GraphRetriever
from llm_v2 import LLMClient

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

DEFAULT_LLMS = ["qwen", "deepseek-local"]
DEFAULT_DATASET = Path("data/label/WHO_4doc_labelled_questions_240.json")
RESULTS_DIR = Path("results")
TOP_K = 5

# Adaptive mode: if fewer than this fraction of entities found graph neighbors, retry
ADAPTIVE_ENTITY_THRESHOLD = 0.3


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_questions(path: Path, split_filter: str | None = None) -> list[dict]:
    """
    Load questions from the nested dataset format.
    Returns flat list of dicts with keys:
      question, gold_answer, split, requires_multihop, id, document_id, gold_entities, gold_relations
    split_filter: "simple" | "multihop" | None (all)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for doc in data.get("documents", []):
        for item in doc.get("items", []):
            items.append({
                "id":              item["id"],
                "document_id":     doc["document_id"],
                "question":        item["question"],
                "gold_answer":     item.get("gold_answer", ""),
                "split":           item.get("split", "unknown"),
                "requires_multihop": item.get("requires_multihop", False),
                "gold_entities":   item.get("gold_entities", []),
                "gold_relations":  item.get("gold_relations", []),
                "source_section":  item.get("source_section", ""),
                "source_page":     item.get("source_page", ""),
            })

    if split_filter == "multihop":
        items = [i for i in items if i["requires_multihop"]]
    elif split_filter == "simple":
        items = [i for i in items if not i["requires_multihop"]]

    return items


# ------------------------------------------------------------------
# Agent mode implementations
# ------------------------------------------------------------------

def run_single(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """Single mode: retrieve → generate with graph entity hints."""
    result = retriever.retrieve(query)
    answer = llm.generate(query, result["chunks"], result["entity_neighbors"])
    return {
        "answer": answer,
        "chunks": result["chunks"],
        "entity_neighbors": result["entity_neighbors"],
        "query_entities": result["query_entities"],
    }


def run_linear(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Linear mode: rewrite query → retrieve → generate with hints → verify grounding.
    Uses original query for generation to avoid drift from rewrite.
    """
    rewritten = llm.rewrite_query(query)
    result = retriever.retrieve(rewritten)
    answer = llm.generate(query, result["chunks"], result["entity_neighbors"])
    grounded = llm.verify_answer(query, answer, result["chunks"])
    return {
        "answer": answer,
        "chunks": result["chunks"],
        "entity_neighbors": result["entity_neighbors"],
        "query_entities": result["query_entities"],
        "rewritten_query": rewritten,
        "grounded": grounded,
    }


def run_adaptive(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Adaptive mode: retrieve → check graph coverage → optionally rewrite and retry.
    Quality signal: entity match rate = entities with graph neighbors / total entities.
    If rate < ADAPTIVE_ENTITY_THRESHOLD, rewrite query and retry.
    """
    result = retriever.retrieve(query)
    all_entities = result["query_entities"] + [
        e for chunk in result["chunks"]
        for e in retriever.chunk_to_entities.get(chunk["chunk_id"], [])
    ]
    total_entities = len(set(all_entities))
    matched_entities = len(result["entity_neighbors"])
    entity_match_rate = matched_entities / max(total_entities, 1)

    rewritten = None
    retrieval_retried = False

    if entity_match_rate < ADAPTIVE_ENTITY_THRESHOLD:
        rewritten = llm.rewrite_query(query)
        retry_result = retriever.retrieve(rewritten)
        retry_total = len(set(
            retry_result["query_entities"] + [
                e for chunk in retry_result["chunks"]
                for e in retriever.chunk_to_entities.get(chunk["chunk_id"], [])
            ]
        ))
        retry_rate = len(retry_result["entity_neighbors"]) / max(retry_total, 1)

        if retry_rate > entity_match_rate:
            result = retry_result
            entity_match_rate = retry_rate
            retrieval_retried = True

    answer = llm.generate(query, result["chunks"], result["entity_neighbors"])
    return {
        "answer": answer,
        "chunks": result["chunks"],
        "entity_neighbors": result["entity_neighbors"],
        "query_entities": result["query_entities"],
        "rewritten_query": rewritten,
        "retrieval_retried": retrieval_retried,
        "entity_match_rate": round(entity_match_rate, 4),
    }


def run_graph_mode(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Graph mode: decompose query → retrieve for each sub-question → merge → generate.
    Chunks deduplicated by chunk_id (highest vector_score kept).
    Entity neighbors merged across all sub-question retrievals.
    """
    sub_questions = llm.decompose_query(query)
    if not sub_questions:
        sub_questions = [query]

    best_chunks: dict[str, dict] = {}
    merged_entity_neighbors: dict[str, list] = {}
    all_query_entities: list[str] = []

    for sq in sub_questions:
        result = retriever.retrieve(sq)

        for chunk in result["chunks"]:
            cid = chunk["chunk_id"]
            if cid not in best_chunks or chunk["vector_score"] > best_chunks[cid]["vector_score"]:
                best_chunks[cid] = chunk

        for entity, neighbors in result["entity_neighbors"].items():
            if entity not in merged_entity_neighbors:
                merged_entity_neighbors[entity] = neighbors

        all_query_entities.extend(result["query_entities"])

    chunks = sorted(best_chunks.values(), key=lambda x: x["vector_score"], reverse=True)[:TOP_K]
    answer = llm.generate(query, chunks, merged_entity_neighbors)

    return {
        "answer": answer,
        "chunks": chunks,
        "entity_neighbors": merged_entity_neighbors,
        "query_entities": list(set(all_query_entities)),
        "sub_questions": sub_questions,
    }


AGENT_MODES: dict[str, callable] = {
    "single":   run_single,
    "linear":   run_linear,
    "adaptive": run_adaptive,
    "graph":    run_graph_mode,
}


# ------------------------------------------------------------------
# Result helpers
# ------------------------------------------------------------------

def _strip_chunks_for_storage(chunks: list[dict]) -> list[dict]:
    return [
        {
            "chunk_id":    c.get("chunk_id"),
            "source_file": c.get("source_file"),
            "page":        c.get("page"),
            "vector_score": c.get("vector_score"),
        }
        for c in chunks
    ]


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------

def run_experiments(
    llm_keys: list[str],
    mode_names: list[str],
    questions: list[dict],
    prior_results: list[dict] | None = None,
) -> None:

    print("Initialising retriever...")
    retriever = GraphRetriever(top_k=TOP_K)

    print("Initialising LLMs...")
    llms: dict[str, LLMClient] = {}
    for key in llm_keys:
        try:
            llms[key] = LLMClient(key)
            print(f"  ✓ {key}")
        except Exception as e:
            print(f"  ✗ {key} — skipped ({e})")

    if not llms:
        print("No LLMs available. Exiting.")
        return

    conditions = [
        (q, lk, mn)
        for q in questions
        for lk in llms
        for mn in mode_names
    ]
    total = len(conditions)
    print(
        f"\nRunning {total} conditions "
        f"({len(questions)} questions × {len(llms)} LLMs × {len(mode_names)} modes)\n"
    )

    # Carry over successful results from a prior run
    results = [r for r in (prior_results or []) if r["status"] == "ok"]

    for i, (q_item, llm_key, mode_name) in enumerate(conditions, 1):
        question = q_item["question"]
        print(
            f"[{i}/{total}] {llm_key} | {mode_name} | "
            f"{q_item['split']} | multihop={q_item['requires_multihop']}"
        )

        t0 = time.perf_counter()
        try:
            output = AGENT_MODES[mode_name](question, retriever, llms[llm_key])
            output["status"] = "ok"
        except Exception as e:
            output = {
                "answer": None, "chunks": [], "entity_neighbors": {},
                "status": "error", "error": str(e),
            }
            print(f"  ERROR: {e}")

        latency_ms = round((time.perf_counter() - t0) * 1000)

        record = {
            # Question metadata
            "id":               q_item["id"],
            "document_id":      q_item["document_id"],
            "split":            q_item["split"],
            "requires_multihop": q_item["requires_multihop"],
            "query":            question,
            "gold_answer":      q_item["gold_answer"],
            "gold_entities":    q_item["gold_entities"],
            "gold_relations":   q_item["gold_relations"],
            # Run metadata
            "llm":              llm_key,
            "agent_mode":       mode_name,
            "status":           output["status"],
            "latency_ms":       latency_ms,
            # Outputs
            "answer":           output.get("answer"),
            "chunks":           _strip_chunks_for_storage(output.get("chunks", [])),
            "entity_neighbors_count": len(output.get("entity_neighbors", {})),
            "query_entities":   output.get("query_entities", []),
            # Mode-specific extras
            "rewritten_query":  output.get("rewritten_query"),
            "sub_questions":    output.get("sub_questions"),
            "grounded":         output.get("grounded"),
            "retrieval_retried": output.get("retrieval_retried"),
            "entity_match_rate": output.get("entity_match_rate"),
        }
        results.append(record)
        print(f"  → {latency_ms}ms | status={record['status']}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"experiment_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out_file

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*60}")
    print(f"Done. {ok}/{total} successful → {out_file}")
    print(f"{'='*60}\n")

    # Per-condition latency summary
    print(f"{'LLM':<18} {'Mode':<12} {'Avg latency'}")
    key_fn = lambda r: (r["llm"], r["agent_mode"])
    for (lk, mn), group in groupby(sorted(results, key=key_fn), key=key_fn):
        rows = [r for r in list(group) if r["status"] == "ok"]
        avg_ms = round(sum(r["latency_ms"] for r in rows) / len(rows)) if rows else 0
        print(f"  {lk:<16} {mn:<12} {avg_ms}ms")

    # Multihop vs simple breakdown
    print(f"\n{'Split':<12} {'Mode':<12} {'Avg latency'}")
    key_fn2 = lambda r: (r["requires_multihop"], r["agent_mode"])
    for (multihop, mn), group in groupby(sorted(results, key=key_fn2), key=key_fn2):
        rows = [r for r in list(group) if r["status"] == "ok"]
        avg_ms = round(sum(r["latency_ms"] for r in rows) / len(rows)) if rows else 0
        label = "multihop" if multihop else "simple"
        print(f"  {label:<12} {mn:<12} {avg_ms}ms")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Health experiment runner")
    parser.add_argument(
        "--llm", nargs="+", default=DEFAULT_LLMS,
        choices=list(MODEL_CHOICES := {"qwen", "qwen-small", "deepseek-local", "deepseek-api"}),
        help="LLMs to test",
    )
    parser.add_argument(
        "--mode", nargs="+", default=list(AGENT_MODES),
        choices=list(AGENT_MODES),
        help="Agent modes to test",
    )
    parser.add_argument(
        "--questions", type=str, default=None,
        help=f"Path to dataset JSON (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--split", choices=["simple", "multihop"], default=None,
        help="Filter questions by split type (default: all questions)",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a prior results JSON — reruns only failed questions and merges output",
    )
    args = parser.parse_args()

    prior_results = None
    questions_override = None

    if args.resume:
        resume_path = Path(args.resume)
        with open(resume_path, encoding="utf-8") as f:
            prior_results = json.load(f)
        failed_ids = {r["id"] for r in prior_results if r["status"] != "ok"}
        print(f"Resuming from {resume_path}: {len(failed_ids)} failed questions to retry")

        # Reconstruct question dicts from the failed records
        questions_override = [
            {
                "id":               r["id"],
                "document_id":      r["document_id"],
                "split":            r["split"],
                "requires_multihop": r["requires_multihop"],
                "question":         r["query"],
                "gold_answer":      r["gold_answer"],
                "gold_entities":    r["gold_entities"],
                "gold_relations":   r["gold_relations"],
                "source_section":   "",
                "source_page":      "",
            }
            for r in prior_results
            if r["status"] != "ok"
        ]

    dataset_path = Path(args.questions) if args.questions else DEFAULT_DATASET
    questions = questions_override or load_questions(dataset_path, split_filter=args.split)

    if not questions_override:
        print(f"Loaded {len(questions)} questions from {dataset_path}"
              + (f" (filter: {args.split})" if args.split else ""))

    run_experiments(
        llm_keys=args.llm,
        mode_names=args.mode,
        questions=questions,
        prior_results=prior_results,
    )


if __name__ == "__main__":
    main()
