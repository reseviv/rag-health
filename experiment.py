"""
experiment.py — Rotation experiment runner

Tests all combinations of:
  LLMs      × agent modes × retrieval methods
  (qwen, deepseek, gpt)  ×  (single, linear, adaptive, graph)  ×  (vanilla, graphrag, hybrid)

Results saved to results/experiment_<timestamp>.json

Usage:
    python experiment.py                        # run all combinations
    python experiment.py --llm qwen             # single LLM only
    python experiment.py --mode single linear   # specific modes only
    python experiment.py --db vanilla hybrid    # specific retrievers only
    python experiment.py --questions my_qs.json # custom question file
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

from graph_retrieve import GraphRetriever
from llm import LLMClient

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

# Retrieval method variants — same GraphRetriever, different alpha/beta weights
RETRIEVER_CONFIGS: dict[str, dict] = {
    "vanilla":  {"alpha": 0.0, "beta": 1.0},   # FAISS only
    "graphrag": {"alpha": 1.0, "beta": 0.0},   # PageRank only
    "hybrid":   {"alpha": 0.4, "beta": 0.6},   # both combined
}

# LLMs to test — remove keys you don't have set up
DEFAULT_LLMS = ["qwen", "deepseek-local"]  # add "deepseek-api" or "gpt" if keys available

# Sample test questions — replace with GraphRAG-Benchmark medical dataset questions
TEST_QUESTIONS = [
    "What are the WHO recommendations for antiretroviral therapy in HIV patients?",
    "What are the diagnostic criteria for infertility according to WHO guidelines?",
    "What complications should be monitored after an abortion procedure?",
    "How should post-abortion care be provided to patients?",
    "What are the first-line treatments for advanced HIV disease?",
    "When should HIV testing be recommended during pregnancy?",
    "What are the WHO guidelines for managing HIV in adolescents?",
    "What surgical options exist for treating infertility?",
]

RESULTS_DIR = Path("results")
TOP_K = 5
ADAPTIVE_SCORE_THRESHOLD = 0.35   # if avg fused score < this, rewrite and retry


# ------------------------------------------------------------------
# Agent mode implementations
# ------------------------------------------------------------------

def run_single(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Single mode: one retrieve → one generate.
    Simplest baseline.
    """
    chunks = retriever.retrieve(query)
    answer = llm.generate(query, chunks)
    return {"answer": answer, "chunks": chunks}


def run_linear(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Linear mode: fixed sequential pipeline.
      1. Rewrite query for better retrieval
      2. Retrieve with rewritten query
      3. Generate answer
      4. Verify answer is grounded
    """
    rewritten = llm.rewrite_query(query)
    chunks = retriever.retrieve(rewritten)
    answer = llm.generate(query, chunks)       # use original query for generation
    grounded = llm.verify_answer(query, answer, chunks)
    return {
        "answer": answer,
        "chunks": chunks,
        "rewritten_query": rewritten,
        "grounded": grounded,
    }


def run_adaptive(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Adaptive mode: retrieve → check quality → optionally rewrite and retry.
    Mimics an agent that decides whether it needs better context.
    """
    chunks = retriever.retrieve(query)
    avg_score = sum(c["fused_score"] for c in chunks) / len(chunks) if chunks else 0

    rewritten = None
    if avg_score < ADAPTIVE_SCORE_THRESHOLD:
        rewritten = llm.rewrite_query(query)
        retry_chunks = retriever.retrieve(rewritten)
        # Keep whichever retrieval scored higher
        retry_avg = sum(c["fused_score"] for c in retry_chunks) / len(retry_chunks) if retry_chunks else 0
        if retry_avg > avg_score:
            chunks = retry_chunks
            avg_score = retry_avg

    answer = llm.generate(query, chunks)
    return {
        "answer": answer,
        "chunks": chunks,
        "avg_retrieval_score": round(avg_score, 4),
        "rewritten_query": rewritten,
        "retrieval_retried": rewritten is not None,
    }


def run_graph_mode(query: str, retriever: GraphRetriever, llm: LLMClient) -> dict:
    """
    Graph mode: decompose query → retrieve for each sub-question → merge → generate.
    Multi-agent style: each sub-question is handled independently then results merged.
    """
    sub_questions = llm.decompose_query(query)
    if not sub_questions:
        sub_questions = [query]

    all_chunks: list[dict] = []
    for sq in sub_questions:
        sq_chunks = retriever.retrieve(sq)
        all_chunks.extend(sq_chunks)

    # Deduplicate by chunk_id, keep highest fused_score per chunk
    best: dict[str, dict] = {}
    for c in all_chunks:
        cid = c["chunk_id"]
        if cid not in best or c["fused_score"] > best[cid]["fused_score"]:
            best[cid] = c

    merged = sorted(best.values(), key=lambda x: x["fused_score"], reverse=True)[:TOP_K]
    answer = llm.generate(query, merged)
    return {
        "answer": answer,
        "chunks": merged,
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
    """Store only essential chunk fields to keep results file small."""
    return [
        {
            "chunk_id": c.get("chunk_id"),
            "source_file": c.get("source_file"),
            "page": c.get("page"),
            "graph_score": c.get("graph_score"),
            "vector_score": c.get("vector_score"),
            "fused_score": c.get("fused_score"),
        }
        for c in chunks
    ]


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------

def run_experiments(
    llm_keys: list[str],
    mode_names: list[str],
    db_names: list[str],
    questions: list,   # list of str OR list of {query, ground_truth, task_type}
) -> None:

    # Init retrievers once — expensive to reload each time
    print("Initialising retrievers...")
    retrievers: dict[str, GraphRetriever] = {
        name: GraphRetriever(alpha=cfg["alpha"], beta=cfg["beta"], top_k=TOP_K)
        for name, cfg in RETRIEVER_CONFIGS.items()
        if name in db_names
    }

    # Init LLM clients
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

    # Experiment loop
    results = []
    conditions = [
        (q, lk, mn, dn)
        for q in questions
        for lk in llms
        for mn in mode_names
        for dn in db_names
        if dn in retrievers
    ]
    total = len(conditions)
    print(f"\nRunning {total} experiment conditions "
          f"({len(questions)} questions × {len(llms)} LLMs × {len(mode_names)} modes × {len(db_names)} DBs)\n")

    for i, (q_item, llm_key, mode_name, db_name) in enumerate(conditions, 1):
        # Support both plain strings and {query, ground_truth, task_type} dicts
        if isinstance(q_item, dict):
            question     = q_item["query"]
            ground_truth = q_item.get("ground_truth", "")
            task_type    = q_item.get("task_type", "unknown")
        else:
            question, ground_truth, task_type = q_item, "", "unknown"

        print(f"[{i}/{total}] {llm_key} | {mode_name} | {db_name} | {task_type}")

        t0 = time.perf_counter()
        try:
            output = AGENT_MODES[mode_name](question, retrievers[db_name], llms[llm_key])
            output["status"] = "ok"
        except Exception as e:
            output = {"answer": None, "chunks": [], "status": "error", "error": str(e)}
            print(f"  ERROR: {e}")

        latency_ms = round((time.perf_counter() - t0) * 1000)

        record = {
            "llm": llm_key,
            "agent_mode": mode_name,
            "retriever": db_name,
            "task_type": task_type,
            "query": question,
            "ground_truth": ground_truth,
            "answer": output.get("answer"),
            "status": output["status"],
            "latency_ms": latency_ms,
            "chunks": _strip_chunks_for_storage(output.get("chunks", [])),
            # mode-specific extras
            "rewritten_query": output.get("rewritten_query"),
            "sub_questions": output.get("sub_questions"),
            "grounded": output.get("grounded"),
            "retrieval_retried": output.get("retrieval_retried"),
            "avg_retrieval_score": output.get("avg_retrieval_score"),
        }
        results.append(record)
        print(f"  → {latency_ms}ms | status={record['status']}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"experiment_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'='*60}")
    print(f"Done. {ok}/{total} successful → {out_file}")
    print(f"{'='*60}\n")

    # Per-condition latency summary
    print(f"{'LLM':<18} {'Mode':<12} {'DB':<10} {'Avg latency'}")
    from itertools import groupby
    key_fn = lambda r: (r["llm"], r["agent_mode"], r["retriever"])
    for (lk, mn, dn), group in groupby(sorted(results, key=key_fn), key=key_fn):
        rows = list(group)
        ok_rows = [r for r in rows if r["status"] == "ok"]
        avg_ms = round(sum(r["latency_ms"] for r in ok_rows) / len(ok_rows)) if ok_rows else 0
        print(f"  {lk:<16} {mn:<12} {dn:<10} {avg_ms}ms")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG-Health experiment runner")
    parser.add_argument("--llm", nargs="+", default=DEFAULT_LLMS,
                        choices=list({"qwen", "deepseek-local", "deepseek-api"}),  # add "gpt" back once OPENAI_API_KEY is set
                        help="LLMs to test")
    parser.add_argument("--mode", nargs="+", default=list(AGENT_MODES),
                        choices=list(AGENT_MODES),
                        help="Agent modes to test")
    parser.add_argument("--db", nargs="+", default=list(RETRIEVER_CONFIGS),
                        choices=list(RETRIEVER_CONFIGS),
                        help="Retrieval methods to test")
    parser.add_argument("--questions", type=str, default=None,
                        help="Path to JSON file with list of questions")
    args = parser.parse_args()

    llm_keys = args.llm
    # GPT disabled — to reactivate: uncomment "gpt" in llm.py MODEL_CONFIGS,
    # add "gpt" to choices above, set OPENAI_API_KEY in .env, then pass --llm gpt

    # Load questions
    if args.questions:
        with open(args.questions, encoding="utf-8") as f:
            questions = json.load(f)
    else:
        questions = TEST_QUESTIONS

    run_experiments(
        llm_keys=llm_keys,
        mode_names=args.mode,
        db_names=args.db,
        questions=questions,
    )


if __name__ == "__main__":
    main()
