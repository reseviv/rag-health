"""
eval/evaluate.py — Scoring harness for experiment results

Reads:  results/experiment_*.json  (output of experiment.py)
Scores: Faithfulness, Answer Relevancy, Context Recall (RAGAS)
        ROUGE-L (rouge_score)
        Latency stats (already in results)
Writes: results/scores_<timestamp>.json  +  prints summary table

Usage:
    python eval/evaluate.py                              # scores latest results file
    python eval/evaluate.py --file results/experiment_20260330_123456.json
    python eval/evaluate.py --file results/experiment_*.json --no-ragas  # ROUGE-L only (no API key needed)

Requirements:
    pip install ragas rouge_score datasets
    OPENAI_API_KEY must be set for RAGAS (uses LLM-as-judge)
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

# ------------------------------------------------------------------
# Lazy imports — only loaded when needed
# ------------------------------------------------------------------

def _load_ragas():
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_recall
    from datasets import Dataset
    return evaluate, faithfulness, answer_relevancy, context_recall, Dataset

def _load_rouge():
    from rouge_score import rouge_scorer
    return rouge_scorer

RESULTS_DIR = Path("results")


# ------------------------------------------------------------------
# ROUGE-L
# ------------------------------------------------------------------

def score_rouge(results: list[dict]) -> float | None:
    """Compute mean ROUGE-L for a list of results. Returns None if no ground_truth available."""
    rouge_scorer_mod = _load_rouge()
    scorer = rouge_scorer_mod.RougeScorer(["rougeL"], use_stemmer=True)

    scores = []
    for r in results:
        if not r.get("answer") or not r.get("ground_truth"):
            continue
        s = scorer.score(r["ground_truth"], r["answer"])
        scores.append(s["rougeL"].fmeasure)

    return round(sum(scores) / len(scores), 4) if scores else None


# ------------------------------------------------------------------
# RAGAS
# ------------------------------------------------------------------

def score_ragas(results: list[dict]) -> dict:
    """
    Run RAGAS metrics on a list of results (can be a condition subset).
    Requires OPENAI_API_KEY (RAGAS uses LLM-as-judge internally).

    faithfulness     — is the answer grounded in retrieved chunks?
    answer_relevancy — does the answer address the question?
    context_recall   — did we retrieve the right chunks? (needs ground_truth)
    """
    evaluate, faithfulness, answer_relevancy, context_recall, Dataset = _load_ragas()

    rows = []
    for r in results:
        if not r.get("answer") or not r.get("chunks"):
            continue
        rows.append({
            "question": r["query"],
            "answer": r["answer"],
            "contexts": [c.get("text", "") for c in r.get("chunks", []) if c.get("text")],
            "ground_truth": r.get("ground_truth", ""),
        })

    if not rows:
        return {}

    dataset = Dataset.from_list(rows)

    metrics = [faithfulness, answer_relevancy]
    if any(r.get("ground_truth") for r in rows):
        metrics.append(context_recall)

    result = evaluate(dataset, metrics=metrics)
    return {k: round(float(v), 4) for k, v in result.items()}


# ------------------------------------------------------------------
# Aggregate scores by condition
# ------------------------------------------------------------------

def aggregate(results: list[dict], use_ragas: bool = False) -> list[dict]:
    """
    Group results by (llm, agent_mode, retriever) and compute per-condition scores.
    ROUGE-L and RAGAS are computed independently for each condition group.
    """
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)
    for r in results:
        key = (r["llm"], r["agent_mode"], r["retriever"])
        groups[key].append(r)

    rows = []
    for (llm, mode, db), group in sorted(groups.items()):
        ok = [r for r in group if r["status"] == "ok"]
        latencies = [r["latency_ms"] for r in ok]

        ragas = score_ragas(ok) if use_ragas else {}

        rows.append({
            "llm": llm,
            "agent_mode": mode,
            "retriever": db,
            "n_total": len(group),
            "n_ok": len(ok),
            "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
            "rouge_l": score_rouge(ok),
            "faithfulness": ragas.get("faithfulness"),
            "answer_relevancy": ragas.get("answer_relevancy"),
            "context_recall": ragas.get("context_recall"),
        })

    return rows


# ------------------------------------------------------------------
# Summary table printer
# ------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    header = f"{'LLM':<18} {'Mode':<12} {'DB':<10} {'OK':>4} {'Lat(ms)':>8} {'Faith':>7} {'AnsRel':>7} {'ROUGE-L':>8}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        faith  = f"{r['faithfulness']:.3f}"  if r['faithfulness']  is not None else "  —  "
        ansrel = f"{r['answer_relevancy']:.3f}" if r['answer_relevancy'] is not None else "  —  "
        rouge  = f"{r['rouge_l']:.3f}"       if r['rouge_l']       is not None else "  —  "
        lat    = str(r['avg_latency_ms']) if r['avg_latency_ms'] is not None else "—"
        print(f"{r['llm']:<18} {r['agent_mode']:<12} {r['retriever']:<10} "
              f"{r['n_ok']:>4} {lat:>8} {faith:>7} {ansrel:>7} {rouge:>8}")
    print("=" * len(header) + "\n")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate experiment results")
    parser.add_argument("--file", type=str, default=None,
                        help="Path to results JSON. Defaults to latest in results/")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS (no OpenAI key needed) — ROUGE-L only")
    args = parser.parse_args()

    # Find results file
    if args.file:
        results_file = Path(args.file)
    else:
        files = sorted(RESULTS_DIR.glob("experiment_*.json"))
        if not files:
            print(f"No experiment results found in {RESULTS_DIR}/")
            print("Run experiment.py first.")
            return
        results_file = files[-1]
        print(f"Using latest results: {results_file}")

    with open(results_file, encoding="utf-8") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {results_file}")

    use_ragas = False
    if not args.no_ragas:
        if not os.getenv("OPENAI_API_KEY"):
            print("\nRAGAS skipped — OPENAI_API_KEY not set.")
            print("  Run with --no-ragas to suppress this message, or set OPENAI_API_KEY.")
        else:
            use_ragas = True
            print("\nRAGAS enabled — will score per condition (calls OpenAI API).")

    # Aggregate and score per condition
    print("\nAggregating per-condition scores...")
    rows = aggregate(results, use_ragas=use_ragas)
    print_table(rows)

    # Save scores
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"scores_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_file": str(results_file),
            "per_condition": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"Scores saved -> {out_file}")


if __name__ == "__main__":
    main()
