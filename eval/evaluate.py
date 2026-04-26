"""
eval/evaluate.py — Scoring harness for the graph-enriched RAG pipeline

Reads:   one or more results/experiment_*.json files
Scores:
  - ROUGE-L         lexical overlap with gold_answer
  - BERTScore F1    semantic similarity with gold_answer (local model, no API)
  - Entity Coverage fraction of gold_entities mentioned in the answer
Breakdown:
  - Per (llm, agent_mode) condition
  - Simple vs multihop split

Usage:
    python eval/evaluate.py                                   # score latest file
    python eval/evaluate.py --files results/exp_A.json results/exp_B.json
    python eval/evaluate.py --no-bert                        # ROUGE-L + entity only
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("results")


# ------------------------------------------------------------------
# Loaders
# ------------------------------------------------------------------

def load_results(paths: list[Path]) -> list[dict]:
    results = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            results.extend(json.load(f))
    return results


# ------------------------------------------------------------------
# ROUGE-L
# ------------------------------------------------------------------

def score_rouge_l(results: list[dict]) -> float | None:
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r["gold_answer"], r["answer"])["rougeL"].fmeasure
        for r in results
        if r.get("answer") and r.get("gold_answer")
    ]
    return round(sum(scores) / len(scores), 4) if scores else None


# ------------------------------------------------------------------
# BERTScore
# ------------------------------------------------------------------

def score_bert(results: list[dict]) -> float | None:
    from bert_score import score as bert_score_fn
    pairs = [
        (r["answer"], r["gold_answer"])
        for r in results
        if r.get("answer") and r.get("gold_answer")
    ]
    if not pairs:
        return None
    preds, refs = zip(*pairs)
    _, _, F1 = bert_score_fn(list(preds), list(refs), lang="en", verbose=False)
    return round(F1.mean().item(), 4)


# ------------------------------------------------------------------
# Entity Coverage
# ------------------------------------------------------------------

def score_entity_coverage(results: list[dict]) -> float | None:
    """
    Fraction of gold_entities that appear (case-insensitive) in the generated answer.
    Measures whether the answer mentions the right medical concepts.
    """
    scores = []
    for r in results:
        if not r.get("answer") or not r.get("gold_entities"):
            continue
        answer_lower = r["answer"].lower()
        hits = sum(
            1 for e in r["gold_entities"]
            if re.search(re.escape(e.lower()), answer_lower)
        )
        scores.append(hits / len(r["gold_entities"]))
    return round(sum(scores) / len(scores), 4) if scores else None


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def compute_scores(group: list[dict], use_bert: bool) -> dict:
    ok = [r for r in group if r.get("status") == "ok"]
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms")]
    return {
        "n_total": len(group),
        "n_ok": len(ok),
        "avg_latency_ms": round(sum(latencies) / len(latencies)) if latencies else None,
        "rouge_l": score_rouge_l(ok),
        "bert_score_f1": score_bert(ok) if use_bert else None,
        "entity_coverage": score_entity_coverage(ok),
    }


def aggregate(results: list[dict], use_bert: bool) -> dict:
    # Group by (llm, agent_mode)
    by_condition: dict[tuple, list] = defaultdict(list)
    for r in results:
        by_condition[(r["llm"], r["agent_mode"])].append(r)

    condition_rows = []
    for (llm, mode), group in sorted(by_condition.items()):
        row = {"llm": llm, "agent_mode": mode}
        row.update(compute_scores(group, use_bert))
        condition_rows.append(row)

    # Split breakdown: simple vs multihop, per (llm, agent_mode)
    split_rows = []
    for (llm, mode), group in sorted(by_condition.items()):
        for label, flag in [("simple", False), ("multihop", True)]:
            subset = [r for r in group if r.get("requires_multihop") == flag]
            if not subset:
                continue
            row = {"llm": llm, "agent_mode": mode, "split": label}
            row.update(compute_scores(subset, use_bert))
            split_rows.append(row)

    return {"by_condition": condition_rows, "by_split": split_rows}


# ------------------------------------------------------------------
# Printer
# ------------------------------------------------------------------

def _fmt(val, decimals=3) -> str:
    return f"{val:.{decimals}f}" if val is not None else "  —  "


def print_table(rows: list[dict], title: str, split_col: bool = False) -> None:
    col = f"{'Split':<10}" if split_col else ""
    header = (
        f"{'LLM':<16} {'Mode':<10} {col}"
        f"{'OK':>5} {'Lat(ms)':>8} {'ROUGE-L':>8} {'BERTScore':>10} {'EntCov':>8}"
    )
    print(f"\n{title}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in rows:
        split_str = f"{r.get('split', ''):<10}" if split_col else ""
        print(
            f"{r['llm']:<16} {r['agent_mode']:<10} {split_str}"
            f"{r['n_ok']:>5} {str(r['avg_latency_ms'] or '—'):>8} "
            f"{_fmt(r['rouge_l']):>8} {_fmt(r['bert_score_f1']):>10} "
            f"{_fmt(r['entity_coverage']):>8}"
        )
    print("=" * len(header))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG-Health experiment results")
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Result JSON files to evaluate. Defaults to latest two experiment files.",
    )
    parser.add_argument(
        "--no-bert", action="store_true",
        help="Skip BERTScore (faster, no model download needed)",
    )
    args = parser.parse_args()

    # Resolve files
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        all_files = sorted(RESULTS_DIR.glob("experiment_*.json"))
        if not all_files:
            print(f"No experiment results found in {RESULTS_DIR}/")
            return
        paths = all_files[-2:]  # default: latest two (simple + multihop)
        print(f"Using: {[str(p) for p in paths]}")

    results = load_results(paths)
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"Loaded {len(results)} results ({ok_count} successful) from {len(paths)} file(s)")

    use_bert = not args.no_bert
    if use_bert:
        print("Computing BERTScore (downloading model if needed)...")

    print("\nAggregating scores...")
    scores = aggregate(results, use_bert)

    print_table(scores["by_condition"], "Results by condition")
    print_table(scores["by_split"], "Results by split (simple vs multihop)", split_col=True)

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"scores_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "source_files": [str(p) for p in paths],
            "by_condition": scores["by_condition"],
            "by_split": scores["by_split"],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nScores saved -> {out_file}")


if __name__ == "__main__":
    main()
