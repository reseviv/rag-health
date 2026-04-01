"""
eval/load_benchmark.py — Fetch GraphRAG-Benchmark medical questions

Downloads the medical dataset from GraphRAG-Benchmark (HuggingFace)
and writes eval/benchmark_questions.json for use in experiment.py.

Usage:
    python eval/load_benchmark.py
    python experiment.py --questions eval/benchmark_questions.json

Requirements:
    pip install datasets
"""

import json
from pathlib import Path

OUT_FILE = Path("eval/benchmark_questions.json")


def load_from_huggingface() -> list[dict]:
    from datasets import load_dataset

    print("Loading GraphRAG-Benchmark medical dataset from HuggingFace...")
    # Medical subset — 4 task types: fact_retrieval, complex_reasoning,
    # contextual_summarization, creative_generation
    ds = load_dataset("GraphRAG-Bench/GraphRAG-Benchmark", "medical", split="test")

    questions = []
    for row in ds:
        questions.append({
            "query": row["question"],
            "ground_truth": row.get("answer", ""),
            "task_type": row.get("task_type", "unknown"),
        })

    print(f"Loaded {len(questions)} questions across task types:")
    from collections import Counter
    counts = Counter(q["task_type"] for q in questions)
    for task, count in sorted(counts.items()):
        print(f"  {task}: {count}")

    return questions


def main() -> None:
    try:
        questions = load_from_huggingface()
    except Exception as e:
        print(f"HuggingFace load failed: {e}")
        print("Falling back to placeholder questions.")
        print("Install datasets: pip install datasets")
        questions = _placeholder_questions()

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(questions)} questions -> {OUT_FILE}")
    print(f"Run experiments with: python experiment.py --questions {OUT_FILE}")


def _placeholder_questions() -> list[dict]:
    """Minimal fallback if HuggingFace is unavailable."""
    return [
        {"query": "What are the WHO recommendations for antiretroviral therapy in HIV patients?",
         "ground_truth": "", "task_type": "fact_retrieval"},
        {"query": "What are the diagnostic criteria for infertility according to WHO guidelines?",
         "ground_truth": "", "task_type": "fact_retrieval"},
        {"query": "What complications should be monitored after an abortion procedure?",
         "ground_truth": "", "task_type": "complex_reasoning"},
        {"query": "How should post-abortion care be provided to patients?",
         "ground_truth": "", "task_type": "complex_reasoning"},
        {"query": "What are the first-line treatments for advanced HIV disease?",
         "ground_truth": "", "task_type": "fact_retrieval"},
        {"query": "When should HIV testing be recommended during pregnancy?",
         "ground_truth": "", "task_type": "contextual_summarization"},
        {"query": "What are the WHO guidelines for managing HIV in adolescents?",
         "ground_truth": "", "task_type": "complex_reasoning"},
        {"query": "What surgical options exist for treating infertility?",
         "ground_truth": "", "task_type": "fact_retrieval"},
    ]


if __name__ == "__main__":
    main()
