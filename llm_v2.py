"""
llm_v2.py — LLM client for the graph-enriched pipeline

Changes from llm.py (teammate's file — not modified):
  - generate() accepts entity_hints dict as optional third parameter
  - Updated system prompt: chunk texts for citation, entity hints for reasoning only
  - _format_entity_hints() converts entity_neighbors dict to readable JSON block
  - Context length guard: warns if estimated prompt exceeds model limit
  - Timeout + retry (up to 2 attempts) on _call
  - Temperature exposed as parameter (default 0.1)

Setup:
    Same as llm.py — Ollama for local models, API keys in .env for cloud
"""

import json
import os
import time

from openai import OpenAI

MODEL_CONFIGS: dict[str, dict] = {
    "qwen": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:7b",
        "context_limit": 8192,
    },
    "qwen-small": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:1.5b",
        "context_limit": 4096,
    },
    "deepseek-local": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "deepseek-r1:7b",
        "context_limit": 8192,
    },
    "deepseek-api": {
        "base_url": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat",
        "context_limit": 32768,
    },
}

SYSTEM_PROMPT = """You are a clinical assistant answering questions based on WHO health guidelines.

Rules:
- Base your answer ONLY on the numbered source passages [1], [2], etc.
- Cite every claim with [1], [2], etc. matching the context numbers.
- The "Relevant concepts" section lists related medical concepts from a knowledge graph — \
use them to guide your reasoning but do not cite them as sources.
- If the answer is not in the source passages, say exactly: \
"This information is not available in the provided guidelines."
- Be concise and factual."""

# Rough chars-per-token estimate for context length guard
_CHARS_PER_TOKEN = 4


class LLMClient:
    def __init__(self, model_key: str):
        if model_key not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model key '{model_key}'. Choose from: {list(MODEL_CONFIGS)}"
            )
        cfg = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.model_name = cfg["model"]
        self.context_limit = cfg["context_limit"]

        client_kwargs = {"api_key": cfg["api_key"] or "no-key"}
        if cfg["base_url"]:
            client_kwargs["base_url"] = cfg["base_url"]
        self.client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        timeout: int = 120,
        retries: int = 2,
    ) -> str:
        last_err = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    timeout=timeout,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                if attempt < retries - 1:
                    time.sleep(2)
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")

    def _format_context(self, chunks: list[dict]) -> str:
        return "\n\n".join(
            f"[{i + 1}] {c['text']}\n(Source: {c['source_file']}, page {c['page']})"
            for i, c in enumerate(chunks)
        )

    def _format_entity_hints(self, entity_hints: dict[str, list]) -> str:
        """
        Convert entity_neighbors dict to a compact JSON block for the prompt.
        Format: {"entity": ["predicate:neighbor", ...], ...}
        """
        formatted = {
            entity: [
                f"{pred}:{nbr}" if pred != "co_occurs_with" else nbr
                for nbr, pred in neighbors
            ]
            for entity, neighbors in entity_hints.items()
        }
        return json.dumps(formatted, ensure_ascii=False)

    def _check_context_length(self, prompt: str) -> None:
        estimated_tokens = len(prompt) // _CHARS_PER_TOKEN
        if estimated_tokens > self.context_limit * 0.9:
            print(
                f"  WARNING: estimated prompt tokens ({estimated_tokens}) is near "
                f"context limit ({self.context_limit}) for {self.model_key}"
            )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        chunks: list[dict],
        entity_hints: dict | None = None,
        temperature: float = 0.1,
    ) -> str:
        """Generate a cited answer from retrieved chunks + optional graph entity hints."""
        context = self._format_context(chunks)

        if entity_hints:
            hints_str = self._format_entity_hints(entity_hints)
            user_content = (
                f"Source passages:\n{context}\n\n"
                f"Relevant concepts (for reasoning only, do not cite):\n{hints_str}\n\n"
                f"Question: {query}"
            )
        else:
            user_content = f"Source passages:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        self._check_context_length(user_content)
        return self._call(messages, temperature=temperature)

    def rewrite_query(self, query: str, temperature: float = 0.1) -> str:
        """Rewrite query to improve clinical retrieval (used in linear + adaptive modes)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the following clinical question to improve document retrieval. "
                    "Expand abbreviations, add clinical synonyms. "
                    "Return only the rewritten question, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ]
        return self._call(messages, temperature=temperature)

    def decompose_query(self, query: str, temperature: float = 0.1) -> list[str]:
        """Break a complex query into 2-3 sub-questions (used in graph agent mode)."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Break the following clinical question into 2-3 simpler sub-questions "
                    "that together cover the original question. "
                    "Return one sub-question per line, no numbering, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ]
        result = self._call(messages, temperature=temperature)
        return [q.strip("•-1234567890. ") for q in result.split("\n") if q.strip()]

    def verify_answer(
        self, query: str, answer: str, chunks: list[dict], temperature: float = 0.1
    ) -> bool:
        """Check whether the answer is grounded in the retrieved chunks (linear mode)."""
        context = self._format_context(chunks)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a fact-checker. Given a question, an answer, and source context, "
                    "reply with only YES if the answer is fully supported by the context, "
                    "or NO if it contains any unsupported claims."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nAnswer: {answer}\n\nContext:\n{context}",
            },
        ]
        verdict = self._call(messages, temperature=temperature).strip().upper()
        return verdict.startswith("YES")
