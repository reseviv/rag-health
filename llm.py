"""
llm.py — Unified LLM interface for all models

All models use the OpenAI-compatible API:
  - Qwen / DeepSeek (local): via Ollama at localhost:11434
  - DeepSeek (cloud):        via api.deepseek.com
  - GPT:                     via api.openai.com

Setup:
    ollama pull qwen2.5:7b
    ollama pull deepseek-r1:7b

    # Optional cloud APIs (add to .env):
    DEEPSEEK_API_KEY=...
    OPENAI_API_KEY=...
"""

import os
from openai import OpenAI

# ------------------------------------------------------------------
# Model registry — add / swap models here
# ------------------------------------------------------------------
MODEL_CONFIGS: dict[str, dict] = {
    # Local via Ollama
    "qwen": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "qwen2.5:7b",
        "source": "ollama",
    },
    "deepseek-local": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "model": "deepseek-r1:7b",
        "source": "ollama",
    },
    # Cloud APIs
    "deepseek-api": {
        "base_url": "https://api.deepseek.com",
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "model": "deepseek-chat",
        "source": "api",
    },
    # -- GPT (disabled) --------------------------------------------------
    # To reactivate:
    #   1. Add OPENAI_API_KEY=sk-... to your .env file
    #   2. Uncomment the block below
    #   3. Run: python experiment.py --llm gpt   (or add "gpt" to DEFAULT_LLMS)
    # --------------------------------------------------------------------
    # "gpt": {
    #     "base_url": None,
    #     "api_key": os.getenv("OPENAI_API_KEY", ""),
    #     "model": "gpt-4o-mini",
    #     "source": "api",
    # },
}

# System prompt used for all answer generation
SYSTEM_PROMPT = """You are a clinical assistant answering questions based on WHO health guidelines.
Rules:
- Answer ONLY using the provided context.
- Cite every claim with [1], [2], etc. matching the context numbers.
- If the answer is not in the context, say exactly: "This information is not available in the provided guidelines."
- Be concise and factual."""


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

class LLMClient:
    def __init__(self, model_key: str):
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODEL_CONFIGS)}")

        cfg = MODEL_CONFIGS[model_key]
        self.model_key = model_key
        self.model_name = cfg["model"]

        client_kwargs = {"api_key": cfg["api_key"] or "no-key"}
        if cfg["base_url"]:
            client_kwargs["base_url"] = cfg["base_url"]
        self.client = OpenAI(**client_kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, messages: list[dict], temperature: float = 0.1) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        return "\n\n".join(
            f"[{i + 1}] {c['text']}\n(Source: {c['source_file']}, page {c['page']})"
            for i, c in enumerate(chunks)
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate(self, query: str, chunks: list[dict]) -> str:
        """Generate a cited answer from retrieved chunks."""
        context = self._format_context(chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        return self._call(messages)

    def rewrite_query(self, query: str) -> str:
        """Rewrite query to improve clinical retrieval (used in Linear + Adaptive modes)."""
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
        return self._call(messages)

    def decompose_query(self, query: str) -> list[str]:
        """Break a complex query into 2-3 sub-questions (used in Graph mode)."""
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
        result = self._call(messages)
        return [q.strip("•-1234567890. ") for q in result.split("\n") if q.strip()]

    def verify_answer(self, query: str, answer: str, chunks: list[dict]) -> bool:
        """Check whether the answer is grounded in the retrieved chunks (used in Linear mode)."""
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
        verdict = self._call(messages).strip().upper()
        return verdict.startswith("YES")
