"""LLM-based arbiter implementation."""

from __future__ import annotations

import os
from typing import Optional

from .base import ArbiterResult, BaseArbiter


class LLMArbiter(BaseArbiter):
    """LLM arbiter via OpenAI-compatible API."""

    SYSTEM_PROMPT = """You are an expert evaluator of text summaries. 
Your task is to compare two TL;DR summaries of a Reddit post and decide which one is better.

Evaluation criteria:
1. Accuracy: Does the summary capture the main points?
2. Conciseness: Is it brief without losing important information?
3. Clarity: Is it easy to understand?
4. Relevance: Does it focus on the most important aspects?

Respond with ONLY one of these options:
- "A" if summary A is better
- "B" if summary B is better  
- "TIE" if they are equally good"""

    USER_TEMPLATE = """Original post:
{prompt}

Summary A:
{response_a}

Summary B:
{response_b}

Which summary is better? Reply with A, B, or TIE."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        from openai import OpenAI

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url
        self._model = model

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    def compare(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        user_message = self.USER_TEMPLATE.format(
            prompt=prompt[:2000],
            response_a=response_a,
            response_b=response_b,
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=10,
                temperature=0.1,
            )

            answer = response.choices[0].message.content.strip().upper()

            if "A" in answer and "B" not in answer:
                winner = "a"
                score_a, score_b = 1.0, 0.0
            elif "B" in answer and "A" not in answer:
                winner = "b"
                score_a, score_b = 0.0, 1.0
            else:
                winner = "tie"
                score_a, score_b = 0.5, 0.5

            return ArbiterResult(
                winner=winner,
                score_a=score_a,
                score_b=score_b,
                reason=f"LLM ({self._model}) chose: {answer}",
                confidence=0.9,
            )

        except Exception as exc:  # pragma: no cover - network dependent
            print(f"LLM arbiter error: {exc}")
            return ArbiterResult(
                winner="tie",
                score_a=0.5,
                score_b=0.5,
                reason=f"Error: {str(exc)[:50]}",
                confidence=0.0,
            )
