"""Hybrid arbiter combining heuristics and LLM judging."""

from __future__ import annotations

from typing import Optional

from .base import ArbiterResult, BaseArbiter
from .heuristic import HeuristicArbiter
from .llm import LLMArbiter


class HybridArbiter(BaseArbiter):
    """Heuristic + LLM arbiter with confidence threshold."""

    def __init__(
        self,
        heuristic: Optional[HeuristicArbiter] = None,
        llm: Optional[LLMArbiter] = None,
        llm_threshold: float = 5.0,
        heuristic_weight: float = 0.3,
    ):
        self.heuristic = heuristic or HeuristicArbiter()
        self.llm = llm
        self.llm_threshold = llm_threshold
        self.heuristic_weight = heuristic_weight

    def compare(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        heuristic_result = self.heuristic.compare(prompt, response_a, response_b)
        score_diff = abs(heuristic_result.score_a - heuristic_result.score_b)

        if score_diff >= self.llm_threshold or self.llm is None:
            return heuristic_result

        llm_result = self.llm.compare(prompt, response_a, response_b)

        h_total = heuristic_result.score_a + heuristic_result.score_b
        if h_total > 0:
            h_score_a = heuristic_result.score_a / h_total
            h_score_b = heuristic_result.score_b / h_total
        else:
            h_score_a = h_score_b = 0.5

        w = self.heuristic_weight
        final_score_a = w * h_score_a + (1 - w) * llm_result.score_a
        final_score_b = w * h_score_b + (1 - w) * llm_result.score_b

        if abs(final_score_a - final_score_b) < 0.1:
            winner = "tie"
        elif final_score_a > final_score_b:
            winner = "a"
        else:
            winner = "b"

        return ArbiterResult(
            winner=winner,
            score_a=final_score_a * 100,
            score_b=final_score_b * 100,
            reason=f"Hybrid: heuristic={heuristic_result.winner}, llm={llm_result.winner}",
            confidence=(heuristic_result.confidence + llm_result.confidence) / 2,
        )


def create_arbiter(
    arbiter_type: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    margin: float = 2.0,
) -> BaseArbiter:
    """Factory for heuristic, llm, or hybrid arbiters."""
    arbiter_type = arbiter_type.lower()

    if arbiter_type == "heuristic":
        return HeuristicArbiter(margin=margin)

    if arbiter_type == "llm":
        return LLMArbiter(api_key=api_key, base_url=base_url, model=model)

    if arbiter_type == "hybrid":
        heuristic = HeuristicArbiter(margin=margin)
        llm = LLMArbiter(api_key=api_key, base_url=base_url, model=model)
        return HybridArbiter(heuristic=heuristic, llm=llm)

    raise ValueError(
        f"Unknown arbiter type: {arbiter_type}. Use 'heuristic', 'llm', or 'hybrid'."
    )
