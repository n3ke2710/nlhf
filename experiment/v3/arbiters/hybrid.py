"""
Hybrid arbiter - комбинация эвристик и LLM.
"""

from typing import Optional

from .base import BaseArbiter, ArbiterResult
from .heuristic import HeuristicArbiter
from .llm import LLMArbiter


class HybridArbiter(BaseArbiter):
    """
    Гибридный арбитр: эвристика + LLM.
    
    Стратегия:
    1. Сначала применяет эвристику
    2. Если разница скоров мала (неуверенное решение), вызывает LLM
    3. Комбинирует результаты
    """
    
    def __init__(
        self,
        heuristic: Optional[HeuristicArbiter] = None,
        llm: Optional[LLMArbiter] = None,
        llm_threshold: float = 5.0,  # Когда вызывать LLM
        heuristic_weight: float = 0.3,  # Вес эвристики в финальном скоре
    ):
        self.heuristic = heuristic or HeuristicArbiter()
        self.llm = llm
        self.llm_threshold = llm_threshold
        self.heuristic_weight = heuristic_weight
    
    def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> ArbiterResult:
        """Сравнить гибридным методом."""
        
        # 1. Heuristic evaluation
        heuristic_result = self.heuristic.compare(prompt, response_a, response_b)
        
        score_diff = abs(heuristic_result.score_a - heuristic_result.score_b)
        
        # 2. If confident enough, use heuristic only
        if score_diff >= self.llm_threshold or self.llm is None:
            return heuristic_result
        
        # 3. Call LLM for uncertain cases
        llm_result = self.llm.compare(prompt, response_a, response_b)
        
        # 4. Combine results
        # Normalize heuristic scores to 0-1
        h_total = heuristic_result.score_a + heuristic_result.score_b
        if h_total > 0:
            h_score_a = heuristic_result.score_a / h_total
            h_score_b = heuristic_result.score_b / h_total
        else:
            h_score_a = h_score_b = 0.5
        
        # Weighted combination
        w = self.heuristic_weight
        final_score_a = w * h_score_a + (1 - w) * llm_result.score_a
        final_score_b = w * h_score_b + (1 - w) * llm_result.score_b
        
        # Determine winner
        if abs(final_score_a - final_score_b) < 0.1:
            winner = "tie"
        elif final_score_a > final_score_b:
            winner = "a"
        else:
            winner = "b"
        
        return ArbiterResult(
            winner=winner,
            score_a=final_score_a * 100,  # Scale back to 0-100
            score_b=final_score_b * 100,
            reason=f"Hybrid: heuristic={heuristic_result.winner}, llm={llm_result.winner}",
            confidence=(heuristic_result.confidence + llm_result.confidence) / 2
        )


def create_arbiter(
    arbiter_type: str,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    margin: float = 2.0
) -> BaseArbiter:
    """
    Фабричная функция для создания арбитра.
    
    Args:
        arbiter_type: "heuristic", "llm", или "hybrid"
        api_key: API ключ для LLM
        base_url: Base URL для API
        model: Модель для LLM арбитра
        margin: Margin для heuristic арбитра
    
    Returns:
        Экземпляр арбитра
    """
    arbiter_type = arbiter_type.lower()
    
    if arbiter_type == "heuristic":
        return HeuristicArbiter(margin=margin)
    
    elif arbiter_type == "llm":
        return LLMArbiter(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
    
    elif arbiter_type == "hybrid":
        heuristic = HeuristicArbiter(margin=margin)
        llm = LLMArbiter(
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        return HybridArbiter(heuristic=heuristic, llm=llm)
    
    else:
        raise ValueError(
            f"Unknown arbiter type: {arbiter_type}. "
            f"Use 'heuristic', 'llm', or 'hybrid'."
        )
