"""
Base arbiter interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArbiterResult:
    """Результат сравнения двух ответов."""
    winner: str  # "a", "b", or "tie"
    score_a: float
    score_b: float
    reason: Optional[str] = None
    confidence: float = 1.0


class BaseArbiter(ABC):
    """Базовый класс для арбитров."""
    
    @abstractmethod
    def compare(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> ArbiterResult:
        """
        Сравнить два ответа и определить победителя.
        
        Args:
            prompt: Исходный промпт
            response_a: Ответ модели A
            response_b: Ответ модели B
        
        Returns:
            ArbiterResult с победителем и скорами
        """
        pass
    
    def __call__(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> ArbiterResult:
        """Callable interface."""
        return self.compare(prompt, response_a, response_b)
