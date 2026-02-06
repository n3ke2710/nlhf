"""Base arbiter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ArbiterResult:
    """Result of comparing two responses."""

    winner: str  # "a", "b", or "tie"
    score_a: float
    score_b: float
    reason: Optional[str] = None
    confidence: float = 1.0


class BaseArbiter(ABC):
    """Base class for arbiters."""

    @abstractmethod
    def compare(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        """Compare two responses and return the result."""
        raise NotImplementedError  # pragma: no cover

    def __call__(self, prompt: str, response_a: str, response_b: str) -> ArbiterResult:
        return self.compare(prompt, response_a, response_b)
