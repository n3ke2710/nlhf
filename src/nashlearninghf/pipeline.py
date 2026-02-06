"""High-level experiment orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .arbiters.base import BaseArbiter
from .models.base import BaseModel, create_response_fn
from .tournament import Tournament, TournamentStats


class PromptProvider(ABC):
    """Abstract provider for prompts."""

    @abstractmethod
    def get_prompts(self) -> List[str]:
        raise NotImplementedError  # pragma: no cover


@dataclass
class InMemoryPromptProvider(PromptProvider):
    """Simple in-memory prompt provider."""

    prompts: List[str]

    def get_prompts(self) -> List[str]:
        return list(self.prompts)


class TournamentExperiment:
    """OOP wrapper for running a tournament with provided models and arbiter."""

    def __init__(
        self,
        models: Dict[str, BaseModel],
        arbiter: BaseArbiter,
        prompt_provider: PromptProvider,
        matches_per_pair: int = 1,
        random_pairing: bool = True,
        alpha: float = 1.0,
        log_dir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        self.models = models
        self.arbiter = arbiter
        self.prompt_provider = prompt_provider
        self.matches_per_pair = matches_per_pair
        self.random_pairing = random_pairing
        self.alpha = alpha
        self.log_dir = log_dir
        self.seed = seed

    def run(self) -> TournamentStats:
        rng = None
        if self.seed is not None:
            import random

            rng = random.Random(self.seed)

        tournament = Tournament(
            model_names=list(self.models.keys()),
            arbiter=self.arbiter,
            alpha=self.alpha,
            log_dir=self.log_dir,
            rng=rng,
        )

        prompts = self.prompt_provider.get_prompts()
        get_response = create_response_fn(self.models)

        return tournament.run_tournament(
            prompts=prompts,
            get_response=get_response,
            matches_per_pair=self.matches_per_pair,
            random_pairing=self.random_pairing,
        )
