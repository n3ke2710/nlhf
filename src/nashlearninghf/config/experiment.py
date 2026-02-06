"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    """Experiment configuration for tournament runs."""

    name: str = "nlhf_experiment"
    seed: int = 42

    num_prompts: int = 50
    matches_per_pair: int = 2
    alpha: float = 1.0

    dataset_name: str = "trl-lib/tldr"
    dataset_split: str = "validation[:100]"

    models_config_path: Optional[str] = None
    model_names: List[str] = field(default_factory=list)

    arbiter_type: str = "heuristic"
    arbiter_margin: float = 2.0

    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    output_dir: str = "./output"
    log_dir: str = "./output/logs"
    graphics_dir: str = "./output/graphics"

    def __post_init__(self):
        import os

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.graphics_dir, exist_ok=True)
