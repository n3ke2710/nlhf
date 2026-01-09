"""
Experiment configuration.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ExperimentConfig:
    """Конфигурация эксперимента."""
    
    # Basic
    name: str = "nlhf_experiment"
    seed: int = 42
    
    # Tournament
    num_prompts: int = 50
    matches_per_pair: int = 2
    alpha: float = 1.0  # Softmax temperature for AlphaRank
    
    # Data
    dataset_name: str = "trl-lib/tldr"
    dataset_split: str = "validation[:100]"
    
    # Models (see config/models.py)
    models_config_path: Optional[str] = None  # Path to models.yaml
    model_names: List[str] = field(default_factory=list)  # Override
    
    # Arbiter
    arbiter_type: str = "heuristic"  # "heuristic", "llm", "hybrid"
    arbiter_margin: float = 2.0
    
    # LLM Arbiter settings
    llm_api_key: Optional[str] = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    
    # Paths
    output_dir: str = "./output"
    log_dir: str = "./output/logs"
    graphics_dir: str = "./output/graphics"
    
    def __post_init__(self):
        """Validate and setup config."""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.graphics_dir, exist_ok=True)
