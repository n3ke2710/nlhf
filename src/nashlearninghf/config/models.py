"""Models configuration loader."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import yaml

from ..models.base import BaseModel, ModelConfig, ModelRegistry


@dataclass
class ModelsConfig:
    """Configuration for a set of models."""

    models: List[ModelConfig] = field(default_factory=list)
    default_api_key: Optional[str] = None
    default_base_url: Optional[str] = None
    default_max_tokens: int = 100
    default_temperature: float = 0.7

    def get_model_names(self) -> List[str]:
        return [m.name for m in self.models]

    def create_models(self) -> Dict[str, BaseModel]:
        return ModelRegistry.create_multiple(self.models)


def load_models_from_yaml(yaml_path: str) -> ModelsConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    defaults = data.get("defaults", {})

    def expand_env(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        return value

    default_api_key = expand_env(defaults.get("api_key"))
    default_base_url = defaults.get("base_url", "https://api.openai.com/v1")
    default_max_tokens = defaults.get("max_tokens", 100)
    default_temperature = defaults.get("temperature", 0.7)

    model_configs = []
    for m in data.get("models", []):
        config = ModelConfig(
            name=m["name"],
            model_type=m.get("type", "api"),
            model_id=m.get("model_id", m["name"]),
            api_key=expand_env(m.get("api_key")) or default_api_key,
            base_url=m.get("base_url") or default_base_url,
            model_path=m.get("model_path"),
            base_model_path=m.get("base_model_path"),
            is_peft=m.get("is_peft", False),
            quantization=m.get("quantization"),
            max_tokens=m.get("max_tokens", default_max_tokens),
            temperature=m.get("temperature", default_temperature),
            strength=m.get("strength", 0.5),
            style=m.get("style", "balanced"),
        )
        model_configs.append(config)

    return ModelsConfig(
        models=model_configs,
        default_api_key=default_api_key,
        default_base_url=default_base_url,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
    )
