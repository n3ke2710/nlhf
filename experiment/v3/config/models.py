"""
Models configuration - загрузка моделей из YAML.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path

import yaml

from models.base import ModelConfig, ModelRegistry, BaseModel


@dataclass
class ModelsConfig:
    """Конфигурация набора моделей."""
    models: List[ModelConfig] = field(default_factory=list)
    
    # Default API settings (can be overridden per model)
    default_api_key: Optional[str] = None
    default_base_url: Optional[str] = None
    default_max_tokens: int = 100
    default_temperature: float = 0.7
    
    def get_model_names(self) -> List[str]:
        """Список имён моделей."""
        return [m.name for m in self.models]
    
    def create_models(self) -> Dict[str, BaseModel]:
        """Создать все модели."""
        return ModelRegistry.create_multiple(self.models)


def load_models_from_yaml(yaml_path: str) -> ModelsConfig:
    """
    Загрузить конфигурацию моделей из YAML файла.
    
    Example YAML:
    ```yaml
    defaults:
      api_key: ${OPENAI_API_KEY}
      base_url: https://api.openai.com/v1
      max_tokens: 100
      temperature: 0.7
    
    models:
      - name: gpt-4o-mini
        type: openai
        model_id: gpt-4o-mini
      
      - name: gpt-3.5-turbo
        type: openai
        model_id: gpt-3.5-turbo
      
      - name: llama-local
        type: local
        model_path: ~/.tune/models/Llama-3.2-3B/
        quantization: 4bit
    ```
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Defaults
    defaults = data.get('defaults', {})
    
    # Expand environment variables
    def expand_env(value):
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            env_var = value[2:-1]
            return os.environ.get(env_var, value)
        return value
    
    default_api_key = expand_env(defaults.get('api_key'))
    default_base_url = defaults.get('base_url', 'https://api.openai.com/v1')
    default_max_tokens = defaults.get('max_tokens', 100)
    default_temperature = defaults.get('temperature', 0.7)
    
    # Parse models
    model_configs = []
    for m in data.get('models', []):
        config = ModelConfig(
            name=m['name'],
            model_type=m.get('type', 'api'),
            model_id=m.get('model_id', m['name']),
            api_key=expand_env(m.get('api_key')) or default_api_key,
            base_url=m.get('base_url') or default_base_url,
            model_path=m.get('model_path'),
            base_model_path=m.get('base_model_path'),
            is_peft=m.get('is_peft', False),
            quantization=m.get('quantization'),
            max_tokens=m.get('max_tokens', default_max_tokens),
            temperature=m.get('temperature', default_temperature),
            strength=m.get('strength', 0.5),
            style=m.get('style', 'balanced')
        )
        model_configs.append(config)
    
    return ModelsConfig(
        models=model_configs,
        default_api_key=default_api_key,
        default_base_url=default_base_url,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature
    )


# Пример YAML конфига
EXAMPLE_MODELS_YAML = """
# Models configuration for NLHF Tournament
# 
# Supported types:
#   - openai: OpenAI API models (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)
#   - anthropic: Anthropic Claude models
#   - deepinfra: DeepInfra hosted models (Llama, Mistral, etc.)
#   - local: Local HuggingFace models
#   - lora: LoRA fine-tuned models
#   - mock: Mock models for testing

defaults:
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  max_tokens: 100
  temperature: 0.7

models:
  # OpenAI models
  - name: gpt-4o-mini
    type: openai
    model_id: gpt-4o-mini
  
  - name: gpt-3.5-turbo
    type: openai
    model_id: gpt-3.5-turbo
  
  # Uncomment to add more models:
  #
  # - name: gpt-4o
  #   type: openai
  #   model_id: gpt-4o
  #
  # - name: llama-70b
  #   type: deepinfra
  #   model_id: meta-llama/Llama-3.3-70B-Instruct-Turbo
  #   api_key: ${DEEPINFRA_API_KEY}
  #   base_url: https://api.deepinfra.com/v1/openai
  #
  # - name: llama-local
  #   type: local
  #   model_path: ~/.tune/models/Llama-3.2-3B/
  #   quantization: 4bit
  #
  # - name: my-lora
  #   type: lora
  #   base_model_path: ~/.tune/models/Llama-3.2-3B-Base/
  #   model_path: ~/.tune/checkpoints/my-lora/
  #   quantization: 4bit
"""


def create_example_models_yaml(output_path: str = "models.yaml"):
    """Создать пример models.yaml."""
    with open(output_path, 'w') as f:
        f.write(EXAMPLE_MODELS_YAML)
    print(f"Created example config: {output_path}")
