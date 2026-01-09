"""
Models module - генераторы ответов для турнира.

Поддерживаемые типы моделей:
- API модели (OpenAI, Anthropic, DeepInfra, etc.)
- Локальные модели (HuggingFace, LoRA)
- Mock модели (для тестирования)
"""

from .base import BaseModel, ModelRegistry
from .api import APIModel, OpenAIModel, AnthropicModel, DeepInfraModel
from .local import LocalModel, LoRAModel
from .mock import MockModel

__all__ = [
    "BaseModel",
    "ModelRegistry",
    "APIModel",
    "OpenAIModel", 
    "AnthropicModel",
    "DeepInfraModel",
    "LocalModel",
    "LoRAModel",
    "MockModel",
]
