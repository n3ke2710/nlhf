"""
Base model interface and registry.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    name: str
    model_type: str  # "api", "local", "mock"
    
    # API settings
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_id: Optional[str] = None  # e.g., "gpt-4o-mini"
    
    # Local settings
    model_path: Optional[str] = None
    base_model_path: Optional[str] = None
    is_peft: bool = False
    quantization: Optional[str] = None  # "4bit", "8bit", None
    
    # Generation settings
    max_tokens: int = 100
    temperature: float = 0.7
    
    # Mock settings (for testing)
    strength: float = 0.5
    style: str = "balanced"


class BaseModel(ABC):
    """Базовый класс для всех моделей."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.name = config.name
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Сгенерировать ответ на промпт."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Проверить доступность модели."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class ModelRegistry:
    """Реестр моделей - фабрика для создания моделей по конфигу."""
    
    _model_types: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_type: str):
        """Декоратор для регистрации типа модели."""
        def decorator(model_class: Type[BaseModel]):
            cls._model_types[model_type] = model_class
            return model_class
        return decorator
    
    @classmethod
    def create(cls, config: ModelConfig) -> BaseModel:
        """Создать модель по конфигу."""
        model_type = config.model_type
        
        if model_type not in cls._model_types:
            available = list(cls._model_types.keys())
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {available}"
            )
        
        model_class = cls._model_types[model_type]
        return model_class(config)
    
    @classmethod
    def create_multiple(cls, configs: List[ModelConfig]) -> Dict[str, BaseModel]:
        """Создать несколько моделей."""
        return {cfg.name: cls.create(cfg) for cfg in configs}
    
    @classmethod
    def available_types(cls) -> List[str]:
        """Список доступных типов моделей."""
        return list(cls._model_types.keys())


def create_response_fn(models: Dict[str, BaseModel]):
    """Создать функцию get_response для турнира."""
    def get_response(model_name: str, prompt: str) -> str:
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
        return models[model_name].generate(prompt)
    return get_response
