"""
API-based models (OpenAI, Anthropic, DeepInfra, etc.)
"""

import os
from typing import Optional

from .base import BaseModel, ModelConfig, ModelRegistry


@ModelRegistry.register("api")
class APIModel(BaseModel):
    """Универсальная модель через OpenAI-совместимый API."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        from openai import OpenAI
        
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = config.base_url or "https://api.openai.com/v1"
        self._model_id = config.model_id or config.name
        
        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url
        )
    
    def generate(self, prompt: str) -> str:
        """Сгенерировать TL;DR через API."""
        try:
            response = self._client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant that summarizes Reddit posts. Provide concise TL;DR summaries."
                    },
                    {
                        "role": "user", 
                        "content": f"{prompt}\n\nTL;DR:"
                    }
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error for {self.name}: {e}")
            return "[Error generating response]"
    
    def is_available(self) -> bool:
        """Проверить доступность API."""
        try:
            self._client.models.list()
            return True
        except Exception:
            return False


@ModelRegistry.register("openai")
class OpenAIModel(APIModel):
    """OpenAI модель (gpt-4o, gpt-4o-mini, gpt-3.5-turbo)."""
    
    def __init__(self, config: ModelConfig):
        config.base_url = config.base_url or "https://api.openai.com/v1"
        config.api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        super().__init__(config)


@ModelRegistry.register("anthropic")
class AnthropicModel(BaseModel):
    """Anthropic Claude модель."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        
        self._api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model_id = config.model_id or "claude-3-haiku-20240307"
        self._client = Anthropic(api_key=self._api_key)
    
    def generate(self, prompt: str) -> str:
        """Сгенерировать ответ через Anthropic API."""
        try:
            response = self._client.messages.create(
                model=self._model_id,
                max_tokens=self.config.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize this Reddit post concisely:\n\n{prompt}\n\nTL;DR:"
                    }
                ]
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"Anthropic error for {self.name}: {e}")
            return "[Error generating response]"
    
    def is_available(self) -> bool:
        return self._api_key is not None


@ModelRegistry.register("deepinfra")
class DeepInfraModel(APIModel):
    """DeepInfra модель (Llama, Mistral, etc.)."""
    
    def __init__(self, config: ModelConfig):
        config.base_url = config.base_url or "https://api.deepinfra.com/v1/openai"
        config.api_key = config.api_key or os.environ.get("DEEPINFRA_API_KEY")
        super().__init__(config)


# Удобные фабричные функции
def create_openai_models(
    model_names: list[str],
    api_key: Optional[str] = None
) -> dict[str, OpenAIModel]:
    """Создать набор OpenAI моделей."""
    return {
        name: OpenAIModel(ModelConfig(
            name=name,
            model_type="openai",
            model_id=name,
            api_key=api_key
        ))
        for name in model_names
    }


def create_deepinfra_models(
    model_names: list[str],
    api_key: Optional[str] = None
) -> dict[str, DeepInfraModel]:
    """Создать набор DeepInfra моделей."""
    return {
        name: DeepInfraModel(ModelConfig(
            name=name,
            model_type="deepinfra",
            model_id=name,
            api_key=api_key
        ))
        for name in model_names
    }
