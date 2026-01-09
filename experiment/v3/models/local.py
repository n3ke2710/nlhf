"""
Local models (HuggingFace, LoRA, GGUF, etc.)
"""

import os
from typing import Optional

import torch

from .base import BaseModel, ModelConfig, ModelRegistry


@ModelRegistry.register("local")
class LocalModel(BaseModel):
    """Локальная HuggingFace модель."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        model_path = os.path.expanduser(config.model_path)
        
        # Quantization config
        bnb_config = None
        if config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        print(f"Loading local model: {model_path}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if bnb_config is None else None
        )
        
        self._device = next(self._model.parameters()).device
        print(f"  ✓ Loaded on {self._device}")
    
    def generate(self, prompt: str) -> str:
        """Сгенерировать ответ локально."""
        inputs = self._tokenizer(
            prompt + "\nTL;DR:",
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id
            )
        
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract TL;DR part
        if "TL;DR:" in response:
            response = response.split("TL;DR:")[-1].strip()
        
        return response
    
    def is_available(self) -> bool:
        return self._model is not None


@ModelRegistry.register("lora")
class LoRAModel(BaseModel):
    """Модель с LoRA адаптером (PEFT)."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        base_path = os.path.expanduser(config.base_model_path)
        adapter_path = os.path.expanduser(config.model_path)
        
        # Quantization
        bnb_config = None
        if config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        elif config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        print(f"Loading LoRA model: {config.name}")
        print(f"  Base: {base_path}")
        print(f"  Adapter: {adapter_path}")
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(base_path)
        self._tokenizer.pad_token = self._tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self._model = PeftModel.from_pretrained(base_model, adapter_path)
        self._device = next(self._model.parameters()).device
        
        print(f"  ✓ Loaded on {self._device}")
    
    def generate(self, prompt: str) -> str:
        """Сгенерировать ответ."""
        inputs = self._tokenizer(
            prompt + "\nTL;DR:",
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self._device)
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id
            )
        
        response = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "TL;DR:" in response:
            response = response.split("TL;DR:")[-1].strip()
        
        return response
    
    def is_available(self) -> bool:
        return self._model is not None


# Удобные функции для загрузки
def load_local_model(
    name: str,
    model_path: str,
    quantization: str = "4bit"
) -> LocalModel:
    """Загрузить локальную модель."""
    return LocalModel(ModelConfig(
        name=name,
        model_type="local",
        model_path=model_path,
        quantization=quantization
    ))


def load_lora_model(
    name: str,
    base_model_path: str,
    adapter_path: str,
    quantization: str = "4bit"
) -> LoRAModel:
    """Загрузить LoRA модель."""
    return LoRAModel(ModelConfig(
        name=name,
        model_type="lora",
        base_model_path=base_model_path,
        model_path=adapter_path,
        is_peft=True,
        quantization=quantization
    ))
