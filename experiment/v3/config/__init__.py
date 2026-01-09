"""
Configuration module - конфиги для экспериментов.
"""

from .experiment import ExperimentConfig
from .models import ModelsConfig, load_models_from_yaml

__all__ = [
    "ExperimentConfig",
    "ModelsConfig", 
    "load_models_from_yaml",
]
