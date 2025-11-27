"""
Data Module for NLHF
====================

Handles dataset loading, preprocessing, and preference pair creation.
"""

from .dataset_loader import load_sft_dataset, load_validation_prompts, load_real_preferences
from .preference_scoring import score_summary, create_preference_pairs
from .preference_dataset import PreferenceDataset, create_collate_fn

__all__ = [
    "load_sft_dataset",
    "load_validation_prompts",
    "load_real_preferences",
    "score_summary",
    "create_preference_pairs",
    "PreferenceDataset",
    "create_collate_fn",
]
