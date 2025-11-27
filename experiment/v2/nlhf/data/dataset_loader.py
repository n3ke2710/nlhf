"""
Dataset Loader Module
=====================

Functions for loading and preprocessing datasets.
"""

from datasets import load_dataset
from typing import List, Dict, Any
import logging

logger = logging.getLogger('nlhf.data')


def load_sft_dataset(
    dataset_name: str = "trl-lib/tldr",
    split: str = "train[:500]",
    max_length: int = 512
) -> Any:
    """
    Load dataset for Supervised Fine-Tuning.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split specification
        max_length: Maximum sequence length
    
    Returns:
        Dataset object
    """
    logger.info(f"Loading SFT dataset: {dataset_name}, split={split}")
    
    dataset = load_dataset(dataset_name, split=split)
    
    logger.info(f"Loaded {len(dataset)} examples for SFT")
    
    return dataset


def load_validation_prompts(
    dataset_name: str = "trl-lib/tldr",
    split: str = "validation[:100]"
) -> List[str]:
    """
    Load validation prompts for policy evaluation.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split specification
    
    Returns:
        List of prompt strings
    """
    logger.info(f"Loading validation prompts: {dataset_name}, split={split}")
    
    dataset = load_dataset(dataset_name, split=split)
    prompts = [item['prompt'] for item in dataset]
    
    logger.info(f"Loaded {len(prompts)} validation prompts")
    
    return prompts


def load_real_preferences(
    dataset_name: str = "openai/summarize_from_feedback",
    config: str = "comparisons",
    max_samples: int = 2000
) -> List[Dict[str, str]]:
    """
    Load real preference pairs from OpenAI TL;DR dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        config: Dataset configuration
        max_samples: Maximum number of samples to load
    
    Returns:
        List of preference pairs with keys: prompt, chosen, rejected
    """
    logger.info(f"Loading real preference dataset: {dataset_name}/{config}")
    logger.info(f"Max samples: {max_samples}")
    
    dataset = load_dataset(
        dataset_name,
        config,
        split=f"train[:{max_samples}]"
    )
    
    logger.info(f"Loaded {len(dataset)} examples from OpenAI dataset")
    
    # Convert to preference pairs format
    preference_pairs = []
    
    for item in dataset:
        try:
            # Extract information
            post = item['info']['post']
            title = item['info'].get('title', '')
            subreddit = item['info'].get('subreddit', 'reddit')
            
            # Format prompt
            prompt = f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"
            
            # Extract chosen and rejected summaries
            summaries = item['summaries']
            choice = item['choice']
            
            chosen = summaries[choice]['text'].strip()
            rejected = summaries[1 - choice]['text'].strip()
            
            # Validate
            if len(chosen) > 0 and len(rejected) > 0 and chosen != rejected:
                preference_pairs.append({
                    'prompt': prompt,
                    'chosen': chosen,
                    'rejected': rejected
                })
        except Exception as e:
            logger.warning(f"Skipping problematic example: {e}")
            continue
    
    logger.info(f"Created {len(preference_pairs)} valid preference pairs")
    logger.info(f"Filtered {len(dataset) - len(preference_pairs)} problematic examples")
    
    return preference_pairs
