"""
Configuration Module for NLHF
=============================

Centralized configuration for all NLHF components.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class NLHFConfig:
    """Main configuration for NLHF training pipeline"""
    
    # ===== Model Configuration =====
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_revision: str = "main"
    trust_remote_code: bool = True
    
    # ===== Data Configuration =====
    dataset_name: str = "trl-lib/tldr"
    dataset_split_train: str = "train[:500]"
    dataset_split_validation: str = "validation[:100]"
    max_length: int = 512
    
    # Real dataset option for reward model
    use_real_dataset: bool = True
    real_dataset_name: str = "openai/summarize_from_feedback"
    real_dataset_config: str = "comparisons"
    real_dataset_size: int = 2000
    
    # ===== SFT Configuration =====
    sft_output_dir: str = "qwen2.5-3b-tldr-lora"
    sft_max_steps: int = 500
    sft_learning_rate: float = 2e-4
    sft_per_device_train_batch_size: int = 4
    sft_gradient_accumulation_steps: int = 4
    sft_warmup_steps: int = 50
    sft_logging_steps: int = 25
    sft_save_steps: int = 500
    
    # LoRA for SFT
    sft_lora_r: int = 32
    sft_lora_alpha: int = 64
    sft_lora_dropout: float = 0.05
    sft_lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # ===== Policy Generation Configuration =====
    num_policies: int = 10  # N in paper
    policy_noise_scale: float = 0.03
    generation_temperature: float = 0.8
    generation_max_new_tokens: int = 64
    generation_batch_size: int = 8
    
    # ===== Preference Pairs Configuration =====
    pairs_per_prompt: int = 15
    use_improved_scoring: bool = True  # Use multi-factor heuristic
    
    # ===== Reward Model Configuration =====
    rm_output_dir: str = "nlhf_reward_model"
    rm_max_steps: int = 400  # Adaptive: 400 for synthetic, 500 for real
    rm_learning_rate: float = 2e-5
    rm_per_device_train_batch_size: int = 16
    rm_gradient_accumulation_steps: int = 4
    rm_warmup_steps: int = 40
    rm_max_grad_norm: float = 1.0
    rm_logging_steps: int = 50
    rm_save_steps: int = 400
    
    # LoRA for Reward Model
    rm_lora_r: int = 16
    rm_lora_alpha: int = 32
    rm_lora_dropout: float = 0.05
    rm_lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    
    # NLHF specific
    kl_tau: float = 0.1  # KL regularization strength
    
    # ===== Compute Configuration =====
    device: str = "cuda:0"
    torch_dtype: str = "bfloat16"
    use_fp16: bool = False
    use_bf16: bool = True
    
    # ===== Logging Configuration =====
    log_root: str = "logs"
    experiment_name: Optional[str] = None
    log_level: str = "INFO"
    
    # ===== Paths =====
    @property
    def exp_dir(self) -> str:
        """Get experiment directory path"""
        from datetime import datetime
        if self.experiment_name:
            name = self.experiment_name
        else:
            ts = datetime.now().strftime('%Y%m%d-%H%M%S')
            name = f'exp_{ts}_run'
        return os.path.join(self.log_root, name)
    
    @property
    def vis_dir(self) -> str:
        """Get visualization directory path"""
        return os.path.join(self.exp_dir, 'visualizations')
    
    @property
    def log_file(self) -> str:
        """Get log file path"""
        return os.path.join(self.exp_dir, 'experiment.log')
    
    def __post_init__(self):
        """Validate and adjust configuration after initialization"""
        # Adjust RM steps based on dataset type
        if self.use_real_dataset and self.real_dataset_size >= 1500:
            self.rm_max_steps = 500
        
        # Create directories
        os.makedirs(self.log_root, exist_ok=True)


# Preset configurations
def get_quick_config() -> NLHFConfig:
    """Quick configuration for fast experiments (~25-30 min)"""
    config = NLHFConfig()
    config.num_policies = 5
    config.pairs_per_prompt = 10
    config.real_dataset_size = 1000
    config.rm_max_steps = 300
    config.sft_max_steps = 300
    return config


def get_quality_config() -> NLHFConfig:
    """High-quality configuration for best results (~50-60 min)"""
    config = NLHFConfig()
    config.num_policies = 10
    config.pairs_per_prompt = 15
    config.real_dataset_size = 2000
    config.rm_max_steps = 500
    config.use_real_dataset = True
    return config


def get_demo_config() -> NLHFConfig:
    """Demo configuration for understanding the concept"""
    config = NLHFConfig()
    config.num_policies = 5
    config.pairs_per_prompt = 5
    config.use_real_dataset = False
    config.dataset_split_train = "train[:100]"
    config.dataset_split_validation = "validation[:20]"
    config.sft_max_steps = 200
    config.rm_max_steps = 200
    return config
