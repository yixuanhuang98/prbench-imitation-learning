"""Prbench Imitation Learning Package."""

from .policy import DiffusionPolicy, DiffusionPolicyDataset, ConditionalUNet1D
from .train import train_diffusion_policy, get_default_training_config
from .evaluate import PolicyEvaluator, evaluate_policy
from .data_generation import (
    generate_lerobot_dataset, 
    setup_environment, 
    get_available_environments
)

__all__ = [
    # Policy models
    "DiffusionPolicy",
    "DiffusionPolicyDataset", 
    "ConditionalUNet1D",
    
    # Training
    "train_diffusion_policy",
    "get_default_training_config",
    
    # Evaluation
    "PolicyEvaluator",
    "evaluate_policy",
    
    # Data generation
    "generate_lerobot_dataset",
    "setup_environment",
    "get_available_environments",
]
