"""Prbench Imitation Learning Package."""

from .data_generation import (
    generate_lerobot_dataset,
    get_available_environments,
    setup_environment,
)
from .evaluate import PolicyEvaluator, evaluate_policy
from .policy import ConditionalUNet1D, DiffusionPolicy, DiffusionPolicyDataset
from .train import get_default_training_config, train_diffusion_policy, train_lerobot_diffusion_policy

__all__ = [
    # Policy models
    "DiffusionPolicy",
    "DiffusionPolicyDataset",
    "ConditionalUNet1D",
    # Training
    "train_diffusion_policy",
    "train_lerobot_diffusion_policy",
    "get_default_training_config",
    # Evaluation
    "PolicyEvaluator",
    "evaluate_policy",
    # Data generation
    "generate_lerobot_dataset",
    "setup_environment",
    "get_available_environments",
]
