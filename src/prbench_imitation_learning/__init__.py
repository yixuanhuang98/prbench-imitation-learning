"""Prbench Imitation Learning Package."""

from .data_generation import (
    generate_lerobot_dataset,
    get_available_environments,
    setup_environment,
)
from .evaluate import PolicyEvaluator, evaluate_policy
from .policy import (
    BehaviorCloningPolicy,
    ConditionalUNet1D,
    DiffusionPolicy,
    DiffusionPolicyDataset,
)
from .train import (
    get_default_training_config,
    train_act_policy,
    train_behavior_cloning_policy,
    train_diffusion_policy,
    train_lerobot_diffusion_policy,
)
from .act_dataset import ACTDataset, collate_act_batch

__all__ = [
    # Policy models
    "DiffusionPolicy",
    "DiffusionPolicyDataset",
    "BehaviorCloningPolicy",
    "ConditionalUNet1D",
    # ACT specific
    "ACTDataset",
    "collate_act_batch",
    # Training
    "train_diffusion_policy",
    "train_lerobot_diffusion_policy",
    "train_behavior_cloning_policy",
    "train_act_policy",
    "get_default_training_config",
    # Evaluation
    "PolicyEvaluator",
    "evaluate_policy",
    # Data generation
    "generate_lerobot_dataset",
    "setup_environment",
    "get_available_environments",
]
