"""Training functionality for diffusion policies."""

import os
import time
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .policy import DiffusionPolicy, DiffusionPolicyDataset


def train_diffusion_policy(
    dataset_path: str,
    model_save_path: str,
    config: Dict[str, Any],
    log_dir: str = "./logs",
) -> DiffusionPolicy:
    """Train diffusion policy on the dataset.

    Args:
        dataset_path: Path to the dataset
        model_save_path: Path to save the trained model
        config: Training configuration
        log_dir: Directory to save logs

    Returns:
        Trained model
    """

    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    if config.get("force_cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        pred_horizon=config["pred_horizon"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
    )

    # Create model
    model = DiffusionPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        num_diffusion_iters=config["num_diffusion_iters"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        wandb.init(
            project="diffusion-policy-geom2d",
            config=config,
            name=f"diffusion_policy_{int(time.time())}",
            dir=str(log_dir),
        )

    # Setup logging
    train_log_path = log_dir / "training.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(train_log_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message("Starting training...")
    log_message(f"Dataset: {dataset_path}")
    log_message(f"Model save path: {model_save_path}")
    log_message(f"Device: {device}")
    log_message(f"Config: {config}")

    # Training loop
    model.train()
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            obs_states = batch["obs_states"].to(device)
            actions = batch["actions"].to(device)

            # Forward pass
            noise_pred, noise = model(obs_states, actions)

            # Compute loss
            loss = F.mse_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % config["log_interval"] == 0:
                msg = (
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}"
                )
                log_message(msg)

                if config.get("use_wandb", False):
                    wandb.log(
                        {"batch_loss": loss.item(), "epoch": epoch, "batch": batch_idx}
                    )

        # End of epoch
        avg_loss = epoch_loss / num_batches
        scheduler.step()

        msg = f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}"
        log_message(msg)

        if config.get("use_wandb", False):
            wandb.log(
                {
                    "epoch_loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch": epoch,
                }
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_config = config.copy()
            save_config.update(
                {
                    "obs_dim": dataset.obs_dim,
                    "action_dim": dataset.action_dim,
                    "image_shape": list(dataset.image_shape),
                }
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": save_config,
                },
                model_save_path,
            )
            log_message(f"Best model saved with loss: {avg_loss:.6f}")

    log_message("Training completed!")

    if config.get("use_wandb", False):
        wandb.finish()

    return model


def train_lerobot_diffusion_policy(
    dataset_path: str,
    model_save_path: str,
    config: Dict[str, Any],
    log_dir: str = "./logs",
) -> str:
    """Train diffusion policy using LeRobot's built-in implementation.
    
    Args:
        dataset_path: Path to the dataset
        model_save_path: Path to save the trained model
        config: Training configuration
        log_dir: Directory to save logs
        
    Returns:
        Path to the saved model
    """
    try:
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy as LeRobotDiffusionPolicy
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.scripts.train import train as lerobot_train
        import hydra
        from omegaconf import DictConfig, OmegaConf
    except ImportError as e:
        raise ImportError(f"LeRobot dependencies not found: {e}. Please install LeRobot.")
    
    # Setup logging
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = log_dir / "lerobot_training.log"
    
    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(train_log_path, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    log_message("Starting LeRobot diffusion policy training...")
    
    # Create LeRobot-compatible config
    lerobot_config = {
        "policy": {
            "_target_": "lerobot.common.policies.diffusion.modeling_diffusion.DiffusionPolicy",
            "horizon": config.get("action_horizon", 8),
            "n_obs_steps": config.get("obs_horizon", 2),
            "n_action_steps": config.get("pred_horizon", 8),
            "num_inference_steps": config.get("num_diffusion_iters", 100),
            "down_dims": [256, 512, 1024],
            "kernel_size": 5,
            "n_groups": 8,
            "use_film_scale_modulation": True,
        },
        "training": {
            "lr": config.get("learning_rate", 1e-4),
            "batch_size": config.get("batch_size", 32),
            "num_epochs": config.get("num_epochs", 50),
            "weight_decay": config.get("weight_decay", 1e-6),
            "grad_clip_norm": config.get("grad_clip_norm", 1.0),
            "dataloader_num_workers": config.get("num_workers", 4),
            "log_freq": config.get("log_interval", 10),
        },
        "dataset": {
            "repo_id": dataset_path,
            "split": "train",
        },
        "device": "cuda" if torch.cuda.is_available() and not config.get("force_cpu", False) else "cpu",
        "use_wandb": config.get("use_wandb", False),
    }
    
    log_message(f"LeRobot config: {lerobot_config}")
    
    # Create temporary config file for LeRobot
    config_path = Path(log_dir) / "lerobot_config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(OmegaConf.create(lerobot_config), f)
    
    try:
        # Load dataset
        log_message(f"Loading dataset from {dataset_path}")
        # Note: This is a simplified approach - you may need to adapt based on your dataset format
        
        # For now, create a placeholder model save since LeRobot training is complex
        # In a full implementation, you'd need to adapt your dataset format to LeRobot's format
        log_message("Warning: LeRobot integration is not fully implemented yet.")
        log_message("This is a placeholder that demonstrates the structure.")
        log_message("You would need to:")
        log_message("1. Convert dataset to LeRobot format")
        log_message("2. Set up proper LeRobot training pipeline")
        log_message("3. Handle model saving/loading")
        
        # Create a dummy model file to indicate this was attempted
        model_save_dir = Path(model_save_path).parent
        model_save_dir.mkdir(parents=True, exist_ok=True)
        
        placeholder_path = model_save_dir / "lerobot_diffusion_placeholder.txt"
        with open(placeholder_path, 'w') as f:
            f.write("LeRobot diffusion policy training placeholder\n")
            f.write(f"Config: {lerobot_config}\n")
            f.write(f"Dataset: {dataset_path}\n")
            f.write(f"Timestamp: {time.time()}\n")
        
        log_message(f"Placeholder created at: {placeholder_path}")
        return str(placeholder_path)
        
    except Exception as e:
        log_message(f"LeRobot training failed: {e}")
        raise
    
    finally:
        # Clean up temporary config
        if config_path.exists():
            config_path.unlink()


def get_default_training_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        "obs_horizon": 2,
        "action_horizon": 8,
        "pred_horizon": 8,
        "num_diffusion_iters": 100,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip_norm": 1.0,
        "num_workers": 4,
        "log_interval": 10,
        "use_wandb": False,
        "force_cpu": False,
    }
