"""Training functionality for diffusion policies."""

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .policy import BehaviorCloningPolicy, DiffusionPolicy, DiffusionPolicyDataset

# LeRobot imports
try:
    from lerobot.configs.policies import FeatureType, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    # isort: off
    from lerobot.policies.diffusion.modeling_diffusion import (
        DiffusionPolicy as LeRobotDiffusionPolicy,
    )

    # isort: on

    # Import GradScaler inside try block since it's only used with LeRobot
    # pylint: disable=ungrouped-imports
    from torch.amp import GradScaler

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


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
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

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
        video_backend=config.get("video_backend", "pyav"),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=config["num_workers"] > 0,
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
            dir=str(log_dir_path),
        )

    # Setup logging
    train_log_path = log_dir_path / "training.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(train_log_path, "a", encoding="utf-8") as f:
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
    use_original_lerobot_params: bool = True,
) -> LeRobotDiffusionPolicy:
    """Train LeRobot diffusion policy on the dataset.

    Args:
        dataset_path: Path to the dataset
        model_save_path: Path to save the trained model
        config: Training configuration
        log_dir: Directory to save logs

    Returns:
        Trained model
    """
    if not LEROBOT_AVAILABLE:
        raise ImportError(
            "LeRobot is not available. Please install it with: pip install lerobot"
        )

    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Set device
    if config.get("force_cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset - use our custom dataset wrapper for LeRobot
    dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        obs_horizon=config.get("obs_horizon", 2),
        action_horizon=config.get("action_horizon", 8),
        pred_horizon=config.get("pred_horizon", 8),
        video_backend=config.get("video_backend", "pyav"),
    )
    print(f"Dataset loaded with {len(dataset)} sequences")

    # Get dimensions from our dataset
    obs_state_shape = (dataset.obs_dim,)
    action_shape = (dataset.action_dim,)
    image_shape = dataset.image_shape

    print(f"Observation state shape: {obs_state_shape}")
    print(f"Action shape: {action_shape}")
    if image_shape is not None:
        print(f"Image shape: {image_shape}")

    # Create LeRobot diffusion config with proper input/output features
    # Use actual robot state from observations (obs_dim), environment state empty
    input_features = {
        "observation.state": PolicyFeature(shape=list(obs_state_shape), type=FeatureType.STATE),
        "observation.environment_state": PolicyFeature(
            shape=[0], type=FeatureType.ENV
        ),
    }
    if image_shape is not None:
        # Dataset already provides images in (C, H, W) format
        # Convert torch.Size to list if needed
        lerobot_image_shape = list(image_shape) if hasattr(image_shape, '__iter__') else [image_shape]
        input_features["observation.image"] = PolicyFeature(
            shape=lerobot_image_shape, type=FeatureType.VISUAL)

    output_features = {
        "action": PolicyFeature(shape=list(action_shape), type=FeatureType.ACTION),
    }

    # Create config and set required features
    diffusion_config = DiffusionConfig()
    diffusion_config.input_features = input_features
    diffusion_config.output_features = output_features
    diffusion_config.n_obs_steps = config.get("obs_horizon", 2)
    diffusion_config.horizon = config.get("action_horizon", 8)
    diffusion_config.n_action_steps = config.get("pred_horizon", 8)
    diffusion_config.num_train_timesteps = config.get("num_diffusion_iters", 100)
    diffusion_config.optimizer_lr = config.get("learning_rate", 1e-4)
    # Configure timestep embedding dimension for conditioning
    # For PushT: global_cond = obs_state_dim * n_obs_steps = 2 * 2 = 4
    # Total cond_dim = timestep_embed + global_cond
    diffusion_config.diffusion_step_embed_dim = 128

    # Disable cropping since we're not using images for now
    diffusion_config.crop_shape = None

    # Compute dataset statistics for normalization
    print("Computing dataset statistics for normalization...")

    stats = {}

    # Compute state statistics
    all_states = []
    all_actions = []
    for i in range(len(dataset)):
        sample = dataset[i]
        all_states.extend(sample["obs_states"].flatten().tolist())
        all_actions.extend(sample["actions"].flatten().tolist())

    state_stats = {
        "min": np.min(all_states),
        "max": np.max(all_states),
        "mean": np.mean(all_states),
        "std": max(np.std(all_states), 1e-8),  # Avoid zero std
    }

    action_stats = {
        "min": np.min(all_actions),
        "max": np.max(all_actions),
        "mean": np.mean(all_actions),
        "std": max(np.std(all_actions), 1e-8),  # Avoid zero std
    }

    # Format stats for LeRobot - use actual state statistics for observation.state
    stats = {
        "observation.state": {
            "min": torch.tensor([state_stats["min"]] * obs_state_shape[0]).float(),
            "max": torch.tensor([state_stats["max"]] * obs_state_shape[0]).float(),
            "mean": torch.tensor([state_stats["mean"]] * obs_state_shape[0]).float(),
            "std": torch.tensor([state_stats["std"]] * obs_state_shape[0]).float(),
        },
        "observation.environment_state": {
            "min": torch.tensor([]).float(),
            "max": torch.tensor([]).float(),
            "mean": torch.tensor([]).float(),
            "std": torch.tensor([]).float(),
        },
        "action": {
            "min": torch.tensor([action_stats["min"]] * action_shape[0]).float(),
            "max": torch.tensor([action_stats["max"]] * action_shape[0]).float(),
            "mean": torch.tensor([action_stats["mean"]] * action_shape[0]).float(),
            "std": torch.tensor([action_stats["std"]] * action_shape[0]).float(),
        },
    }

    # Add image stats if images are present
    if image_shape is not None:
        # Images are normalized to [0, 1], so use standard stats
        # Shape should be (C, H, W) - same as lerobot_image_shape
        C, H, W = lerobot_image_shape
        stats["observation.image"] = {
            "min": torch.zeros(C, H, W).float(),
            "max": torch.ones(C, H, W).float(),
            "mean": torch.ones(C, H, W).float() * 0.5,
            "std": torch.ones(C, H, W).float() * 0.5,
        }

    print("Dataset statistics computed.")

    # Create model with stats
    model = LeRobotDiffusionPolicy(diffusion_config, dataset_stats=stats).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if use_original_lerobot_params:
        # Use original LeRobot parameters: Adam optimizer, no weight decay, no complex scheduling
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
        )
        lr_scheduler = None  # No learning rate scheduling
        print("Using original LeRobot training parameters")
    else:
        # Use improved parameters: AdamW with weight decay and cosine scheduling
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            betas=diffusion_config.optimizer_betas,
            eps=diffusion_config.optimizer_eps,
            weight_decay=diffusion_config.optimizer_weight_decay,
        )

        # Create scheduler
        # Calculate total steps, ensuring at least 1 step per epoch to avoid division by zero
        steps_per_epoch = max(1, len(dataset) // config["batch_size"])
        total_steps = config["num_epochs"] * steps_per_epoch
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        persistent_workers=config.get("num_workers", 4) > 0,
    )

    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        wandb.init(
            project="lerobot-diffusion-policy-geom2d",
            config=config,
            name=f"lerobot_diffusion_policy_{int(time.time())}",
            dir=str(log_dir_path),
        )

    # Setup logging
    train_log_path = log_dir_path / "lerobot_training.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message("Starting LeRobot diffusion policy training...")
    log_message(f"Dataset: {dataset_path}")
    log_message(f"Model save path: {model_save_path}")
    log_message(f"Device: {device}")
    log_message(f"Config: {config}")

    # Training loop
    model.train()
    best_loss = float("inf")
    grad_scaler = GradScaler()

    if use_original_lerobot_params:
        # Use original LeRobot approach: train for a fixed number of steps
        training_steps = min(config.get("training_steps", 5000), config["num_epochs"] * len(dataloader))
        print(f"Training for {training_steps} steps using original LeRobot approach")
    else:
        # Use epoch-based training
        training_steps = config["num_epochs"] * len(dataloader)

    step = 0
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Check if we've reached the step limit for original LeRobot approach
            if use_original_lerobot_params and step >= training_steps:
                break

            # Convert our custom batch format to LeRobot format
            obs_states = batch["obs_states"].to(device)
            actions = batch["actions"].to(device)
            batch_size, action_horizon = actions.shape[:2]

            # Use actual robot state from observations instead of dummy values
            # For PushT, obs_states contains the agent position which is the robot state
            robot_state = obs_states  # Shape: [batch, obs_horizon, state_dim]

            # Create environment state as empty tensor since we're using robot state
            # This maintains compatibility with the dual-feature setup
            env_state = torch.zeros(
                batch_size, config.get("obs_horizon", 2), 0, device=device
            )

            lerobot_batch = {
                "observation.state": robot_state,  # Shape: [batch, obs_horizon, state_dim]
                # Empty environment state to maintain feature structure
                "observation.environment_state": env_state,
                "action": actions,  # Shape: [batch, action_horizon, action_dim]
                "action_is_pad": torch.zeros(
                    batch_size, action_horizon, dtype=torch.bool, device=device
                ),  # No padding
            }

            # Add images if available
            if "obs_images" in batch:
                obs_images = batch["obs_images"].to(device)  # Shape: [batch, obs_horizon, C, H, W]
                lerobot_batch["observation.image"] = obs_images

            # Forward pass through LeRobot policy
            loss, _ = model.forward(lerobot_batch)

            # Backward pass - simplified for original LeRobot approach
            if use_original_lerobot_params:
                # Use simple backward pass like original LeRobot script
                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("grad_clip_norm", 1.0)
                )

                optimizer.step()
            else:
                # Use gradient scaling for improved stability
                optimizer.zero_grad()
                grad_scaler.scale(loss).backward()

                # Gradient clipping
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("grad_clip_norm", 1.0)
                )

                grad_scaler.step(optimizer)
                grad_scaler.update()

            if lr_scheduler is not None:
                lr_scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1
            step += 1

            # Log progress
            if batch_idx % config.get("log_interval", 10) == 0:
                if use_original_lerobot_params:
                    msg = f"Step {step}/{training_steps}, Loss: {loss.item():.6f}"
                else:
                    msg = (
                        f"Epoch {epoch+1}/{config['num_epochs']}, "
                        f"Batch {batch_idx+1}/{len(dataloader)}, "
                        f"Loss: {loss.item():.6f}"
                    )
                log_message(msg)

                if config.get("use_wandb", False):
                    wandb.log(
                        {"batch_loss": loss.item(), "epoch": epoch, "batch": batch_idx, "step": step}
                    )

        # End of epoch
        avg_loss = epoch_loss / num_batches

        msg = f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}"
        log_message(msg)

        if config.get("use_wandb", False):
            wandb.log(
                {
                    "epoch_loss": avg_loss,
                    "learning_rate": (
                        lr_scheduler.get_last_lr()[0]
                        if lr_scheduler
                        else config.get("learning_rate", 1e-4)
                    ),
                    "epoch": epoch,
                }
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_config = config.copy()
            save_config.update(
                {
                    "obs_dim": obs_state_shape[0],
                    "action_dim": action_shape[0],
                    "image_shape": list(image_shape) if image_shape else None,
                    "policy_type": "lerobot",
                }
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": save_config,
                    "diffusion_config": diffusion_config,
                    "dataset_stats": stats,  # Save normalization stats for proper model loading
                },
                model_save_path,
            )
            log_message(f"Best LeRobot model saved with loss: {avg_loss:.6f}")

    log_message("LeRobot training completed!")

    if config.get("use_wandb", False):
        wandb.finish()

    return model


def get_default_training_config() -> Dict[str, Any]:
    """Get default training configuration."""
    return {
        "obs_horizon": 2,
        "action_horizon": 8,
        "pred_horizon": 8,
        "num_diffusion_iters": 100,
        "batch_size": 64,  # Increased to match original LeRobot script
        "num_epochs": 50,
        "training_steps": 5000,  # Match original LeRobot script
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "grad_clip_norm": 1.0,
        "num_workers": 4,
        "log_interval": 1,  # More frequent logging like original script
        "use_wandb": False,
        "force_cpu": False,
    }


def train_behavior_cloning_policy(
    dataset_path: str,
    model_save_path: str,
    config: Dict[str, Any],
    log_dir: str = "./logs",
) -> BehaviorCloningPolicy:
    """Train behavior cloning policy on the dataset.

    Args:
        dataset_path: Path to the dataset
        model_save_path: Path to save the trained model
        config: Training configuration
        log_dir: Directory to save logs

    Returns:
        Trained behavior cloning policy
    """
    # Setup logging
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / "bc_training.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message("Starting Behavior Cloning training...")
    log_message(f"Dataset: {dataset_path}")
    log_message(f"Model save path: {model_save_path}")
    log_message(f"Config: {config}")

    # Set device
    if config.get("force_cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    # Create dataset
    dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        pred_horizon=config["pred_horizon"],
        video_backend=config.get("video_backend", "pyav"),
    )

    # Create data loader
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )

    log_message(f"Dataset size: {len(dataset)}")
    log_message(f"Batch size: {config['batch_size']}")

    # Get dimensions from dataset
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    log_message(f"Observation dimension: {obs_dim}")
    log_message(f"Action dimension: {action_dim}")

    # Create model
    model = BehaviorCloningPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        hidden_dim=config.get("hidden_dim", 1024),
        num_layers=config.get("num_layers", 5),
    ).to(device)

    log_message(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Setup wandb if requested
    if config.get("use_wandb", False):
        wandb.init(
            project="prbench-behavior-cloning",
            config=config,
            name=f"bc_{Path(dataset_path).name}",
        )

    # Training loop
    model.train()
    best_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            obs_seq = batch["obs_states"].to(device)  # [B, obs_horizon, obs_dim]
            action_seq = batch["actions"].to(device)  # [B, action_horizon, action_dim]

            # Forward pass
            predicted_actions = model(obs_seq)

            # Compute MSE loss
            loss = F.mse_loss(predicted_actions, action_seq)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % 10 == 0:
                log_message(
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        # Update learning rate
        scheduler.step()

        # Compute average loss
        avg_loss = epoch_loss / num_batches

        log_message(
            f"Epoch {epoch+1}/{config['num_epochs']} completed. "
            f"Average loss: {avg_loss:.6f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        # Log to wandb if enabled
        if config.get("use_wandb", False):
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            log_message(f"New best loss: {best_loss:.6f}. Saving model...")

            # Save model checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "loss": best_loss,
                "config": {
                    **config,
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "policy_type": "behavior_cloning",
                },
            }

            torch.save(checkpoint, model_save_path)

    # Final save
    log_message("Training completed!")
    log_message(f"Best loss: {best_loss:.6f}")
    log_message(f"Model saved to: {model_save_path}")

    if config.get("use_wandb", False):
        wandb.finish()

    return model
