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
    # Use separate robot state (1) and environment state (obs_dim)
    input_features = {
        "observation.state": PolicyFeature(shape=[1], type=FeatureType.STATE),
        "observation.environment_state": PolicyFeature(
            shape=list(obs_state_shape), type=FeatureType.ENV
        ),
    }
    # if image_shape is not None:
    #     # Convert from (H, W, C) to (C, H, W) format for LeRobot
    #     lerobot_image_shape = [image_shape[2], image_shape[0],
    #                           image_shape[1]]  # (C, H, W)
    #     input_features["observation.image"] = PolicyFeature(
    #         shape=lerobot_image_shape, type=FeatureType.VISUAL)

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
    # Align conditioning dim used in LeRobot residual FiLM blocks with eval path
    # Set timestep embedding so that
    # cond_dim = timestep_embed(=248) + global_cond(=120) = 368
    diffusion_config.diffusion_step_embed_dim = 248

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

    # Format stats for LeRobot
    stats = {
        "observation.state": {
            "min": torch.tensor([0.0]).float(),
            "max": torch.tensor([1.0]).float(),
            "mean": torch.tensor([0.5]).float(),
            "std": torch.tensor([0.5]).float(),
        },
        "observation.environment_state": {
            "min": torch.tensor([state_stats["min"]] * obs_state_shape[0]).float(),
            "max": torch.tensor([state_stats["max"]] * obs_state_shape[0]).float(),
            "mean": torch.tensor([state_stats["mean"]] * obs_state_shape[0]).float(),
            "std": torch.tensor([state_stats["std"]] * obs_state_shape[0]).float(),
        },
        "action": {
            "min": torch.tensor([action_stats["min"]] * action_shape[0]).float(),
            "max": torch.tensor([action_stats["max"]] * action_shape[0]).float(),
            "mean": torch.tensor([action_stats["mean"]] * action_shape[0]).float(),
            "std": torch.tensor([action_stats["std"]] * action_shape[0]).float(),
        },
    }

    # Skip image stats for now since we're only using state observations

    print("Dataset statistics computed.")

    # Create model with stats
    model = LeRobotDiffusionPolicy(diffusion_config, dataset_stats=stats).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler manually (LeRobot factory needs different config)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        betas=diffusion_config.optimizer_betas,
        eps=diffusion_config.optimizer_eps,
        weight_decay=diffusion_config.optimizer_weight_decay,
    )

    # Create scheduler
    total_steps = config["num_epochs"] * (len(dataset) // config["batch_size"])
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

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Convert our custom batch format to LeRobot format
            obs_states = batch["obs_states"].to(device)
            actions = batch["actions"].to(device)
            batch_size, action_horizon = actions.shape[:2]

            # Separate features: dummy robot state and environment state
            dummy_robot_state = (
                torch.ones(batch_size, config.get("obs_horizon", 2), 1, device=device)
                * 0.5
            )

            lerobot_batch = {
                "observation.state": dummy_robot_state,  # Shape: [batch, obs_horizon, 1]
                # [batch, obs_horizon, state_dim]
                "observation.environment_state": obs_states,
                "action": actions,  # Shape: [batch, action_horizon, action_dim]
                "action_is_pad": torch.zeros(
                    batch_size, action_horizon, dtype=torch.bool, device=device
                ),  # No padding
            }

            # Skip images for now - only using state observations

            # Forward pass through LeRobot policy
            loss, _ = model.forward(lerobot_batch)

            # Backward pass
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

            # Log progress
            if batch_idx % config.get("log_interval", 10) == 0:
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


def train_act_policy(
    dataset_path: str,
    model_save_path: str,
    config: Dict[str, Any],
    log_dir: str = "./logs",
):
    """Train Action Chunking Transformer (ACT) policy from LeRobot.

    Args:
        dataset_path: Path to the dataset
        model_save_path: Path to save the trained model
        config: Training configuration
        log_dir: Directory to save logs

    Returns:
        Trained ACT model
    """
    if not LEROBOT_AVAILABLE:
        raise ImportError(
            "LeRobot is not available. Please install it with: pip install lerobot"
        )

    # pylint: disable=import-outside-toplevel
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Set device
    if config.get("force_cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset
    dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        obs_horizon=config.get("obs_horizon", 1),  # ACT typically uses single obs
        action_horizon=config.get("action_horizon", 100),  # ACT chunk size
        pred_horizon=config.get("pred_horizon", 100),
    )
    print(f"Dataset loaded with {len(dataset)} sequences")

    # Get dimensions
    obs_state_shape = (dataset.obs_dim,)
    action_shape = (dataset.action_dim,)

    print(f"Observation state shape: {obs_state_shape}")
    print(f"Action shape: {action_shape}")

    # Create ACT config  
    # ACT typically uses images + proprioceptive state
    # Use images from dataset (C, H, W) format for LeRobot
    image_c, image_h, image_w = 3, 84, 84  # Resize to smaller resolution
    
    input_features = {
        "observation.image": PolicyFeature(
            shape=[image_c, image_h, image_w], type=FeatureType.VISUAL
        ),
        "observation.state": PolicyFeature(
            shape=list(obs_state_shape), type=FeatureType.STATE
        ),
    }

    output_features = {
        "action": PolicyFeature(shape=list(action_shape), type=FeatureType.ACTION),
    }

    act_config = ACTConfig()
    act_config.input_features = input_features
    act_config.output_features = output_features
    act_config.n_obs_steps = config.get("obs_horizon", 1)
    act_config.chunk_size = config.get("action_horizon", 100)
    act_config.n_action_steps = config.get("action_horizon", 100)
    act_config.optimizer_lr = config.get("learning_rate", 1e-5)
    act_config.hidden_dim = config.get("hidden_dim", 512)
    act_config.dim_feedforward = config.get("dim_feedforward", 3200)
    act_config.num_encoder_layers = config.get("num_encoder_layers", 4)
    act_config.num_decoder_layers = config.get("num_decoder_layers", 7)
    act_config.crop_shape = None  # No cropping
    act_config.image_size = (image_h, image_w)  # Target image size

    # Compute dataset statistics for normalization
    print("Computing dataset statistics for normalization...")
    print(f"Processing images to {image_h}x{image_w} for statistics computation...")
    stats = {}

    # pylint: disable=import-outside-toplevel
    import torchvision.transforms.functional as TF

    all_states = []
    all_actions = []
    all_images_processed = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        all_states.extend(sample["obs_states"].flatten().tolist())
        all_actions.extend(sample["actions"].flatten().tolist())
        
        # Process images for statistics
        obs_images = sample["obs_images"]  # [obs_horizon, H, W, C]
        for img in obs_images:
            # Ensure img is numpy array
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            img = np.asarray(img)  # Ensure it's an array
            
            # Convert to torch format and process
            # numpy (H, W, C) -> torch (C, H, W)
            if len(img.shape) == 3 and img.shape[2] == 3:  # (H, W, C)
                img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif len(img.shape) == 3 and img.shape[0] == 3:  # Already (C, H, W)
                img_torch = torch.from_numpy(img).float() / 255.0
            else:
                print(f"Warning: unexpected image shape {img.shape}, skipping")
                continue
                
            # Resize to target size
            img_resized = TF.resize(img_torch, [image_h, image_w])
            all_images_processed.append(img_resized.numpy())  # Store as [C, H, W]
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(dataset)} samples for statistics...")

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    
    # Stack images carefully
    if len(all_images_processed) > 0:
        all_images_processed = np.stack(all_images_processed, axis=0)  # [N, C, H, W]
        print(f"Stacked image array shape: {all_images_processed.shape}")
    else:
        raise ValueError("No images were processed for statistics!")

    # Reshape to proper dimensions for computing statistics
    all_states_reshaped = all_states.reshape(-1, obs_state_shape[0])
    all_actions_reshaped = all_actions.reshape(-1, action_shape[0])

    # Compute image statistics on resized images
    # Shape: [C, H, W] - compute mean/std across all samples (axis=0)
    print(f"Computing image statistics from {len(all_images_processed)} processed images...")
    image_mean = np.mean(all_images_processed, axis=0).astype(np.float32)  # [C, H, W]
    image_std = np.std(all_images_processed, axis=0).astype(np.float32) + 1e-8  # [C, H, W]
    
    stats["observation.image"] = {
        "mean": image_mean,
        "std": image_std,
    }
    
    stats["observation.state"] = {
        "mean": np.mean(all_states_reshaped, axis=0).astype(np.float32),
        "std": (np.std(all_states_reshaped, axis=0).astype(np.float32) + 1e-8),
    }
    
    stats["action"] = {
        "mean": np.mean(all_actions_reshaped, axis=0).astype(np.float32),
        "std": (np.std(all_actions_reshaped, axis=0).astype(np.float32) + 1e-8),
    }
    
    print(f"Image statistics computed: mean shape {image_mean.shape}, std shape {image_std.shape}")

    # Create model
    model = ACTPolicy(act_config, dataset_stats=stats)
    model = model.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-5),
        weight_decay=config.get("weight_decay", 1e-6),
    )

    # Create data loader
    batch_size = config.get("batch_size", 8)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    # Training loop
    num_epochs = config.get("num_epochs", 100)
    print(f"\nStarting ACT training for {num_epochs} epochs...")

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            obs_seq = batch["obs_states"].to(device)  # [B, obs_horizon, obs_dim]
            action_seq = batch["actions"].to(device)  # [B, action_horizon, action_dim]
            obs_images = batch["obs_images"]  # [B, obs_horizon, H, W, C]

            # Prepare batch for ACT
            # ACT expects: batch_size, timesteps, features
            # Take only the first observation (ACT typically uses single observation)
            obs_single = obs_seq[:, 0, :]  # [B, obs_dim]
            
            # Process images: [B, H, W, C] -> [B, C, H, W] and resize
            img_single = obs_images[:, 0]  # [B, H, W, C]
            
            # Convert to torch and resize
            import torchvision.transforms.functional as TF
            img_processed = []
            for idx, img in enumerate(img_single):
                # Ensure img is numpy array
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                img = np.asarray(img)
                
                # Debug first image
                if idx == 0 and batch_idx == 0 and epoch == 0:
                    print(f"Raw image shape before processing: {img.shape}")
                
                # Convert to torch format
                # numpy (H, W, C) -> torch (C, H, W)
                if len(img.shape) == 3 and img.shape[2] == 3:  # (H, W, C)
                    img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                elif len(img.shape) == 3 and img.shape[0] == 3:  # Already (C, H, W)
                    img_torch = torch.from_numpy(img).float() / 255.0
                else:
                    print(f"Error: unexpected image shape {img.shape}")
                    # Create dummy image
                    img_torch = torch.zeros(3, image_h, image_w)
                    
                # Resize to target size
                img_resized = TF.resize(img_torch, [image_h, image_w])
                
                if idx == 0 and batch_idx == 0 and epoch == 0:
                    print(f"Processed image shape: {img_resized.shape}")
                    
                img_processed.append(img_resized)
            
            img_tensor = torch.stack(img_processed).to(device)  # [B, C, H, W]
            # Don't add temporal dimension - ACT handles it internally
            # img_tensor shape: [B, C, H, W]
            
            # State also without temporal dimension
            # obs_single shape: [B, obs_dim]

            # Create action padding mask (all False since we have no padding)
            action_is_pad = torch.zeros(
                action_seq.shape[0], action_seq.shape[1], dtype=torch.bool, device=device
            )  # [B, chunk_size]
            
            batch_act = {
                "observation.image": img_tensor,  # [B, C, H, W]
                "observation.state": obs_single,  # [B, obs_dim]
                "action": action_seq,  # [B, chunk_size, action_dim]
                "action_is_pad": action_is_pad,  # [B, chunk_size]
            }

            # Debug: Print shapes on first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Batch shapes:")
                print(f"  image: {img_tensor.shape}")
                print(f"  state: {obs_single.shape}")
                print(f"  action: {action_seq.shape}")

            # Forward pass
            output = model(batch_act)
            
            # Handle ACT output format
            if isinstance(output, dict) and "loss" in output:
                loss = output["loss"]
            elif isinstance(output, tuple):
                # ACT returns predictions during forward, compute loss ourselves
                # For now, use simple MSE loss between predicted and actual actions
                actions_pred = output[0] if len(output) > 0 else output
                
                # Ensure shapes match
                if actions_pred.shape != action_seq.shape:
                    print(f"Warning: shape mismatch - pred: {actions_pred.shape}, target: {action_seq.shape}")
                    # Use only matching dimensions
                    min_len = min(actions_pred.shape[1], action_seq.shape[1])
                    actions_pred = actions_pred[:, :min_len, :]
                    action_seq_trimmed = action_seq[:, :min_len, :]
                    loss = torch.nn.functional.mse_loss(actions_pred, action_seq_trimmed)
                else:
                    loss = torch.nn.functional.mse_loss(actions_pred, action_seq)
            else:
                raise ValueError(f"Unexpected output format from ACT model: {type(output)}")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.get("grad_clip_norm", 1.0)
            )
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % config.get("log_interval", 10) == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")

    print("Training completed!")

    # Save model
    save_config = config.copy()
    save_config.update(
        {
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "image_shape": list(dataset.image_shape),
            "policy_type": "act",
        }
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": save_config,
        "act_config": model.config,
        "stats": stats,
    }

    torch.save(checkpoint, model_save_path)
    print(f"Model saved to: {model_save_path}")

    return model


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
