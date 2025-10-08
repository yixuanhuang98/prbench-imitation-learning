"""Training functionality for diffusion policies."""

import sys
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
    from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

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

    # Import LeRobot classes here to avoid issues if not available
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.utils import dataset_to_policy_features

    # Create log directory
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # Set device
    if config.get("force_cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use LeRobot's metadata approach like the original script
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # Configure delta_timestamps to match the policy configuration
    # This tells the dataset how to load temporal data (observations and actions)
    obs_horizon = config.get("obs_horizon", 2)
    pred_horizon = config.get("pred_horizon", 16)
    
    delta_timestamps = {}
    # For observations: load the last n_obs_steps frames
    for key in input_features.keys():
        delta_timestamps[key] = [i / dataset_metadata.fps for i in range(1 - obs_horizon, 1)]
    
    # For actions: load pred_horizon future actions
    for key in output_features.keys():
        delta_timestamps[key] = [i / dataset_metadata.fps for i in range(pred_horizon)]
    
    print(f"Delta timestamps configuration: {delta_timestamps}")
    
    # Create dataset with proper delta_timestamps - use pyav backend to avoid FFmpeg issues
    try:
        dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps, video_backend="pyav")
        print(f"Dataset loaded with {len(dataset)} sequences")
    except Exception as e:
        print(f"Warning: Failed to load with delta_timestamps, trying without: {e}")
        # Fallback: load without delta_timestamps
        dataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
        print(f"Dataset loaded with fallback settings: {len(dataset)} sequences")

    # Get dimensions from features (which contain the shape information)
    state_feature = None
    action_feature = None
    image_feature = None

    for key, feature in features.items():
        if feature.type == FeatureType.STATE and "state" in key:
            state_feature = feature
        elif feature.type == FeatureType.ACTION:
            action_feature = feature
        elif feature.type == FeatureType.VISUAL and "image" in key:
            image_feature = feature

    if state_feature is None:
        raise ValueError("No state feature found in dataset")
    if action_feature is None:
        raise ValueError("No action feature found in dataset")

    obs_state_shape = (state_feature.shape[0],)
    action_shape = (action_feature.shape[0],)
    image_shape = image_feature.shape if image_feature else None

    print(f"Observation state shape: {obs_state_shape}")
    print(f"Action shape: {action_shape}")
    if image_shape is not None:
        print(f"Image shape: {image_shape}")

    # Use the features extracted from metadata (already properly formatted)
    # Create config and set required features - use LeRobot's recommended settings for PushT
    diffusion_config = DiffusionConfig()
    diffusion_config.input_features = input_features
    diffusion_config.output_features = output_features

    # Use exact configuration from successful LeRobot training
    diffusion_config.n_obs_steps = config.get("obs_horizon", 2)
    diffusion_config.horizon = config.get("pred_horizon", 16)
    diffusion_config.n_action_steps = config.get("action_horizon", 8)
    diffusion_config.num_train_timesteps = config.get("num_diffusion_iters", 100)

    # Beta schedule parameters (matching successful run)
    diffusion_config.beta_schedule = config.get("beta_schedule", "squaredcos_cap_v2")
    diffusion_config.beta_start = config.get("beta_start", 0.0001)
    diffusion_config.beta_end = config.get("beta_end", 0.02)
    diffusion_config.prediction_type = "epsilon"
    diffusion_config.clip_sample = True
    diffusion_config.clip_sample_range = 1.0

    # Frame dropping configuration
    diffusion_config.drop_n_last_frames = config.get("drop_n_last_frames", 7)

    # Optimizer parameters (matching successful run)
    diffusion_config.optimizer_lr = config.get("learning_rate", 1e-4)
    diffusion_config.optimizer_betas = tuple(config.get("optimizer_betas", [0.95, 0.999]))
    diffusion_config.optimizer_eps = config.get("optimizer_eps", 1e-08)
    diffusion_config.optimizer_weight_decay = config.get("weight_decay", 1e-06)

    # Diffusion step embedding
    diffusion_config.diffusion_step_embed_dim = config.get("diffusion_step_embed_dim", 128)

    # Vision backbone configuration (matching successful run)
    diffusion_config.vision_backbone = config.get("vision_backbone", "resnet18")
    diffusion_config.down_dims = tuple(config.get("down_dims", [512, 1024, 2048]))
    diffusion_config.kernel_size = config.get("kernel_size", 5)
    diffusion_config.n_groups = config.get("n_groups", 8)
    diffusion_config.crop_shape = tuple(config.get("crop_shape", [84, 84]))
    diffusion_config.crop_is_random = True
    diffusion_config.spatial_softmax_num_keypoints = config.get("spatial_softmax_num_keypoints", 32)
    diffusion_config.use_film_scale_modulation = config.get("use_film_scale_modulation", True)
    diffusion_config.use_group_norm = config.get("use_group_norm", True)
    diffusion_config.use_separate_rgb_encoder_per_camera = False

    print("Using improved LeRobot configuration:")
    print(f"  horizon: {diffusion_config.horizon}")
    print(f"  n_action_steps: {diffusion_config.n_action_steps}")
    print(f"  drop_n_last_frames: {diffusion_config.drop_n_last_frames}")
    print(f"  beta_schedule: {diffusion_config.beta_schedule}")

    # LeRobot dataset handles statistics internally, no need to compute manually

    # Create model without dataset_stats (stats are handled by preprocessors)
    model = LeRobotDiffusionPolicy(diffusion_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create preprocessor and postprocessor for data handling
    preprocessor, postprocessor = make_diffusion_pre_post_processors(
        config=diffusion_config,
        dataset_stats=dataset_metadata.stats,
    )
    print("Created preprocessor and postprocessor pipelines")

    # Use Adam optimizer with exact parameters from successful run
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        betas=diffusion_config.optimizer_betas,
        eps=diffusion_config.optimizer_eps,
        weight_decay=diffusion_config.optimizer_weight_decay,
    )
    
    # Use cosine annealing with warmup (matching LeRobot's scheduler)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    
    steps_per_epoch = max(1, len(dataset) // config["batch_size"])
    total_steps = config.get("training_steps", 200000)
    warmup_steps = 500  # Matching LeRobot's warmup
    
    # Create warmup scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.001,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    
    # Create cosine annealing scheduler
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=0
    )
    
    # Combine warmup and cosine
    lr_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    
    print(f"Using Adam optimizer with cosine annealing (warmup: {warmup_steps} steps)")

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

    # Training loop - step-based training matching LeRobot
    model.train()
    best_loss = float("inf")
    
    # Use gradient scaler for mixed precision (optional, but good for performance)
    use_amp = config.get("use_amp", False)
    grad_scaler = GradScaler(enabled=use_amp)

    training_steps = config.get("training_steps", 200000)
    log_interval = config.get("log_interval", 200)
    save_freq = config.get("save_freq", 25000)
    
    print(f"Training for {training_steps} steps (log every {log_interval} steps)")

    step = 0
    epoch = 0
    
    # Create infinite dataloader iterator
    from itertools import cycle
    dataloader_iterator = cycle(dataloader)
    
    while step < training_steps:
        # Get next batch
        batch = next(dataloader_iterator)
        
        # Preprocess batch (normalizes, moves to device, etc.)
        batch = preprocessor(batch)

        # Forward pass through LeRobot policy (with optional AMP)
        with torch.amp.autocast('cuda', enabled=use_amp):
            loss, _ = model.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        
        if use_amp:
            # Use gradient scaling for mixed precision
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.get("grad_clip_norm", 10.0),
                error_if_nonfinite=False
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config.get("grad_clip_norm", 10.0),
                error_if_nonfinite=False
            )
            optimizer.step()
        
        # Step the learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        step += 1
        
        # Log progress
        if step % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            msg = f"Step {step}/{training_steps}, Loss: {loss.item():.6f}, LR: {current_lr:.2e}"
            log_message(msg)
            
            if config.get("use_wandb", False):
                wandb.log({
                    "loss": loss.item(),
                    "learning_rate": current_lr,
                    "step": step
                })
        
        # Save checkpoint periodically
        if step % save_freq == 0 or step == training_steps:
            save_config = config.copy()
            save_config.update({
                "obs_dim": obs_state_shape[0],
                "action_dim": action_shape[0],
                "image_shape": list(image_shape) if image_shape else None,
                "policy_type": "lerobot",
            })
            
            checkpoint_path = Path(model_save_path).parent / f"checkpoint_step_{step}.pth"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
                "loss": loss.item(),
                "config": save_config,
                "diffusion_config": diffusion_config,
            }, checkpoint_path)
            log_message(f"Checkpoint saved at step {step}: {checkpoint_path}")
            
        # Track best loss and save best model
        if loss.item() < best_loss:
            best_loss = loss.item()
            save_config = config.copy()
            save_config.update({
                "obs_dim": obs_state_shape[0],
                "action_dim": action_shape[0],
                "image_shape": list(image_shape) if image_shape else None,
                "policy_type": "lerobot",
            })
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
                "loss": best_loss,
                "config": save_config,
                "diffusion_config": diffusion_config,
            }, model_save_path)
            log_message(f"Best model updated at step {step} with loss: {best_loss:.6f}")

    log_message("LeRobot training completed!")

    if config.get("use_wandb", False):
        wandb.finish()

    return model


def get_default_training_config() -> Dict[str, Any]:
    """Get default training configuration matching successful LeRobot setup."""
    return {
        # Observation and action horizons
        "obs_horizon": 2,  # n_obs_steps
        "action_horizon": 8,  # n_action_steps
        "pred_horizon": 16,  # horizon (full prediction horizon)
        "num_diffusion_iters": 100,  # num_train_timesteps
        
        # Training parameters (matching successful LeRobot run)
        "batch_size": 64,
        "num_epochs": 500,  # Will train for 200K steps total
        "training_steps": 200000,  # Total training steps
        "learning_rate": 1e-4,
        "weight_decay": 1e-6,
        "optimizer_betas": [0.95, 0.999],
        "optimizer_eps": 1e-08,
        "grad_clip_norm": 10.0,  # Matching LeRobot's grad_clip_norm
        
        # LeRobot-specific diffusion parameters
        "beta_schedule": "squaredcos_cap_v2",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "drop_n_last_frames": 7,  # horizon - n_action_steps - n_obs_steps + 1 = 16 - 8 - 2 + 1 = 7
        "diffusion_step_embed_dim": 128,
        
        # Vision parameters (matching LeRobot)
        "vision_backbone": "resnet18",
        "down_dims": [512, 1024, 2048],
        "n_groups": 8,
        "kernel_size": 5,
        "crop_shape": [84, 84],
        "spatial_softmax_num_keypoints": 32,
        "use_film_scale_modulation": True,
        "use_group_norm": True,
        
        # Dataset parameters
        "video_backend": "pyav",  # Use pyav to avoid FFmpeg issues
        "num_workers": 4,
        
        # Logging
        "log_interval": 200,  # Log every 200 steps like LeRobot
        "save_freq": 25000,  # Save checkpoint every 25000 steps
        "eval_freq": 25000,  # Evaluate every 25000 steps
        "use_wandb": False,
        "force_cpu": False,
        "seed": 100000,  # Matching successful run
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
