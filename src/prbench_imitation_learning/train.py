"""
Training functionality for diffusion policies.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from .policy import DiffusionPolicy, DiffusionPolicyDataset


def train_diffusion_policy(dataset_path: str, model_save_path: str, config: Dict[str, Any], 
                          log_dir: str = "./logs") -> DiffusionPolicy:
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
        pred_horizon=config["pred_horizon"]
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False
    )
    
    # Create model
    model = DiffusionPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        num_diffusion_iters=config["num_diffusion_iters"]
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])
    
    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        wandb.init(
            project="diffusion-policy-geom2d",
            config=config,
            name=f"diffusion_policy_{int(time.time())}",
            dir=str(log_dir)
        )
    
    # Setup logging
    train_log_path = log_dir / "training.log"
    
    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(train_log_path, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    log_message("Starting training...")
    log_message(f"Dataset: {dataset_path}")
    log_message(f"Model save path: {model_save_path}")
    log_message(f"Device: {device}")
    log_message(f"Config: {config}")
    
    # Training loop
    model.train()
    best_loss = float('inf')
    
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
                msg = (f"Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.6f}")
                log_message(msg)
                
                if config.get("use_wandb", False):
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        msg = f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}"
        log_message(msg)
        
        if config.get("use_wandb", False):
            wandb.log({
                "epoch_loss": avg_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch": epoch
            })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_config = config.copy()
            save_config.update({
                'obs_dim': dataset.obs_dim,
                'action_dim': dataset.action_dim,
                'image_shape': list(dataset.image_shape)
            })
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': save_config
            }, model_save_path)
            log_message(f"Best model saved with loss: {avg_loss:.6f}")
    
    log_message("Training completed!")
    
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
