#!/usr/bin/env python3
"""
Script to train a diffusion policy on LeRobot format data from geom2d environments.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Diffusion imports
from diffusers import DDPMScheduler
from einops import rearrange
import wandb

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class DiffusionPolicyDataset(Dataset):
    """Dataset wrapper for LeRobot datasets for diffusion policy training."""
    
    def __init__(self, dataset_path: str, obs_horizon: int = 2, action_horizon: int = 8, pred_horizon: int = 8):
        """
        Args:
            dataset_path: Path to LeRobot dataset
            obs_horizon: Number of observation frames to use as context
            action_horizon: Number of action frames to predict
            pred_horizon: Number of future action steps to predict
        """
        self.dataset_path = Path(dataset_path)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        
        # Try to load as LeRobot dataset first, fallback to pickle
        print(f"Loading dataset from {dataset_path}")
        self.dataset = None
        self.episodes_data = None
        
        # Check if it's a pickle file
        pickle_path = self.dataset_path / 'dataset.pkl'
        if pickle_path.exists():
            print("Loading pickle format dataset...")
            import pickle
            with open(pickle_path, 'rb') as f:
                data_dict = pickle.load(f)
            self.episodes_data = data_dict['episodes']
            print(f"Loaded {len(self.episodes_data)} frames from pickle")
        else:
            # Try LeRobot format
            try:
                self.dataset = LeRobotDataset(dataset_path)
                print(f"Dataset loaded with {len(self.dataset)} frames")
            except Exception as e:
                print(f"Failed to load LeRobot dataset: {e}")
                raise ValueError(f"Could not load dataset from {dataset_path}")
        
        # Get dataset info
        if self.dataset:
            self.episodes = self._group_by_episodes()
            print(f"Found {len(self.episodes)} episodes")
            
            # Get observation and action dimensions
            sample_data = self.dataset[0]
            self.obs_dim = sample_data["observation.state"].shape[0]
            self.action_dim = sample_data["action"].shape[0]
            self.image_shape = sample_data["observation.image"].shape
        else:
            # Using pickle data
            self.episodes = self._group_episodes_from_pickle()
            print(f"Found {len(self.episodes)} episodes")
            
            # Get dimensions from first sample
            sample_data = self.episodes_data[0]
            self.obs_dim = sample_data["observation.state"].shape[0]
            self.action_dim = sample_data["action"].shape[0]
            self.image_shape = sample_data["observation.image"].shape
        
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        print(f"Image shape: {self.image_shape}")
        
    def _group_by_episodes(self):
        """Group dataset frames by episode."""
        episodes = {}
        for i in range(len(self.dataset)):
            episode_idx = self.dataset[i]["episode_index"]
            if episode_idx not in episodes:
                episodes[episode_idx] = []
            episodes[episode_idx].append(i)
        return episodes
    
    def _group_episodes_from_pickle(self):
        """Group pickle data frames by episode."""
        episodes = {}
        for i, frame in enumerate(self.episodes_data):
            episode_idx = frame["episode_index"]
            if episode_idx not in episodes:
                episodes[episode_idx] = []
            episodes[episode_idx].append(i)
        return episodes
    
    def __len__(self):
        # Count valid sequences
        total_sequences = 0
        for episode_indices in self.episodes.values():
            episode_length = len(episode_indices)
            if episode_length >= self.obs_horizon + self.pred_horizon:
                total_sequences += episode_length - self.obs_horizon - self.pred_horizon + 1
        return total_sequences
    
    def __getitem__(self, idx):
        # Find which episode and position this index corresponds to
        current_idx = 0
        for episode_indices in self.episodes.values():
            episode_length = len(episode_indices)
            if episode_length >= self.obs_horizon + self.pred_horizon:
                valid_starts = episode_length - self.obs_horizon - self.pred_horizon + 1
                if current_idx + valid_starts > idx:
                    # This episode contains our target index
                    start_pos = idx - current_idx
                    break
                current_idx += valid_starts
        
        # Extract observation sequence
        obs_states = []
        obs_images = []
        for i in range(start_pos, start_pos + self.obs_horizon):
            frame_idx = episode_indices[i]
            if self.dataset:
                frame = self.dataset[frame_idx]
            else:
                frame = self.episodes_data[frame_idx]
            obs_states.append(frame["observation.state"])
            obs_images.append(frame["observation.image"])
        
        # Extract action sequence
        actions = []
        for i in range(start_pos + self.obs_horizon, start_pos + self.obs_horizon + self.pred_horizon):
            frame_idx = episode_indices[i]
            if self.dataset:
                frame = self.dataset[frame_idx]
            else:
                frame = self.episodes_data[frame_idx]
            actions.append(frame["action"])
        
        # Convert to tensors
        obs_states = torch.stack([torch.from_numpy(obs) for obs in obs_states])
        obs_images = torch.stack([torch.from_numpy(img).permute(2, 0, 1) / 255.0 for img in obs_images])
        actions = torch.stack([torch.from_numpy(action) for action in actions])
        
        return {
            "obs_states": obs_states.float(),
            "obs_images": obs_images.float(),
            "actions": actions.float(),
        }


class ConditionalUNet1D(nn.Module):
    """1D U-Net for diffusion policy."""
    
    def __init__(self, input_dim: int, global_cond_dim: int, diffusion_step_embed_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.global_cond_dim = global_cond_dim
        
        # Time embedding
        self.diffusion_step_encoder = nn.Sequential(
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
            nn.SiLU(),
            nn.Linear(diffusion_step_embed_dim, diffusion_step_embed_dim),
        )
        
        # Global condition encoder
        self.global_cond_encoder = nn.Sequential(
            nn.Linear(global_cond_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
        )
        
        # U-Net architecture
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_dim, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, 3, padding=1, stride=2),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
                nn.Conv1d(128, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, 3, padding=1, stride=2),
                nn.GroupNorm(8, 256),
                nn.SiLU(),
                nn.Conv1d(256, 256, 3, padding=1),
                nn.GroupNorm(8, 256),
                nn.SiLU(),
            ),
        ])
        
        # Middle block
        self.mid_block = nn.Sequential(
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
        )
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(512, 128, 4, stride=2, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
                nn.Conv1d(128, 128, 3, padding=1),
                nn.GroupNorm(8, 128),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(256, 64, 4, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv1d(64, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv1d(128, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.SiLU(),
                nn.Conv1d(64, input_dim, 3, padding=1),
            ),
        ])
        
        # Condition projection layers (project to channel dimensions)
        self.cond_proj_layers = nn.ModuleList([
            nn.Linear(diffusion_step_embed_dim + 128, 64),
            nn.Linear(diffusion_step_embed_dim + 128, 128), 
            nn.Linear(diffusion_step_embed_dim + 128, 256),
        ])
    
    def forward(self, x, timestep, global_cond):
        # Encode timestep
        timestep_embed = self.diffusion_step_encoder(self.get_timestep_embedding(timestep, 128))
        
        # Encode global condition
        global_cond_embed = self.global_cond_encoder(global_cond)
        
        # Combine conditions
        cond = torch.cat([timestep_embed, global_cond_embed], dim=-1)
        
        # U-Net forward pass
        skip_connections = []
        
        # Downsampling
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            skip_connections.append(x)
        
        # Middle
        x = self.mid_block(x)
        
        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                x = torch.cat([x, skip], dim=1)
            x = up_block(x)
        
        return x
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DiffusionPolicy(nn.Module):
    """Diffusion Policy model."""
    
    def __init__(self, obs_dim: int, action_dim: int, obs_horizon: int, action_horizon: int, 
                 num_diffusion_iters: int = 100):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_diffusion_iters = num_diffusion_iters
        
        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # Noise prediction network
        self.noise_pred_net = ConditionalUNet1D(
            input_dim=action_dim,
            global_cond_dim=256,
        )
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon"
        )
    
    def forward(self, obs_seq, actions=None):
        # Encode observations
        batch_size = obs_seq.shape[0]
        obs_features = self.obs_encoder(obs_seq.reshape(batch_size, -1))
        
        if self.training and actions is not None:
            # Training: predict noise
            # Sample random timesteps
            timesteps = torch.randint(0, self.num_diffusion_iters, (batch_size,), device=obs_seq.device)
            
            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)
            
            # Predict noise
            noise_pred = self.noise_pred_net(
                noisy_actions.transpose(1, 2), 
                timesteps, 
                obs_features
            ).transpose(1, 2)
            
            return noise_pred, noise
        else:
            # Inference: denoise step by step
            actions_pred = torch.randn((batch_size, self.action_horizon, self.action_dim), 
                                     device=obs_seq.device)
            
            for t in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    actions_pred.transpose(1, 2),
                    t.expand(batch_size).to(obs_seq.device),
                    obs_features
                ).transpose(1, 2)
                
                actions_pred = self.noise_scheduler.step(noise_pred, t, actions_pred).prev_sample
            
            return actions_pred


def train_diffusion_policy(dataset_path: str, model_save_path: str, config: Dict[str, Any]):
    """Train diffusion policy on the dataset."""
    
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
            name=f"diffusion_policy_{int(time.time())}"
        )
    
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
                print(f"Epoch {epoch+1}/{config['num_epochs']}, "
                      f"Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.6f}")
                
                if config.get("use_wandb", False):
                    wandb.log({
                        "batch_loss": loss.item(),
                        "epoch": epoch,
                        "batch": batch_idx
                    })
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        scheduler.step()
        
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}")
        
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
            print(f"Best model saved with loss: {avg_loss:.6f}")
    
    print("Training completed!")
    
    if config.get("use_wandb", False):
        wandb.finish()
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train diffusion policy on LeRobot data")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to LeRobot dataset")
    parser.add_argument("--model-save-path", type=str, default="./diffusion_policy_model.pth",
                       help="Path to save trained model")
    parser.add_argument("--config", type=str, 
                       help="Path to JSON config file")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        "obs_horizon": 2,
        "action_horizon": 8,
        "pred_horizon": 8,
        "num_diffusion_iters": 100,
        "batch_size": args.batch_size,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-6,
        "grad_clip_norm": 1.0,
        "num_workers": 4,
        "log_interval": 10,
        "use_wandb": args.use_wandb,
    }
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            file_config = json.load(f)
        config.update(file_config)
    
    print("="*50)
    print("Diffusion Policy Training")
    print("="*50)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*50)
    
    try:
        # Create model save directory
        os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
        
        # Train model
        model = train_diffusion_policy(args.dataset_path, args.model_save_path, config)
        
        print("\n" + "="*50)
        print("SUCCESS: Training completed!")
        print(f"Model saved to: {args.model_save_path}")
        print("="*50)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
