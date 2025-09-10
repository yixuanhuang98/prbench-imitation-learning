"""Diffusion Policy models and dataset classes for imitation learning."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Diffusion imports
from diffusers import DDPMScheduler

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """Dataset wrapper for LeRobot datasets for diffusion policy training."""

    def __init__(
        self,
        dataset_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        pred_horizon: int = 8,
    ):
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
        pickle_path = self.dataset_path / "dataset.pkl"
        if pickle_path.exists():
            print("Loading pickle format dataset...")
            with open(pickle_path, "rb") as f:
                data_dict = pickle.load(f)
            self.episodes_data = data_dict["episodes"]
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
                total_sequences += (
                    episode_length - self.obs_horizon - self.pred_horizon + 1
                )
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
        for i in range(
            start_pos + self.obs_horizon,
            start_pos + self.obs_horizon + self.pred_horizon,
        ):
            frame_idx = episode_indices[i]
            if self.dataset:
                frame = self.dataset[frame_idx]
            else:
                frame = self.episodes_data[frame_idx]
            actions.append(frame["action"])

        # Convert to tensors
        obs_states = torch.stack([torch.from_numpy(obs) for obs in obs_states])
        obs_images = torch.stack(
            [torch.from_numpy(img).permute(2, 0, 1) / 255.0 for img in obs_images]
        )
        actions = torch.stack([torch.from_numpy(action) for action in actions])

        return {
            "obs_states": obs_states.float(),
            "obs_images": obs_images.float(),
            "actions": actions.float(),
        }


class ConditionalUNet1D(nn.Module):
    """1D U-Net for diffusion policy."""

    def __init__(
        self, input_dim: int, global_cond_dim: int, diffusion_step_embed_dim: int = 128
    ):
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
        self.down_blocks = nn.ModuleList(
            [
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
            ]
        )

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
        self.up_blocks = nn.ModuleList(
            [
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
            ]
        )

        # Condition projection layers (project to channel dimensions)
        self.cond_proj_layers = nn.ModuleList(
            [
                nn.Linear(diffusion_step_embed_dim + 128, 64),
                nn.Linear(diffusion_step_embed_dim + 128, 128),
                nn.Linear(diffusion_step_embed_dim + 128, 256),
            ]
        )

    def forward(self, x, timestep, global_cond):
        # Encode timestep
        timestep_embed = self.diffusion_step_encoder(
            self.get_timestep_embedding(timestep, 128)
        )

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
                skip = skip_connections[-(i + 1)]
                x = torch.cat([x, skip], dim=1)
            x = up_block(x)

        return x

    def get_timestep_embedding(self, timesteps, embedding_dim):
        """Create sinusoidal timestep embeddings."""
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
        )
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class DiffusionPolicy(nn.Module):
    """Diffusion Policy model."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int,
        action_horizon: int,
        num_diffusion_iters: int = 100,
    ):
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
            prediction_type="epsilon",
        )

    def forward(self, obs_seq, actions=None):
        # Encode observations
        batch_size = obs_seq.shape[0]
        obs_features = self.obs_encoder(obs_seq.reshape(batch_size, -1))

        if self.training and actions is not None:
            # Training: predict noise
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.num_diffusion_iters, (batch_size,), device=obs_seq.device
            )

            # Add noise to actions
            noise = torch.randn_like(actions)
            noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

            # Predict noise
            noise_pred = self.noise_pred_net(
                noisy_actions.transpose(1, 2), timesteps, obs_features
            ).transpose(1, 2)

            return noise_pred, noise
        else:
            # Inference: denoise step by step
            actions_pred = torch.randn(
                (batch_size, self.action_horizon, self.action_dim),
                device=obs_seq.device,
            )

            for t in self.noise_scheduler.timesteps:
                noise_pred = self.noise_pred_net(
                    actions_pred.transpose(1, 2),
                    t.expand(batch_size).to(obs_seq.device),
                    obs_features,
                ).transpose(1, 2)

                actions_pred = self.noise_scheduler.step(
                    noise_pred, t, actions_pred
                ).prev_sample

            return actions_pred
