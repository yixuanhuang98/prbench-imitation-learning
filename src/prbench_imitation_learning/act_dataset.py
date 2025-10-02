"""ACT-specific dataset format for LeRobot compatibility."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle


class ACTDataset(Dataset):
    """Dataset formatted specifically for LeRobot's ACT policy.
    
    ACT expects:
    - observation.images: [B, C, H, W] - Single images (no temporal dim)
    - observation.environment_state: [B, state_dim] - Environment state
    - observation.state: [B, state_dim] - Same as environment_state
    - action: [B, chunk_size, action_dim] - Action sequences
    - action_is_pad: [B, chunk_size] - Padding mask
    """

    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 8,
        image_size: tuple = (84, 84),
    ):
        """Initialize ACT dataset.
        
        Args:
            dataset_path: Path to the dataset
            chunk_size: Number of actions to predict (ACT chunk size)
            image_size: Target image size (H, W)
        """
        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size
        self.image_size = image_size
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset from pickle file (LeRobot format)."""
        dataset_file = self.dataset_path / "dataset.pkl"
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")
        
        with open(dataset_file, "rb") as f:
            data = pickle.load(f)
        
        # LeRobot format: data has 'episodes' list where each item is a frame
        frames = data["episodes"]
        
        # Group frames by episode
        episodes_dict = {}
        for frame in frames:
            ep_idx = frame["episode_index"]
            if ep_idx not in episodes_dict:
                episodes_dict[ep_idx] = []
            episodes_dict[ep_idx].append(frame)
        
        # Extract per-episode data
        self.observations = []
        self.actions = []
        self.images = []
        
        for ep_idx in sorted(episodes_dict.keys()):
            ep_frames = episodes_dict[ep_idx]
            
            ep_obs = [f["observation.state"] for f in ep_frames]
            ep_actions = [f["action"] for f in ep_frames]
            
            self.observations.append(np.array(ep_obs))
            self.actions.append(np.array(ep_actions))
            
            if "observation.image" in ep_frames[0]:
                ep_images = [f["observation.image"] for f in ep_frames]
                self.images.append(np.array(ep_images))
        
        # Get dimensions
        first_obs = self.observations[0][0]
        first_action = self.actions[0][0]
        
        self.obs_dim = len(first_obs) if isinstance(first_obs, (list, np.ndarray)) else first_obs.shape[0]
        self.action_dim = len(first_action) if isinstance(first_action, (list, np.ndarray)) else first_action.shape[0]
        
        # Create indices for all valid sequences
        self.indices = []
        for ep_idx, ep_actions in enumerate(self.actions):
            ep_len = len(ep_actions)
            # We can start from any timestep
            for t in range(ep_len):
                self.indices.append((ep_idx, t))
        
        print(f"Loaded {len(self.observations)} episodes")
        print(f"Created {len(self.indices)} training samples")
        print(f"Observation dim: {self.obs_dim}")
        print(f"Action dim: {self.action_dim}")
        if len(self.images) > 0:
            first_img = self.images[0][0]
            print(f"Image shape: {first_img.shape if hasattr(first_img, 'shape') else 'N/A'}")
    
    def __len__(self):
        """Return number of samples."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """Get a single sample.
        
        Returns:
            dict with keys:
                - obs_state: [obs_dim] - Current observation state
                - obs_image: [C, H, W] or None - Current observation image (resized)
                - actions: [chunk_size, action_dim] - Action chunk
                - action_is_pad: [chunk_size] - Padding mask (bool)
        """
        ep_idx, t = self.indices[idx]
        
        # Get current observation
        obs_state = np.array(self.observations[ep_idx][t], dtype=np.float32)
        
        # Get image if available
        obs_image = None
        if self.images is not None:
            img = self.images[ep_idx][t]
            if isinstance(img, np.ndarray):
                # Resize and convert to [C, H, W] format
                obs_image = self._process_image(img)
        
        # Get action chunk
        ep_actions = self.actions[ep_idx]
        ep_len = len(ep_actions)
        
        # Extract action chunk starting from current timestep
        action_chunk = []
        action_is_pad = []
        
        for i in range(self.chunk_size):
            if t + i < ep_len:
                action_chunk.append(ep_actions[t + i])
                action_is_pad.append(False)
            else:
                # Pad with last action
                action_chunk.append(ep_actions[-1])
                action_is_pad.append(True)
        
        action_chunk = np.array(action_chunk, dtype=np.float32)
        action_is_pad = np.array(action_is_pad, dtype=bool)
        
        return {
            "obs_state": obs_state,
            "obs_image": obs_image,
            "actions": action_chunk,
            "action_is_pad": action_is_pad,
        }
    
    def _process_image(self, img):
        """Process image to [C, H, W] format with target size.
        
        Args:
            img: Input image [H, W, C] or [C, H, W]
            
        Returns:
            Processed image [C, H, W] with target size
        """
        # pylint: disable=import-outside-toplevel
        import torchvision.transforms.functional as TF
        
        # Convert to numpy if needed
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        img = np.asarray(img)
        
        # Convert to torch format [C, H, W]
        if len(img.shape) == 3:
            if img.shape[2] == 3:  # [H, W, C]
                img_torch = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            elif img.shape[0] == 3:  # [C, H, W]
                img_torch = torch.from_numpy(img).float() / 255.0
            else:
                # Create black image
                img_torch = torch.zeros(3, self.image_size[0], self.image_size[1])
        else:
            # Create black image
            img_torch = torch.zeros(3, self.image_size[0], self.image_size[1])
        
        # Resize to target size
        img_resized = TF.resize(img_torch, list(self.image_size))
        
        return img_resized.numpy()


def collate_act_batch(batch):
    """Collate function for ACT dataset.
    
    Creates batches in the exact format ACT expects:
    - observation.images: [B, C, H, W]
    - observation.environment_state: [B, state_dim]
    - observation.state: [B, state_dim]
    - action: [B, chunk_size, action_dim]
    - action_is_pad: [B, chunk_size]
    
    Args:
        batch: List of samples from ACTDataset
        
    Returns:
        dict: Batch formatted for ACT
    """
    obs_states = []
    obs_images = []
    actions = []
    action_is_pads = []
    has_images = False
    
    for sample in batch:
        obs_states.append(sample["obs_state"])
        actions.append(sample["actions"])
        action_is_pads.append(sample["action_is_pad"])
        
        if sample["obs_image"] is not None:
            obs_images.append(sample["obs_image"])
            has_images = True
    
    # Stack into tensors
    obs_state_tensor = torch.from_numpy(np.stack(obs_states, axis=0))  # [B, obs_dim]
    action_tensor = torch.from_numpy(np.stack(actions, axis=0))  # [B, chunk_size, action_dim]
    action_is_pad_tensor = torch.from_numpy(np.stack(action_is_pads, axis=0))  # [B, chunk_size]
    
    # Create batch dict
    batch_dict = {
        "observation.environment_state": obs_state_tensor,
        "observation.state": obs_state_tensor,  # Same as environment_state
        "action": action_tensor,
        "action_is_pad": action_is_pad_tensor,
    }
    
    # Add images if available
    if has_images:
        # Stack images [B, C, H, W]
        img_tensor = torch.from_numpy(np.stack(obs_images, axis=0))
        batch_dict["observation.images"] = img_tensor
    else:
        # Create dummy black images
        B = len(batch)
        img_h, img_w = batch[0].get("image_size", (84, 84))
        batch_dict["observation.images"] = torch.zeros(B, 3, img_h, img_w)
    
    return batch_dict

