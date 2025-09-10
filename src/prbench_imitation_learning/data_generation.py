"""
Data generation functionality for creating LeRobot datasets from geom2d environments.
"""

import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from datasets import Features, Array2D, Array3D, Value, Image, Video
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import prbench


def setup_environment():
    """Register all prbench environments."""
    prbench.register_all_environments()


def get_available_environments() -> Dict[str, str]:
    """Get all available PRBench environments."""
    # First register all environments
    setup_environment()
    
    # Get all registered PRBench environment IDs
    all_env_ids = prbench.get_all_env_ids()
    
    # Create a mapping of short names to full IDs
    env_mapping = {}
    
    for env_id in sorted(all_env_ids):
        # Extract a reasonable short name from the full ID
        # e.g., "prbench/Motion2D-p2-v0" -> "motion2d-p2"
        if env_id.startswith("prbench/"):
            short_name = env_id[8:]  # Remove "prbench/" prefix
            short_name = short_name.replace("-v0", "")  # Remove version suffix
            short_name = short_name.lower()  # Convert to lowercase
            env_mapping[short_name] = env_id
    
    return env_mapping


def _save_video_frames(frames: List[np.ndarray], video_path: str):
    """Save video from frames using matplotlib animation."""
    try:
        if not frames:
            print(f"  Warning: No frames to save for video {video_path}")
            return
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        def animate(frame_idx):
            ax.clear()
            ax.imshow(frames[frame_idx])
            ax.axis('off')
            ax.set_title(f"Frame {frame_idx + 1}/{len(frames)}")
            return []
        
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                     interval=50, blit=False, repeat=False)
        
        # Save as GIF (more compatible than MP4)
        if video_path.endswith('.mp4'):
            video_path = video_path.replace('.mp4', '.gif')
        elif not video_path.endswith('.gif'):
            video_path += '.gif'
            
        anim.save(video_path, writer='pillow', fps=20)
        plt.close(fig)
        
    except Exception as e:
        print(f"  Warning: Failed to save video {video_path}: {e}")


def create_dataset_features(env: gym.Env, image_height: int = 256, image_width: int = 256) -> Dict[str, Any]:
    """Create LeRobot dataset features based on environment specifications."""
    
    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Extract state and image dimensions
    if isinstance(obs_space, gym.spaces.Dict):
        if 'state' in obs_space.spaces:
            state_dim = obs_space.spaces['state'].shape[0]
        else:
            state_dim = 4  # Default fallback
            
        if 'image' in obs_space.spaces:
            image_shape = obs_space.spaces['image'].shape
        else:
            image_shape = (image_height, image_width, 3)  # Default RGB
    else:
        # Fallback for non-dict observation spaces
        state_dim = obs_space.shape[0] if hasattr(obs_space, 'shape') else 4
        image_shape = (image_height, image_width, 3)
    
    # Action dimension
    if isinstance(action_space, gym.spaces.Box):
        action_dim = action_space.shape[0]
    else:
        action_dim = 2  # Default fallback
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Image shape: {image_shape}")
    
    # Define dataset features
    features = Features({
        "observation.state": Array2D(shape=(state_dim,), dtype="float32"),
        "observation.image": Array3D(shape=image_shape, dtype="uint8"),
        "action": Array2D(shape=(action_dim,), dtype="float32"),
        "episode_index": Value("int64"),
        "frame_index": Value("int64"),
        "timestamp": Value("float32"),
        "next.reward": Value("float32"),
        "next.done": Value("bool"),
    })
    
    return features


def generate_expert_trajectory(env: gym.Env, max_steps: int = 1000, save_video: bool = False, 
                             video_path: str = None) -> Tuple[List[Dict], bool]:
    """Generate a single expert trajectory using the environment's expert policy.
    
    Args:
        env: The environment to generate trajectory from
        max_steps: Maximum number of steps per trajectory
        save_video: Whether to save video of the trajectory
        video_path: Path to save the video (required if save_video=True)
    
    Returns:
        Tuple of (trajectory, success_flag)
    """
    trajectory = []
    frames = [] if save_video else None
    
    try:
        obs, info = env.reset()
        step_count = 0
        total_reward = 0.0
        
        # Capture initial frame if saving video
        if save_video:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass  # Skip if rendering fails
        
        while step_count < max_steps:
            # Use expert policy if available
            if hasattr(env, 'get_expert_action'):
                try:
                    action = env.get_expert_action()
                except:
                    # Fallback to random action if expert policy fails
                    action = env.action_space.sample()
            else:
                # Fallback to random action
                action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            transition = {
                'obs': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done,
                'info': info.copy(),
                'next_info': next_info.copy()
            }
            trajectory.append(transition)
            
            # Capture frame if saving video
            if save_video:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except:
                    pass  # Skip if rendering fails
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
                
            obs = next_obs
            info = next_info
        
        # Save video if requested
        if save_video and frames and video_path:
            _save_video_frames(frames, video_path)
            print(f"  Video saved to: {video_path}")
        
        # Consider successful if environment indicates success or positive reward
        success = next_info.get('success', total_reward > 0)
        print(f"  Generated trajectory: {len(trajectory)} steps, reward: {total_reward:.2f}, success: {success}")
        
        return trajectory, success
        
    except Exception as e:
        print(f"  Error generating trajectory: {e}")
        return [], False


def generate_random_trajectory(env: gym.Env, max_steps: int = 1000, save_video: bool = False, 
                             video_path: str = None) -> Tuple[List[Dict], bool]:
    """Generate a single random trajectory.
    
    Args:
        env: The environment to generate trajectory from
        max_steps: Maximum number of steps per trajectory
        save_video: Whether to save video of the trajectory
        video_path: Path to save the video (required if save_video=True)
    
    Returns:
        Tuple of (trajectory, success_flag)
    """
    trajectory = []
    frames = [] if save_video else None
    
    try:
        obs, info = env.reset()
        step_count = 0
        total_reward = 0.0
        
        # Capture initial frame if saving video
        if save_video:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass  # Skip if rendering fails
        
        while step_count < max_steps:
            # Random action
            action = env.action_space.sample()
            
            # Take step
            next_obs, reward, terminated, truncated, next_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            transition = {
                'obs': obs.copy(),
                'action': action.copy(),
                'reward': reward,
                'next_obs': next_obs.copy(),
                'done': done,
                'info': info.copy(),
                'next_info': next_info.copy()
            }
            trajectory.append(transition)
            
            # Capture frame if saving video
            if save_video:
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(frame)
                except:
                    pass  # Skip if rendering fails
            
            total_reward += reward
            step_count += 1
            
            if done:
                break
                
            obs = next_obs
            info = next_info
        
        # Save video if requested
        if save_video and frames and video_path:
            _save_video_frames(frames, video_path)
            print(f"  Video saved to: {video_path}")
        
        success = next_info.get('success', False)  # Random trajectories are rarely successful
        print(f"  Generated trajectory: {len(trajectory)} steps, reward: {total_reward:.2f}, success: {success}")
        
        return trajectory, success
        
    except Exception as e:
        print(f"  Error generating trajectory: {e}")
        return [], False


def convert_trajectory_to_dataset_format(trajectory: List[Dict], episode_idx: int, 
                                       start_frame_idx: int) -> List[Dict]:
    """Convert trajectory to LeRobot dataset format."""
    dataset_episodes = []
    
    for i, transition in enumerate(trajectory):
        obs = transition['obs']
        action = transition['action']
        reward = transition['reward']
        done = transition['done']
        
        # Handle different observation formats
        if isinstance(obs, dict):
            state = obs.get('state', np.zeros(4, dtype=np.float32))
            image = obs.get('image', np.zeros((256, 256, 3), dtype=np.uint8))
        else:
            # Assume obs is the state directly
            state = obs.astype(np.float32)
            image = np.zeros((256, 256, 3), dtype=np.uint8)  # Dummy image
        
        # Ensure correct dtypes and shapes
        if len(state.shape) == 0:
            state = np.array([state], dtype=np.float32)
        else:
            state = state.astype(np.float32)
            
        if len(action.shape) == 0:
            action = np.array([action], dtype=np.float32)
        else:
            action = action.astype(np.float32)
        
        # Ensure image is uint8 and has correct shape
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert grayscale to RGB
        image = image.astype(np.uint8)
        
        episode_data = {
            "observation.state": state,
            "observation.image": image,
            "action": action,
            "episode_index": episode_idx,
            "frame_index": start_frame_idx + i,
            "timestamp": float(i),
            "next.reward": float(reward),
            "next.done": bool(done),
        }
        
        dataset_episodes.append(episode_data)
    
    return dataset_episodes


def generate_lerobot_dataset(env_name: str, dataset_name: str, num_episodes: int, 
                           data_type: str, output_dir: str, log_dir: str = "./logs",
                           max_steps_per_episode: int = 1000, save_videos: bool = False) -> str:
    """Generate LeRobot dataset from environment.
    
    Args:
        env_name: Name of the environment
        dataset_name: Name for the dataset
        num_episodes: Number of episodes to collect
        data_type: Type of data to collect ("expert" or "random")
        output_dir: Directory to save the dataset
        log_dir: Directory for logs
        max_steps_per_episode: Maximum steps per episode
        save_videos: Whether to save video recordings of trajectories
        
    Returns:
        Path to the generated dataset
    """
    
    # Setup logging
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    data_log_path = log_dir / "data_generation.log"
    
    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(data_log_path, 'a') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    
    # Setup environment
    setup_environment()
    available_envs = get_available_environments()
    
    if env_name not in available_envs:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(available_envs.keys())}")
    
    env_id = available_envs[env_name]
    log_message(f"Creating environment: {env_id}")
    
    try:
        env = gym.make(env_id)
    except Exception as e:
        log_message(f"Failed to create environment: {e}")
        raise
    
    # Create output directory
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create videos directory if saving videos
    videos_dir = None
    if save_videos:
        videos_dir = output_path / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
    
    log_message(f"Starting data generation:")
    log_message(f"  Environment: {env_name} ({env_id})")
    log_message(f"  Dataset: {dataset_name}")
    log_message(f"  Episodes: {num_episodes}")
    log_message(f"  Data type: {data_type}")
    log_message(f"  Output: {output_path}")
    
    # Generate trajectories
    all_episodes = []
    successful_episodes = 0
    total_frames = 0
    frame_idx = 0
    
    for episode_idx in range(num_episodes):
        log_message(f"Generating episode {episode_idx + 1}/{num_episodes}")
        
        # Determine video path if saving videos
        video_path = None
        if save_videos and videos_dir:
            video_path = str(videos_dir / f"episode_{episode_idx + 1}_{data_type}.gif")
        
        if data_type == "expert":
            trajectory, success = generate_expert_trajectory(env, max_steps_per_episode, 
                                                           save_video=save_videos, video_path=video_path)
        elif data_type == "random":
            trajectory, success = generate_random_trajectory(env, max_steps_per_episode,
                                                           save_video=save_videos, video_path=video_path)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if trajectory:
            # Convert to dataset format
            episode_data = convert_trajectory_to_dataset_format(trajectory, episode_idx, frame_idx)
            all_episodes.extend(episode_data)
            
            total_frames += len(trajectory)
            frame_idx += len(trajectory)
            
            if success:
                successful_episodes += 1
    
    env.close()
    
    log_message(f"Data generation completed:")
    log_message(f"  Total episodes: {num_episodes}")
    log_message(f"  Successful episodes: {successful_episodes}")
    log_message(f"  Total frames: {total_frames}")
    log_message(f"  Success rate: {successful_episodes/num_episodes:.2%}")
    
    if not all_episodes:
        raise ValueError("No valid episodes generated!")
    
    # Save as pickle format for compatibility
    dataset_dict = {
        'episodes': all_episodes,
        'metadata': {
            'env_name': env_name,
            'env_id': env_id,
            'dataset_name': dataset_name,
            'num_episodes': num_episodes,
            'total_frames': total_frames,
            'successful_episodes': successful_episodes,
            'data_type': data_type,
            'generated_at': time.time()
        }
    }
    
    pickle_path = output_path / "dataset.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(dataset_dict, f)
    
    log_message(f"Dataset saved to: {pickle_path}")
    
    # Also save metadata as JSON for easy inspection
    import json
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(dataset_dict['metadata'], f, indent=2)
    
    log_message(f"Metadata saved to: {metadata_path}")
    
    return str(output_path)
