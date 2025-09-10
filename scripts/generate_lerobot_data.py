#!/usr/bin/env python3
"""
Script to generate LeRobot format datasets from geom2d environments.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from datasets import Features, Array2D, Array3D, Value, Image, Video
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import gymnasium as gym

# Add the prbench module to path
sys.path.append('/home/yixuan/prbench_dir/prbench/src')
import prbench
from prbench.envs.geom2d.motion2d import Motion2DEnv
from prbench.envs.geom2d.pushpullhook2d import PushPullHook2DEnv
from prbench.envs.geom2d.stickbutton2d import StickButton2DEnv
from prbench.envs.geom2d.clutteredretrieval2d import ClutteredRetrieval2DEnv
from prbench.envs.geom2d.clutteredstorage2d import ClutteredStorage2DEnv
from prbench.envs.geom2d.obstruction2d import Obstruction2DEnv


def setup_environment():
    """Register all prbench environments."""
    prbench.register_all_environments()


def get_available_environments() -> Dict[str, str]:
    """Get available geom2d environments."""
    return {
        "motion2d": "prbench/Motion2D-p2-v0",
        "pushpullhook2d": "prbench/PushPullHook2D-v0", 
        "stickbutton2d": "prbench/StickButton2D-b2-v0",
        "clutteredretrieval2d": "prbench/ClutteredRetrieval2D-o10-v0",
        "clutteredstorage2d": "prbench/ClutteredStorage2D-b3-v0",
        "obstruction2d": "prbench/Obstruction2D-o2-v0"
    }


def create_dataset_features(env: gym.Env, image_height: int = 256, image_width: int = 256) -> Dict[str, Any]:
    """Create LeRobot dataset features based on environment specifications."""
    
    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space
    
    # Observation features (state vector) - flatten to 1D
    obs_shape = (obs_space.shape[0],) if len(obs_space.shape) > 0 else (4,)
    observation_state = Array2D(dtype="float32", shape=obs_shape)
    
    # Action features (5-dimensional for geom2d: dx, dy, dtheta, darm, vacuum)
    action_shape = (action_space.shape[0],) if len(action_space.shape) > 0 else (5,)
    action = Array2D(dtype="float32", shape=action_shape)
    
    # Image observations (rendered frames)
    observation_image = Image()
    
    # Episode and timestamp information
    episode_index = Value("int64")
    frame_index = Value("int64")
    timestamp = Value("float32")
    next_done = Value("bool")
    
    features = {
        "observation.state": observation_state,
        "observation.image": observation_image,
        "action": action,
        "episode_index": episode_index,
        "frame_index": frame_index,
        "timestamp": timestamp,
        "next.done": next_done,
    }
    
    return features


def collect_random_episodes(env: gym.Env, num_episodes: int = 10, max_steps_per_episode: int = 200) -> List[Dict[str, Any]]:
    """Collect random episodes from the environment."""
    episodes = []
    
    for episode_idx in range(num_episodes):
        print(f"Collecting episode {episode_idx + 1}/{num_episodes}")
        
        episode_data = []
        obs, info = env.reset()
        done = False
        step_idx = 0
        
        while not done and step_idx < max_steps_per_episode:
            # Sample random action
            action = env.action_space.sample()
            
            # Render frame
            image = env.render()
            
            # Store step data
            step_data = {
                "observation.state": obs.astype(np.float32),
                "observation.image": image,
                "action": action.astype(np.float32),
                "episode_index": episode_idx,
                "frame_index": step_idx,
                "timestamp": step_idx * 0.1,  # Assume 10 FPS
                "next.done": False,
            }
            
            episode_data.append(step_data)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1
        
        # Mark the last step as done
        if episode_data:
            episode_data[-1]["next.done"] = True
            
        episodes.extend(episode_data)
        print(f"Episode {episode_idx + 1} completed with {len(episode_data)} steps")
    
    return episodes


def collect_expert_episodes(env: gym.Env, num_episodes: int = 10, max_steps_per_episode: int = 200) -> List[Dict[str, Any]]:
    """Collect expert-like episodes using simple heuristics."""
    episodes = []
    
    for episode_idx in range(num_episodes):
        print(f"Collecting expert episode {episode_idx + 1}/{num_episodes}")
        
        episode_data = []
        obs, info = env.reset()
        done = False
        step_idx = 0
        
        while not done and step_idx < max_steps_per_episode:
            # Use simple heuristic policy
            action = get_heuristic_action(env, obs)
            
            # Render frame
            image = env.render()
            
            # Store step data
            step_data = {
                "observation.state": obs.astype(np.float32),
                "observation.image": image,
                "action": action.astype(np.float32),
                "episode_index": episode_idx,
                "frame_index": step_idx,
                "timestamp": step_idx * 0.1,  # Assume 10 FPS
                "next.done": False,
            }
            
            episode_data.append(step_data)
            
            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_idx += 1
        
        # Mark the last step as done
        if episode_data:
            episode_data[-1]["next.done"] = True
            
        episodes.extend(episode_data)
        print(f"Expert episode {episode_idx + 1} completed with {len(episode_data)} steps")
    
    return episodes


def get_heuristic_action(env: gym.Env, obs: np.ndarray) -> np.ndarray:
    """Get heuristic action based on environment and observation."""
    # Extract robot position from observation (first 2 elements typically x, y)
    robot_x, robot_y = obs[0], obs[1]
    
    # Simple heuristic: move towards the right side of the environment
    target_x = 2.0  # Approximate target position
    target_y = 1.0
    
    # Calculate desired movement
    dx = np.clip((target_x - robot_x) * 0.1, -0.05, 0.05)
    dy = np.clip((target_y - robot_y) * 0.1, -0.05, 0.05)
    dtheta = 0.0  # No rotation
    darm = 0.0    # No arm movement
    vacuum = 1.0 if np.random.random() > 0.8 else 0.0  # Occasionally turn on vacuum
    
    action = np.array([dx, dy, dtheta, darm, vacuum], dtype=np.float32)
    
    # Clip to action space bounds
    action = np.clip(action, env.action_space.low, env.action_space.high)
    
    return action


def generate_dataset(env_name: str, dataset_name: str, num_episodes: int = 10, 
                    data_type: str = "random", output_dir: str = "./lerobot_datasets"):
    """Generate a LeRobot dataset from a geom2d environment."""
    
    # Setup environment registry
    setup_environment()
    
    # Get environment ID
    env_map = get_available_environments()
    if env_name not in env_map:
        raise ValueError(f"Environment {env_name} not available. Choose from: {list(env_map.keys())}")
    
    env_id = env_map[env_name]
    
    # Create environment
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, render_mode="rgb_array")
    
    # Create output directory
    output_path = Path(output_dir) / dataset_name
    if output_path.exists():
        import shutil
        print(f"Removing existing dataset directory: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect episodes first
    if data_type == "random":
        episodes = collect_random_episodes(env, num_episodes)
    elif data_type == "expert":
        episodes = collect_expert_episodes(env, num_episodes)
    else:
        raise ValueError(f"Data type {data_type} not supported. Choose 'random' or 'expert'")
    
    print(f"Collected {len(episodes)} steps from {num_episodes} episodes")
    
    # Create dataset features based on collected data
    if episodes:
        sample_data = episodes[0]
        obs_shape = sample_data["observation.state"].shape
        action_shape = sample_data["action"].shape
        
        features = {
            "observation.state": Array2D(dtype="float32", shape=obs_shape),
            "observation.image": Image(),
            "action": Array2D(dtype="float32", shape=action_shape),
            "episode_index": Value("int64"),
            "frame_index": Value("int64"),
            "timestamp": Value("float32"),
            "next.done": Value("bool"),
        }
        
        # Create LeRobot dataset
        print(f"Creating LeRobot dataset: {dataset_name}")
        try:
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=10,  # 10 FPS
                features=features,
                root=str(output_path),
                robot_type="geom2d_robot",
                use_videos=False,  # Use individual images for now
                image_writer_threads=1,  # Reduce threads to avoid issues
            )
            
            # Add episodes to dataset
            print(f"Adding {len(episodes)} steps to dataset...")
            for step_data in episodes:
                dataset.add_frame(step_data)
            
            # Save dataset
            print("Saving dataset...")
            dataset.consolidate()
            
        except Exception as e:
            print(f"LeRobot dataset creation failed: {e}")
            print("Falling back to simple numpy format...")
            
            # Fallback: save as numpy arrays
            import pickle
            data_dict = {
                'episodes': episodes,
                'env_id': env_id,
                'num_episodes': num_episodes,
                'data_type': data_type,
                'obs_shape': obs_shape,
                'action_shape': action_shape,
            }
            
            with open(output_path / 'dataset.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
            
            print(f"Dataset saved as pickle format to: {output_path}")
    
    print(f"Dataset contains {len(episodes)} steps across {num_episodes} episodes")
    
    # Clean up
    env.close()
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate LeRobot format data from geom2d environments")
    parser.add_argument("--env", type=str, default="motion2d", 
                       choices=list(get_available_environments().keys()),
                       help="Environment name")
    parser.add_argument("--dataset-name", type=str, default="geom2d_motion2d_random",
                       help="Dataset name")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes to collect")
    parser.add_argument("--data-type", type=str, default="random", choices=["random", "expert"],
                       help="Type of data to collect")
    parser.add_argument("--output-dir", type=str, default="./lerobot_datasets",
                       help="Output directory for datasets")
    
    args = parser.parse_args()
    
    print("="*50)
    print("LeRobot Data Generation for Geom2D Environments")
    print("="*50)
    print(f"Environment: {args.env}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Data type: {args.data_type}")
    print(f"Output directory: {args.output_dir}")
    print("="*50)
    
    try:
        dataset_path = generate_dataset(
            env_name=args.env,
            dataset_name=args.dataset_name,
            num_episodes=args.num_episodes,
            data_type=args.data_type,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*50)
        print("SUCCESS: Dataset generation completed!")
        print(f"Dataset path: {dataset_path}")
        print("="*50)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
