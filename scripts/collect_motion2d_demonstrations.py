#!/usr/bin/env python3
"""Collect demonstrations from Motion2D environment using BilevelPlanningAgent.

This script collects expert demonstrations from the Motion2D environment using the
BilevelPlanningAgent and saves them in lerobot format for training diffusion policies.
"""

import argparse
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import prbench
from gymnasium.wrappers import RecordVideo

# Add third-party modules to path for bilevel planning
script_dir = Path(__file__).parent
project_root = script_dir.parent
third_party_bilevel = (
    project_root
    / "third-party"
    / "prbench-bilevel-planning"
    / "third-party"
    / "bilevel-planning"
    / "src"
)
third_party_prbench_bilevel = (
    project_root / "third-party" / "prbench-bilevel-planning" / "src"
)
third_party_prbench_models = (
    project_root
    / "third-party"
    / "prbench-bilevel-planning"
    / "third-party"
    / "prbench-models"
    / "src"
)

if third_party_bilevel.exists():
    sys.path.insert(0, str(third_party_bilevel))
if third_party_prbench_bilevel.exists():
    sys.path.insert(0, str(third_party_prbench_bilevel))
if third_party_prbench_models.exists():
    sys.path.insert(0, str(third_party_prbench_models))

# pylint: disable=wrong-import-position,import-error
from prbench_bilevel_planning.agent import BilevelPlanningAgent  # pylint: disable=import-error
from prbench_bilevel_planning.env_models import create_bilevel_planning_models  # pylint: disable=import-error


def setup_environment():
    """Register all prbench environments."""
    prbench.register_all_environments()


def collect_expert_trajectory(
    env,
    agent: BilevelPlanningAgent,
    max_steps: int = 5000,
) -> tuple[List[Dict[str, Any]], bool, float]:
    """Collect a single expert trajectory using BilevelPlanningAgent.

    Args:
        env: The Motion2D environment
        agent: The BilevelPlanningAgent
        max_steps: Maximum number of steps per trajectory

    Returns:
        Tuple of (trajectory, success_flag, total_reward)
    """
    trajectory: List[Dict[str, Any]] = []

    try:
        obs, info = env.reset()
        agent.reset(obs, info)

        step_count = 0
        total_reward = 0.0

        while step_count < max_steps:
            try:
                # Get action from expert agent
                action = agent.step()

                # Take step in environment
                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = terminated or truncated

                # Store transition
                transition = {
                    "obs": obs.copy(),
                    "action": (
                        action.copy()
                        if hasattr(action, "copy")
                        else np.array(action, dtype=np.float32)
                    ),
                    "reward": float(reward),
                    "next_obs": next_obs.copy(),
                    "done": done,
                    "info": info.copy() if hasattr(info, "copy") else dict(info),
                    "next_info": (
                        next_info.copy()
                        if hasattr(next_info, "copy")
                        else dict(next_info)
                    ),
                }
                trajectory.append(transition)

                # Update agent
                agent.update(next_obs, reward, done, next_info)

                total_reward += float(reward)
                step_count += 1

                if done:
                    break

                obs = next_obs
                info = next_info

            except Exception as e:
                print(f"  Error during trajectory collection: {e}")
                break

        # Check success
        success = next_info.get("success", False) if "next_info" in locals() else False

        print(
            f"  Generated trajectory: {len(trajectory)} steps, "
            f"reward: {total_reward:.2f}, success: {success}"
        )

        return trajectory, success, total_reward

    except Exception as e:
        print(f"  Error generating expert trajectory: {e}")
        return [], False, 0.0


def convert_trajectory_to_lerobot_format(
    trajectory: List[Dict], episode_idx: int, start_frame_idx: int
) -> List[Dict]:
    """Convert trajectory to lerobot dataset format."""
    dataset_episodes = []

    for i, transition in enumerate(trajectory):
        obs = transition["obs"]
        action = transition["action"]
        reward = transition["reward"]
        done = transition["done"]

        # Handle Motion2D observation format
        if isinstance(obs, dict):
            # Motion2D provides dict observations
            state = obs.get(
                "state", obs.get("observation", np.zeros(4, dtype=np.float32))
            )
            # For Motion2D, we might not have images, so create a dummy image
            image = obs.get("image", np.zeros((64, 64, 3), dtype=np.uint8))
        else:
            # Assume obs is the state directly
            state = obs.astype(np.float32)
            image = np.zeros((64, 64, 3), dtype=np.uint8)  # Dummy image

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


def collect_motion2d_demonstrations(
    num_passages: int = 2,
    num_episodes: int = 10,
    output_dir: str = "./motion2d_demonstrations",
    max_steps_per_episode: int = 5000,
    save_videos: bool = False,
    max_abstract_plans: int = 10,
    samples_per_step: int = 3,
    planning_timeout: float = 30.0,
    seed: int = 123,
) -> str:
    """Collect expert demonstrations from Motion2D environment.

    Args:
        num_passages: Number of passages in Motion2D environment (1, 2, or 3)
        num_episodes: Number of episodes to collect
        output_dir: Directory to save the dataset
        max_steps_per_episode: Maximum steps per episode
        save_videos: Whether to save video recordings
        max_abstract_plans: Max abstract plans for the agent
        samples_per_step: Samples per planning step
        planning_timeout: Timeout for motion planning
        seed: Random seed

    Returns:
        Path to the generated dataset
    """

    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "demonstration_collection.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    # Setup environment
    setup_environment()
    env_name = f"prbench/Motion2D-p{num_passages}-v0"

    log_message("Starting expert demonstration collection:")
    log_message(f"  Environment: {env_name}")
    log_message(f"  Episodes: {num_episodes}")
    log_message(f"  Max steps per episode: {max_steps_per_episode}")
    log_message(f"  Planning timeout: {planning_timeout}s")

    # Create environment
    render_mode = "rgb_array" if save_videos else None
    env = prbench.make(env_name, render_mode=render_mode)

    # Setup video recording if requested
    if save_videos:
        videos_dir = Path(output_dir) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            str(videos_dir),
            name_prefix=f"Motion2D-p{num_passages}-expert",
            episode_trigger=lambda episode_id: True,  # Record every episode
        )

    # Create environment models and agent
    env_models = create_bilevel_planning_models(
        "motion2d",
        env.observation_space,
        env.action_space,
        num_passages=num_passages,
    )

    agent = BilevelPlanningAgent(
        env_models,
        seed=seed,
        max_abstract_plans=max_abstract_plans,
        samples_per_step=samples_per_step,
        planning_timeout=planning_timeout,
    )

    # Collect trajectories
    all_episodes = []
    successful_episodes = 0
    total_frames = 0
    total_reward = 0.0
    frame_idx = 0

    for episode_idx in range(num_episodes):
        log_message(f"Collecting episode {episode_idx + 1}/{num_episodes}")

        try:
            trajectory, success, episode_reward = collect_expert_trajectory(
                env,
                agent,
                max_steps=max_steps_per_episode,
            )

            if trajectory:
                # Convert to lerobot format
                episode_data = convert_trajectory_to_lerobot_format(
                    trajectory, episode_idx, frame_idx
                )
                all_episodes.extend(episode_data)

                total_frames += len(trajectory)
                total_reward += episode_reward
                frame_idx += len(trajectory)

                if success:
                    successful_episodes += 1

        except Exception as e:
            log_message(f"  Failed to collect episode {episode_idx + 1}: {e}")
            continue

    env.close()

    log_message("Expert demonstration collection completed:")
    log_message(f"  Total episodes: {num_episodes}")
    log_message(f"  Successful episodes: {successful_episodes}")
    log_message(f"  Total frames: {total_frames}")
    log_message(f"  Average reward: {total_reward/num_episodes:.2f}")
    log_message(f"  Success rate: {successful_episodes/num_episodes:.2%}")

    if not all_episodes:
        raise ValueError("No valid episodes collected!")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    dataset_dict = {
        "episodes": all_episodes,
        "metadata": {
            "env_name": env_name,
            "num_passages": num_passages,
            "dataset_name": f"motion2d_p{num_passages}_expert",
            "num_episodes": num_episodes,
            "total_frames": total_frames,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / num_episodes,
            "average_reward": total_reward / num_episodes,
            "data_type": "expert",
            "agent_type": "BilevelPlanningAgent",
            "agent_config": {
                "max_abstract_plans": max_abstract_plans,
                "samples_per_step": samples_per_step,
                "planning_timeout": planning_timeout,
                "seed": seed,
            },
            "generated_at": time.time(),
        },
    }

    # Save as pickle format for compatibility with DiffusionPolicyDataset
    pickle_path = output_path / "dataset.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(dataset_dict, f)

    log_message(f"Dataset saved to: {pickle_path}")

    # Save metadata as JSON for easy inspection
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict["metadata"], f, indent=2)

    log_message(f"Metadata saved to: {metadata_path}")

    return str(output_path)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations from Motion2D environment"
    )
    parser.add_argument(
        "--num_passages",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Number of passages in Motion2D environment",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to collect"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./motion2d_demonstrations",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--max_steps", type=int, default=5000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Save video recordings of trajectories",
    )
    parser.add_argument(
        "--max_abstract_plans",
        type=int,
        default=10,
        help="Maximum abstract plans for agent",
    )
    parser.add_argument(
        "--samples_per_step", type=int, default=3, help="Samples per planning step"
    )
    parser.add_argument(
        "--planning_timeout",
        type=float,
        default=30.0,
        help="Planning timeout in seconds",
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")

    args = parser.parse_args()

    try:
        dataset_path = collect_motion2d_demonstrations(
            num_passages=args.num_passages,
            num_episodes=args.num_episodes,
            output_dir=args.output_dir,
            max_steps_per_episode=args.max_steps,
            save_videos=args.save_videos,
            max_abstract_plans=args.max_abstract_plans,
            samples_per_step=args.samples_per_step,
            planning_timeout=args.planning_timeout,
            seed=args.seed,
        )

        print("\n✅ Successfully collected demonstrations!")
        print(f"Dataset saved to: {dataset_path}")
        print("\nYou can now use this dataset to train a diffusion policy:")
        print(
            f"python -m prbench_imitation_learning.train --dataset_path {dataset_path}"
        )

    except Exception as e:
        print(f"\n❌ Failed to collect demonstrations: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
