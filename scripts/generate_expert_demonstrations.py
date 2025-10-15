#!/usr/bin/env python
"""Expert demonstration generation script for PRBench environments.

This script generates expert demonstrations using BilevelPlanningAgent for geom2d environments.
The generated data can be used for imitation learning training.

Usage examples:
    # Generate expert demos for Motion2D
    python scripts/generate_expert_demonstrations.py \
        --expert_env=motion2d \
        --expert_env_param=2 \
        --expert_episodes=50 \
        --output_dir=./expert_data/motion2d_p2

    # Generate expert demos for StickButton2D with videos
    python scripts/generate_expert_demonstrations.py \
        --expert_env=stickbutton2d \
        --expert_env_param=3 \
        --expert_episodes=100 \
        --expert_save_videos \
        --output_dir=./expert_data/stickbutton_p3

    # Generate expert demos for Obstruction2D with custom settings
    python scripts/generate_expert_demonstrations.py \
        --expert_env=obstruction2d \
        --expert_env_param=2 \
        --expert_episodes=75 \
        --expert_max_steps=3000 \
        --expert_planning_timeout=60.0 \
        --output_dir=./expert_data/obstruction_p2

    # Generate expert demos for ClutteredRetrieval2D with all options
    python scripts/generate_expert_demonstrations.py \
        --expert_env=clutteredretrieval2d \
        --expert_env_param=5 \
        --expert_episodes=200 \
        --expert_max_steps=10000 \
        --expert_save_videos \
        --expert_max_abstract_plans=15 \
        --expert_samples_per_step=5 \
        --expert_planning_timeout=60.0 \
        --seed=42
"""

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np

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


def collect_expert_demonstrations(
    env_name: str,
    env_param: int,
    num_episodes: int = 50,
    output_dir: str = "./expert_demonstrations",
    max_steps_per_episode: int = 5000,
    save_videos: bool = False,
    max_abstract_plans: int = 10,
    samples_per_step: int = 3,
    planning_timeout: float = 30.0,
    seed: int = 123,
    set_random_seed: bool = False,
) -> str:
    """Collect expert demonstrations using BilevelPlanningAgent.

    Args:
        env_name: Environment name (e.g., 'motion2d', 'stickbutton2d', 'obstruction2d')
        env_param: Environment parameter (passages, buttons, obstructions, etc.)
        num_episodes: Number of episodes to collect
        output_dir: Directory to save the dataset
        max_steps_per_episode: Maximum steps per episode
        save_videos: Whether to save video recordings
        max_abstract_plans: Max abstract plans for the agent
        samples_per_step: Samples per planning step
        planning_timeout: Timeout for motion planning
        seed: Random seed
        set_random_seed: Whether to use specific seeds for environment resets

    Returns:
        Path to the generated dataset directory
    """
    import prbench
    from gymnasium.wrappers import RecordVideo
    from prbench_bilevel_planning.agent import BilevelPlanningAgent
    from prbench_bilevel_planning.env_models import create_bilevel_planning_models

    # Map environment names to their full IDs and parameter names
    env_mapping = {
        "motion2d": {
            "env_id": f"prbench/Motion2D-p{env_param}-v0",
            "param_name": "num_passages",
        },
        "stickbutton2d": {
            "env_id": f"prbench/StickButton2D-b{env_param}-v0",
            "param_name": "num_buttons",
        },
        "obstruction2d": {
            "env_id": f"prbench/Obstruction2D-o{env_param}-v0",
            "param_name": "num_obstructions",
        },
        "clutteredstorage2d": {
            "env_id": f"prbench/ClutteredStorage2D-b{env_param}-v0",
            "param_name": "num_blocks",
        },
        "clutteredretrieval2d": {
            "env_id": f"prbench/ClutteredRetrieval2D-o{env_param}-v0",
            "param_name": "num_obstructions",
        },
    }

    if env_name not in env_mapping:
        raise ValueError(
            f"Unsupported environment: {env_name}. "
            f"Supported: {list(env_mapping.keys())}"
        )

    env_info = env_mapping[env_name]
    full_env_id = env_info["env_id"]
    param_name = env_info["param_name"]

    logging.info(f"Collecting expert demonstrations for: {full_env_id}")
    logging.info(f"  Episodes: {num_episodes}")
    logging.info(f"  Max steps per episode: {max_steps_per_episode}")

    # Setup environment
    prbench.register_all_environments()
    render_mode = "rgb_array" if save_videos else None
    env = prbench.make(full_env_id, render_mode=render_mode)

    # Setup video recording if requested
    if save_videos:
        videos_dir = Path(output_dir) / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            str(videos_dir),
            name_prefix=f"{env_name}-{env_param}-expert",
            episode_trigger=lambda episode_id: True,
        )

    # Create environment models and agent
    env_models = create_bilevel_planning_models(
        env_name,
        env.observation_space,
        env.action_space,
        **{param_name: env_param},
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
        logging.info(f"Collecting episode {episode_idx + 1}/{num_episodes}")

        # Wrap the per-episode collection block in a try/except. On failure or timeout, skip to next episode.
        episode_start_time = time.perf_counter()
        try:
            trajectory = []
            reset_seed = seed if set_random_seed else np.random.randint(0, 1000000)
            # Reset env; if this fails, skip episode
            try:
                obs, info = env.reset(seed=reset_seed)
            except Exception as e:
                logging.warning(f"  Env reset failed: {e}")
                raise
            # Initialize planner; if planning fails (e.g., no plan), skip episode
            try:
                agent.reset(obs, info)
            except Exception as e:
                logging.warning(f"  Agent reset/planning failed: {e}")
                raise

            step_count = 0
            episode_reward = 0.0

            while step_count < max_steps_per_episode:
                # Abort the episode if a time limit is exceeded
                if (time.perf_counter() - episode_start_time) > planning_timeout:
                    logging.warning(
                        f"  Episode time limit exceeded ({planning_timeout:.1f}s); skipping episode {episode_idx + 1}."
                    )
                    raise TimeoutError("Episode time limit exceeded")

                try:
                    action = agent.step()
                    next_obs, reward, terminated, truncated, next_info = env.step(
                        action
                    )
                    done = terminated or truncated

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
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                    trajectory.append(transition)

                    agent.update(next_obs, reward, done, next_info)
                    episode_reward += float(reward)
                    step_count += 1

                    if done:
                        break

                    obs = next_obs
                    info = next_info

                except Exception as e:
                    logging.warning(f"  Error during step {step_count}: {e}")
                    # Propagate up to outer handler to skip current episode entirely
                    raise

        except Exception as e:
            logging.warning(
                f"  Skipping episode {episode_idx + 1} due to failure/timeout: {e}"
            )
            episode_idx -= 1
            continue

        # Check success
        success = next_info.get("success", False) if "next_info" in locals() else False
        if not success and len(trajectory) > 0:
            last_transition = trajectory[-1]
            if (
                last_transition.get("terminated", False)
                and not last_transition.get("truncated", False)
                and step_count < max_steps_per_episode
            ):
                success = True

        logging.info(
            f"  Episode {episode_idx + 1}: {len(trajectory)} steps, "
            f"reward: {episode_reward:.2f}, success: {success}"
        )

        if trajectory:
            # Convert to LeRobot format
            episode_data = []
            for i, transition in enumerate(trajectory):
                obs = transition["obs"]
                action = transition["action"]
                reward = transition["reward"]
                done = transition["done"]

                # Handle observation format
                if isinstance(obs, dict):
                    state = obs.get(
                        "state",
                        obs.get("observation", np.zeros(4, dtype=np.float32)),
                    )
                    image = obs.get("image", np.zeros((64, 64, 3), dtype=np.uint8))
                else:
                    state = obs.astype(np.float32)
                    image = np.zeros((64, 64, 3), dtype=np.uint8)

                # Ensure correct dtypes and shapes
                if len(state.shape) == 0:
                    state = np.array([state], dtype=np.float32)
                else:
                    state = state.astype(np.float32)

                if len(action.shape) == 0:
                    action = np.array([action], dtype=np.float32)
                else:
                    action = action.astype(np.float32)

                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=-1)
                image = image.astype(np.uint8)

                episode_step = {
                    "observation.state": state,
                    "observation.image": image,
                    "action": action,
                    "episode_index": episode_idx,
                    "frame_index": frame_idx + i,
                    "timestamp": float(i),
                    "next.reward": float(reward),
                    "next.done": bool(done),
                }

                episode_data.append(episode_step)

            all_episodes.extend(episode_data)
            total_frames += len(trajectory)
            total_reward += episode_reward
            frame_idx += len(trajectory)

            if success:
                successful_episodes += 1

    env.close()

    logging.info("Expert demonstration collection completed:")
    logging.info(f"  Total episodes: {num_episodes}")
    logging.info(f"  Successful episodes: {successful_episodes}")
    logging.info(f"  Total frames: {total_frames}")
    logging.info(f"  Average reward: {total_reward/num_episodes:.2f}")
    logging.info(f"  Success rate: {successful_episodes/num_episodes:.2%}")

    if not all_episodes:
        raise ValueError("No valid episodes collected!")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save dataset
    dataset_dict = {
        "episodes": all_episodes,
        "metadata": {
            "env_name": full_env_id,
            "env_type": env_name,
            "env_param": env_param,
            "dataset_name": f"{env_name}_{env_param}_expert",
            "num_episodes": num_episodes,
            "total_frames": total_frames,
            "successful_episodes": successful_episodes,
            "success_rate": successful_episodes / num_episodes,
            "average_reward": total_reward / num_episodes,
            "data_type": "expert",
            "agent_type": "BilevelPlanningAgent",
            "generated_at": time.time(),
        },
    }

    # Save as pickle
    pickle_path = output_path / "dataset.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(dataset_dict, f)

    # Save metadata as JSON
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict["metadata"], f, indent=2)

    logging.info(f"Dataset saved to: {output_path}")
    return str(output_path)


def main():
    """Main entry point for expert demonstration generation."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Parse arguments
    arg_parser = argparse.ArgumentParser(
        description="Generate expert demonstrations for PRBench environments using BilevelPlanningAgent"
    )

    arg_parser.add_argument(
        "--expert_env",
        type=str,
        choices=[
            "motion2d",
            "stickbutton2d",
            "obstruction2d",
            "clutteredstorage2d",
            "clutteredretrieval2d",
        ],
        required=True,
        help="Environment for expert demonstration generation",
    )
    arg_parser.add_argument(
        "--expert_env_param",
        type=int,
        required=True,
        help="Environment parameter (passages, buttons, obstructions, etc.)",
    )
    arg_parser.add_argument(
        "--expert_episodes",
        type=int,
        default=50,
        help="Number of expert episodes to generate (default: 50)",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the dataset (default: auto-generated based on env and timestamp)",
    )
    arg_parser.add_argument(
        "--expert_max_steps",
        type=int,
        default=5000,
        help="Maximum steps per expert episode (default: 5000)",
    )
    arg_parser.add_argument(
        "--expert_save_videos",
        action="store_true",
        help="Save videos during expert demonstration collection",
    )
    arg_parser.add_argument(
        "--expert_max_abstract_plans",
        type=int,
        default=10,
        help="Maximum abstract plans for BilevelPlanningAgent (default: 10)",
    )
    arg_parser.add_argument(
        "--expert_samples_per_step",
        type=int,
        default=3,
        help="Samples per step for BilevelPlanningAgent (default: 3)",
    )
    arg_parser.add_argument(
        "--expert_planning_timeout",
        type=float,
        default=30.0,
        help="Planning timeout in seconds (default: 30.0)",
    )
    arg_parser.add_argument(
        "--expert_set_random_seed",
        action="store_true",
        help="Use specific random seeds for environment resets",
    )
    arg_parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility (default: 123)",
    )

    args = arg_parser.parse_args()

    # Generate output directory name if not provided
    if args.output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"./expert_data/{args.expert_env}_p{args.expert_env_param}_{timestamp}"

    logging.info("=" * 80)
    logging.info("EXPERT DEMONSTRATION GENERATION")
    logging.info("=" * 80)

    try:
        dataset_path = collect_expert_demonstrations(
            env_name=args.expert_env,
            env_param=args.expert_env_param,
            num_episodes=args.expert_episodes,
            output_dir=args.output_dir,
            max_steps_per_episode=args.expert_max_steps,
            save_videos=args.expert_save_videos,
            max_abstract_plans=args.expert_max_abstract_plans,
            samples_per_step=args.expert_samples_per_step,
            planning_timeout=args.expert_planning_timeout,
            seed=args.seed,
            set_random_seed=args.expert_set_random_seed,
        )

        logging.info("=" * 80)
        logging.info("Expert demonstration generation completed!")
        logging.info(f"Dataset saved to: {dataset_path}")
        logging.info("=" * 80)

        logging.info("\nDataset files:")
        logging.info(f"  - {dataset_path}/dataset.pkl (pickled dataset)")
        logging.info(f"  - {dataset_path}/metadata.json (dataset metadata)")
        if args.expert_save_videos:
            logging.info(f"  - {dataset_path}/videos/ (recorded videos)")

        logging.info("\nYou can use this dataset for training with LeRobot or other imitation learning frameworks.")

    except Exception as e:
        logging.error(f"Failed to generate expert demonstrations: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
