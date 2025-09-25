#!/usr/bin/env python3
"""Complete pipeline script for generating data, training, and evaluating diffusion
policies on geom2d environments."""

import argparse
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
from prbench_imitation_learning import (
    evaluate_policy,
    generate_lerobot_dataset,
    get_available_environments,
    get_default_training_config,
    train_behavior_cloning_policy,
    train_diffusion_policy,
    train_lerobot_diffusion_policy,
)

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

# Expert demonstration collection is always available through generic collector


def collect_geom2d_demonstrations(
    env_name: str,
    env_param: int,
    num_episodes: int = 10,
    output_dir: str = "./geom2d_demonstrations",
    max_steps_per_episode: int = 5000,
    save_videos: bool = False,
    max_abstract_plans: int = 10,
    samples_per_step: int = 3,
    planning_timeout: float = 30.0,
    seed: int = 123,
    set_random_seed: bool = False,
) -> str:
    """Collect expert demonstrations from any geom2d environment using
    BilevelPlanningAgent.

    Args:
        env_name: Environment name (e.g., 'motion2d', 'stickbutton2d', 'obstruction2d')
        env_param: Environment parameter (passages, buttons, obstructions, blocks, etc.)
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
    # Import required modules (inside function to handle optional dependencies)
    # pylint: disable=import-outside-toplevel
    import prbench
    from gymnasium.wrappers import RecordVideo
    from prbench_bilevel_planning.agent import (  # pylint: disable=import-error
        BilevelPlanningAgent,
    )
    from prbench_bilevel_planning.env_models import (  # pylint: disable=import-error
        create_bilevel_planning_models,
    )

    # Setup logging
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{env_name}_demonstration_collection.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    # Setup environment
    prbench.register_all_environments()

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

    log_message("Starting expert demonstration collection:")
    log_message(f"  Environment: {full_env_id}")
    log_message(f"  Episodes: {num_episodes}")
    log_message(f"  Max steps per episode: {max_steps_per_episode}")
    log_message(f"  Planning timeout: {planning_timeout}s")

    # Create environment
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
            episode_trigger=lambda episode_id: True,  # Record every episode
        )

    # Create environment models and agent
    try:
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
    except ImportError as e:
        log_message(f"❌ Import error for {env_name}: {e}")
        log_message("This environment may not have complete bilevel planning support.")
        log_message("Available working environments: motion2d, obstruction2d")
        raise ValueError(
            f"Environment {env_name} is not supported due to missing dependencies. "
            f"Working environments: motion2d, obstruction2d. "
            f"Original error: {e}"
        )

    # Collect trajectories using the same logic as Motion2D
    all_episodes = []
    successful_episodes = 0
    total_frames = 0
    total_reward = 0.0
    frame_idx = 0

    for episode_idx in range(num_episodes):
        log_message(f"Collecting episode {episode_idx + 1}/{num_episodes}")

        try:
            # Use the same trajectory collection logic as Motion2D
            trajectory: list[dict] = []

            # Use specific seed if set_random_seed is True, otherwise use default
            reset_seed = seed if set_random_seed else np.random.randint(0, 1000000)
            obs, info = env.reset(seed=reset_seed)
            agent.reset(obs, info)

            step_count = 0
            episode_reward = 0.0

            while step_count < max_steps_per_episode:
                try:
                    # Get action from expert agent
                    action = agent.step()

                    # Take step in environment
                    next_obs, reward, terminated, truncated, next_info = env.step(
                        action
                    )
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
                        "terminated": terminated,
                        "truncated": truncated,
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

                    episode_reward += float(reward)
                    step_count += 1

                    if done:
                        break

                    obs = next_obs
                    info = next_info

                except Exception as e:
                    log_message(f"  Error during trajectory collection: {e}")
                    break

            # Check success
            # An episode is successful if:
            # 1. Environment explicitly indicates success, OR
            # 2. Episode terminated naturally (not truncated) before max steps
            success = False
            if "next_info" in locals():
                # Check explicit success flag from environment
                success = next_info.get("success", False)

                # If no explicit success flag, infer from termination conditions
                if not success and len(trajectory) > 0:
                    last_transition = trajectory[-1]
                    # Success if terminated (not truncated) before reaching max steps
                    # In gymnasium: terminated=True means task completion,
                    # truncated=True means timeout
                    if (
                        last_transition.get("terminated", False)
                        and not last_transition.get("truncated", False)
                        and step_count < max_steps_per_episode
                    ):
                        success = True

            log_message(
                f"  Generated trajectory: {len(trajectory)} steps, "
                f"reward: {episode_reward:.2f}, success: {success}"
            )

            if trajectory:
                # Convert to lerobot format (same logic as Motion2D)
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

                    # Ensure image is uint8 and has correct shape
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


def load_precomputed_demonstrations(
    demos_dir: str,
    output_dir: str,
    dataset_name: str,
) -> str:
    """Load precomputed demonstrations and convert them to lerobot format.

    Args:
        demos_dir: Directory containing precomputed demonstration files (.p files)
        output_dir: Output directory for the converted dataset
        dataset_name: Name for the dataset

    Returns:
        Path to the converted dataset directory
    """

    demos_path = Path(demos_dir)
    if not demos_path.exists():
        raise ValueError(f"Precomputed demonstrations directory not found: {demos_dir}")

    print(f"Loading precomputed demonstrations from: {demos_dir}")

    # Find all .p files in subdirectories
    demo_files = []
    for subdir in demos_path.iterdir():
        if subdir.is_dir():
            for file_path in subdir.glob("*.p"):
                demo_files.append(file_path)

    if not demo_files:
        raise ValueError(f"No demonstration files found in: {demos_dir}")

    print(f"Found {len(demo_files)} demonstration files")

    # Convert demonstrations to lerobot format
    all_episodes = []
    frame_idx = 0

    for episode_idx, demo_file in enumerate(demo_files):
        print(
            f"Processing demonstration"
            f"{episode_idx + 1}/{len(demo_files)}: {demo_file.name}"
        )

        try:
            with open(demo_file, "rb") as f:
                demo_data = pickle.load(f)

            # Extract trajectory data
            observations = demo_data["observations"]
            actions = demo_data["actions"]
            rewards = demo_data["rewards"]

            # Convert to lerobot format
            episode_data = []
            for i in range(len(actions)):  # actions is one shorter than observations
                obs = observations[i]
                action = actions[i]
                reward = rewards[i]
                done = i == len(actions) - 1  # Last step is done

                # Handle observation format - Motion2D uses numpy arrays directly
                if isinstance(obs, np.ndarray):
                    state = obs.astype(np.float32)
                else:
                    state = np.array(obs, dtype=np.float32)

                # Handle action format
                if isinstance(action, np.ndarray):
                    action = action.astype(np.float32)
                else:
                    action = np.array(action, dtype=np.float32)

                # Create dummy image since Motion2D doesn't have visual observations
                image = np.zeros((64, 64, 3), dtype=np.uint8)

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
            frame_idx += len(episode_data)

        except Exception as e:
            print(f"Error processing {demo_file}: {e}")
            continue

    if not all_episodes:
        raise ValueError("No valid episodes found in demonstration files!")

    print(f"Converted {len(demo_files)} episodes with {len(all_episodes)} total frames")

    # Create output directory
    output_path = Path(output_dir) / dataset_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dataset dictionary
    dataset_dict = {
        "episodes": all_episodes,
        "metadata": {
            "dataset_name": dataset_name,
            "num_episodes": len(demo_files),
            "total_frames": len(all_episodes),
            "data_type": "precomputed",
            "source_dir": str(demos_dir),
            "generated_at": time.time(),
        },
    }

    # Save as pickle format for compatibility with DiffusionPolicyDataset
    pickle_path = output_path / "dataset.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(dataset_dict, f)

    # Save metadata as JSON for easy inspection
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict["metadata"], f, indent=2)

    print(f"Dataset saved to: {output_path}")
    return str(output_path)


def _parse_environment_name(env_name: str) -> tuple[str, int]:
    """Parse environment name to extract type and parameter.

    Args:
        env_name: Environment name like 'motion2d-p1', 'stickbutton2d-b3', etc.

    Returns:
        Tuple of (environment_type, parameter_value)
    """
    # Handle different naming patterns
    if env_name.startswith("motion2d"):
        # motion2d-p1, motion2d-p2, etc.
        parts = env_name.split("-")
        if len(parts) >= 2 and parts[1].startswith("p"):
            param = int(parts[1][1:])  # Remove 'p' prefix
        else:
            param = 2  # Default
        return "motion2d", param

    if env_name.startswith("stickbutton2d"):
        # stickbutton2d-b1, stickbutton2d-b5, etc.
        parts = env_name.split("-")
        if len(parts) >= 2 and parts[1].startswith("b"):
            param = int(parts[1][1:])  # Remove 'b' prefix
        else:
            param = 1  # Default
        return "stickbutton2d", param

    if env_name.startswith("obstruction2d"):
        # obstruction2d-o0, obstruction2d-o1, etc.
        parts = env_name.split("-")
        if len(parts) >= 2 and parts[1].startswith("o"):
            param = int(parts[1][1:])  # Remove 'o' prefix
        else:
            param = 1  # Default
        return "obstruction2d", param

    if env_name.startswith("clutteredstorage2d"):
        # clutteredstorage2d-b1, clutteredstorage2d-b7, etc.
        parts = env_name.split("-")
        if len(parts) >= 2 and parts[1].startswith("b"):
            param = int(parts[1][1:])  # Remove 'b' prefix
        else:
            param = 1  # Default
        return "clutteredstorage2d", param

    if env_name.startswith("clutteredretrieval2d"):
        # clutteredretrieval2d-o1, clutteredretrieval2d-o3, etc.
        parts = env_name.split("-")
        if len(parts) >= 2 and parts[1].startswith("o"):
            param = int(parts[1][1:])  # Remove 'o' prefix
        else:
            param = 1  # Default
        return "clutteredretrieval2d", param

    # Fallback: assume motion2d
    return "motion2d", 2


def main():
    """Main function to run the complete diffusion policy pipeline."""
    # Get available environments dynamically
    try:
        available_envs = get_available_environments()
        env_choices = list(available_envs.keys())
        default_env = "motion2d-p2" if "motion2d-p2" in env_choices else env_choices[0]
    except Exception as e:
        print(f"Warning: Could not load environments dynamically: {e}")
        # Fallback to a few common ones
        env_choices = ["motion2d-p2", "pushpullhook2d", "stickbutton2d-b2"]
        default_env = "motion2d-p2"

    parser = argparse.ArgumentParser(
        description="Complete diffusion policy pipeline for all PRBench environments"
    )

    # Environment and data options
    parser.add_argument(
        "--env",
        type=str,
        default=default_env,
        choices=env_choices,
        help=(
            f"Environment name. Available: {', '.join(env_choices[:10])}"
            f"{'...' if len(env_choices) > 10 else ''}"
        ),
    )
    parser.add_argument(
        "--data-episodes",
        type=int,
        default=20,
        help="Number of episodes for data collection",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="random",
        choices=["random", "expert", "precomputed"],
        help="Type of data to collect",
    )
    parser.add_argument(
        "--save-demo-videos",
        action="store_true",
        help="Save videos of demonstration trajectories during data generation",
    )

    # Expert demonstration specific options
    parser.add_argument(
        "--num-passages",
        type=int,
        default=2,
        choices=[1, 2, 3],
        help="Number of passages for Motion2D environment (only for expert data)",
    )
    parser.add_argument(
        "--set-random-seed",
        action="store_true",
        help="Use specific random seeds for environment resets (for reproducibility)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed to use when --set-random-seed is enabled",
    )
    parser.add_argument(
        "--env-param",
        type=int,
        help=(
            "Generic environment parameter (passages, buttons, obstructions, "
            "blocks, etc.) - auto-detects based on environment name"
        ),
    )
    parser.add_argument(
        "--max-abstract-plans",
        type=int,
        default=10,
        help="Maximum abstract plans for BilevelPlanningAgent (only for expert data)",
    )
    parser.add_argument(
        "--samples-per-step",
        type=int,
        default=3,
        help="Samples per planning step for BilevelPlanningAgent (only for expert data)",
    )
    parser.add_argument(
        "--planning-timeout",
        type=float,
        default=30.0,
        help=(
            "Planning timeout in seconds for BilevelPlanningAgent "
            "(only for expert data)"
        ),
    )

    # Training options
    parser.add_argument(
        "--train-epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="custom",
        choices=["custom", "lerobot", "behavior_cloning"],
        help=(
            "Type of policy to use (custom diffusion, LeRobot diffusion, "
            "or behavior cloning)"
        ),
    )

    # Evaluation options
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render during evaluation"
    )
    parser.add_argument(
        "--save-videos", action="store_true", help="Save evaluation videos"
    )

    # Pipeline control
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation (use existing dataset)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip evaluation"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to existing dataset (if skipping data generation)",
    )
    parser.add_argument(
        "--precomputed-demos-dir",
        type=str,
        help="Directory containing demonstrations (for data-type=precomputed)",
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to existing model (if skipping training)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./diffusion_pipeline_results",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory for logs"
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List all available environments and exit",
    )

    args = parser.parse_args()

    # Handle list environments command
    if args.list_envs:
        print("Available PRBench Environments:")
        print("=" * 50)
        for short_name, full_id in sorted(available_envs.items()):
            print(f"  {short_name:<25} -> {full_id}")
        print(f"\nTotal: {len(available_envs)} environments")
        print("\nUsage: --env <short_name>")
        return

    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        args.experiment_name = f"{args.env}_{args.data_type}_{timestamp}"

    # Setup directories
    output_dir = Path(args.output_dir) / args.experiment_name
    dataset_dir = output_dir / "datasets"
    model_dir = output_dir / "models"
    eval_dir = output_dir / "evaluation"
    log_dir = Path(args.log_dir)

    for dir_path in [output_dir, dataset_dir, model_dir, eval_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Setup main log file
    main_log_path = log_dir / f"{args.experiment_name}_pipeline.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message("=" * 80)
    log_message("DIFFUSION POLICY PIPELINE FOR GEOM2D ENVIRONMENTS")
    log_message("=" * 80)
    log_message(f"Experiment: {args.experiment_name}")
    log_message(f"Environment: {args.env}")
    log_message(f"Output directory: {output_dir}")
    log_message(f"Log directory: {log_dir}")
    log_message(
        f"Steps: {'Data' if not args.skip_data else 'Skip Data'} -> "
        f"{'Train' if not args.skip_training else 'Skip Train'} -> "
        f"{'Eval' if not args.skip_evaluation else 'Skip Eval'}"
    )
    log_message("=" * 80)

    # Save experiment configuration
    experiment_config = {
        "experiment_name": args.experiment_name,
        "environment": args.env,
        "data_episodes": args.data_episodes,
        "data_type": args.data_type,
        "policy_type": args.policy_type,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_episodes": args.eval_episodes,
        "timestamp": time.time(),
    }

    # Add expert-specific configuration if using expert data
    if args.data_type == "expert":
        experiment_config.update(
            {
                "num_passages": args.num_passages,
                "max_abstract_plans": args.max_abstract_plans,
                "samples_per_step": args.samples_per_step,
                "planning_timeout": args.planning_timeout,
                "set_random_seed": args.set_random_seed,
                "seed": args.seed,
            }
        )

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)
    log_message(f"Experiment config saved to: {config_path}")

    dataset_path = None
    model_path = None

    try:
        # Step 1: Data Generation
        if not args.skip_data:
            log_message(f"\n🔄 STEP 1: Generating {args.data_type} data for {args.env}")
            dataset_name = f"{args.env}_{args.data_type}_{args.data_episodes}ep"

            if args.data_type == "expert":
                # Use expert demonstration collection for geom2d environments

                # Extract environment type and parameter
                env_type, env_param = _parse_environment_name(args.env)

                # Use env_param argument if provided, otherwise use parsed value
                if args.env_param is not None:
                    env_param = args.env_param

                log_message("Using BilevelPlanningAgent for expert demonstrations")
                log_message("Using bilevel planning from third-party submodule")
                log_message(f"Environment type: {env_type}, parameter: {env_param}")

                # Use generic geom2d collector for all environments
                dataset_path = collect_geom2d_demonstrations(
                    env_name=env_type,
                    env_param=env_param,
                    num_episodes=args.data_episodes,
                    output_dir=str(dataset_dir / dataset_name),
                    max_steps_per_episode=1000,  # Reasonable default
                    save_videos=args.save_demo_videos,
                    max_abstract_plans=args.max_abstract_plans,
                    samples_per_step=args.samples_per_step,
                    planning_timeout=args.planning_timeout,
                    seed=args.seed,
                    set_random_seed=args.set_random_seed,
                )
            elif args.data_type == "precomputed":
                # Use precomputed demonstrations
                if not args.precomputed_demos_dir:
                    raise ValueError(
                        "Must provide --precomputed-demos-dir when data-type=precomputed"
                    )

                log_message(
                    f"Loading precomputed demonstrations"
                    f"from: {args.precomputed_demos_dir}"
                )
                dataset_name = f"{args.env}_precomputed"

                dataset_path = load_precomputed_demonstrations(
                    demos_dir=args.precomputed_demos_dir,
                    output_dir=str(dataset_dir),
                    dataset_name=dataset_name,
                )
            else:
                # Use random data generation
                dataset_path = generate_lerobot_dataset(
                    env_name=args.env,
                    dataset_name=dataset_name,
                    num_episodes=args.data_episodes,
                    data_type=args.data_type,
                    output_dir=str(dataset_dir),
                    log_dir=str(log_dir),
                    save_videos=args.save_demo_videos,
                )

            log_message(f"✅ Data generation completed: {dataset_path}")
        else:
            dataset_path = args.dataset_path
            if not dataset_path:
                raise ValueError(
                    "Must provide --dataset-path when skipping data generation"
                )
            log_message(f"⏭️  Skipping data generation, using: {dataset_path}")

        # Step 2: Training
        if not args.skip_training:
            log_message(f"\n🔄 STEP 2: Training {args.policy_type} diffusion policy")

            # Get default config and update with user settings
            train_config = get_default_training_config()
            train_config.update(
                {
                    "batch_size": args.batch_size,
                    "num_epochs": args.train_epochs,
                    "learning_rate": args.learning_rate,
                    "use_wandb": args.use_wandb,
                }
            )

            model_path = str(
                model_dir / f"{args.experiment_name}_{args.policy_type}_model.pth"
            )

            if args.policy_type == "lerobot":
                train_lerobot_diffusion_policy(
                    dataset_path=dataset_path,
                    model_save_path=model_path,
                    config=train_config,
                    log_dir=str(log_dir),
                )
            elif args.policy_type == "behavior_cloning":
                train_behavior_cloning_policy(
                    dataset_path=dataset_path,
                    model_save_path=model_path,
                    config=train_config,
                    log_dir=str(log_dir),
                )
            else:  # custom
                train_diffusion_policy(
                    dataset_path=dataset_path,
                    model_save_path=model_path,
                    config=train_config,
                    log_dir=str(log_dir),
                )
            log_message(f"✅ Training completed: {model_path}")
        else:
            model_path = args.model_path
            if not model_path:
                raise ValueError("Must provide --model-path when skipping training")
            log_message(f"⏭️  Skipping training, using: {model_path}")

        # Step 3: Evaluation
        if not args.skip_evaluation:
            log_message("\n🔄 STEP 3: Evaluating trained policy")

            # Get environment ID
            env_id = available_envs.get(args.env, args.env)

            results = evaluate_policy(
                model_path=model_path,
                env_id=env_id,
                num_episodes=args.eval_episodes,
                output_dir=str(eval_dir),
                render=args.render,
                save_videos=args.save_videos,
                save_plots=True,
                log_dir=str(log_dir),
                max_episode_steps=400,  # Short episodes for testing
                set_random_seed=args.set_random_seed,
                seed=args.seed,
            )

            log_message(f"✅ Evaluation completed: {eval_dir}")
            log_message(
                f"   Mean Return: {results['mean_return']:.2f} ± "
                f"{results['std_return']:.2f}"
            )
            log_message(f"   Success Rate: {results['success_rate']:.2%}")
        else:
            log_message("⏭️  Skipping evaluation")

        # Create summary
        summary = {
            "experiment_name": args.experiment_name,
            "status": "completed",
            "dataset_path": dataset_path,
            "model_path": model_path,
            "evaluation_dir": str(eval_dir) if not args.skip_evaluation else None,
            "completed_at": time.time(),
        }

        summary_path = output_dir / "pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        log_message("\n" + "=" * 80)
        log_message("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        log_message("=" * 80)
        log_message(f"Experiment: {args.experiment_name}")
        log_message(f"Output directory: {output_dir}")
        if dataset_path:
            log_message(f"Dataset: {dataset_path}")
        if model_path:
            log_message(f"Model: {model_path}")
        if not args.skip_evaluation:
            log_message(f"Evaluation: {eval_dir}")
        log_message(f"Summary: {summary_path}")
        log_message(f"Logs: {log_dir}")
        log_message("=" * 80)

    except Exception as e:
        log_message(f"\n❌ PIPELINE FAILED: {str(e)}")
        traceback.print_exc()

        # Log full traceback to file
        with open(main_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} - FULL TRACEBACK:\n")
            traceback.print_exc(file=f)

        # Save error info
        error_info = {
            "experiment_name": args.experiment_name,
            "status": "failed",
            "error": str(e),
            "failed_at": time.time(),
        }

        error_path = output_dir / "pipeline_error.json"
        with open(error_path, "w", encoding="utf-8") as f:
            json.dump(error_info, f, indent=2)

        sys.exit(1)


if __name__ == "__main__":
    main()
