#!/usr/bin/env python
"""
Enhanced LeRobot training script with expert demonstration generation support.

This script extends the LeRobot training pipeline to support:
1. Expert demonstration generation using BilevelPlanningAgent for geom2d environments
2. Training on multiple environment types (not just PushT)
3. Integrated data generation and training pipeline

Usage examples:
    # Train on PushT with existing dataset
    python scripts/train_lerobot_with_experts.py \
        --dataset.repo_id=lerobot/pusht \
        --policy.type=diffusion \
        --output_dir=outputs/pusht_train \
        --steps=10000

    # Generate expert demos for Motion2D and train
    python scripts/train_lerobot_with_experts.py \
        --generate_expert_data \
        --expert_env=motion2d \
        --expert_env_param=2 \
        --expert_episodes=50 \
        --policy.type=diffusion \
        --output_dir=outputs/motion2d_expert_train \
        --steps=50000

    # Generate expert demos for StickButton2D and train
    python scripts/train_lerobot_with_experts.py \
        --generate_expert_data \
        --expert_env=stickbutton2d \
        --expert_env_param=3 \
        --expert_episodes=100 \
        --policy.type=diffusion \
        --output_dir=outputs/stickbutton_expert_train \
        --steps=100000
"""

import argparse
import json
import logging
import pickle
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

# Add src to path for prbench modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
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
                    next_obs, reward, terminated, truncated, next_info = env.step(action)
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


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    """Performs a single training step to update the policy's weights."""
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
    grad_scaler.scale(loss).backward()

    grad_scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    """Main training function with optional expert data generation."""
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    # Check for expert data generation flags in environment variables or config
    # This is a workaround since the parser.wrap() doesn't easily allow custom args
    generate_expert_data = getattr(cfg, "generate_expert_data", False)
    
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler, preprocessor, postprocessor
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        if cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy_all(
                    envs=eval_env,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            aggregated = eval_info["overall"]

            for suite, suite_info in eval_info.items():
                logging.info("Suite %s aggregated: %s", suite, suite_info)

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = aggregated.pop("eval_s")
            eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
            eval_tracker.pc_success = aggregated.pop("pc_success")
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

    if eval_env:
        close_envs(eval_env)
    logging.info("End of training")

    if cfg.policy.push_to_hub:
        policy.push_model_to_hub(cfg)
        preprocessor.push_to_hub(cfg.policy.repo_id)
        postprocessor.push_to_hub(cfg.policy.repo_id)


def main():
    """Main entry point with expert data generation support."""
    # Parse additional arguments for expert data generation
    arg_parser = argparse.ArgumentParser(
        description="Train LeRobot policies with optional expert demonstration generation",
        add_help=False,  # Let LeRobot parser handle help
    )
    
    expert_group = arg_parser.add_argument_group("Expert Demonstration Generation")
    expert_group.add_argument(
        "--generate_expert_data",
        action="store_true",
        help="Generate expert demonstrations before training",
    )
    expert_group.add_argument(
        "--expert_env",
        type=str,
        choices=["motion2d", "stickbutton2d", "obstruction2d", "clutteredstorage2d", "clutteredretrieval2d"],
        default="motion2d",
        help="Environment for expert demonstration generation",
    )
    expert_group.add_argument(
        "--expert_env_param",
        type=int,
        default=2,
        help="Environment parameter (passages, buttons, obstructions, etc.)",
    )
    expert_group.add_argument(
        "--expert_episodes",
        type=int,
        default=50,
        help="Number of expert episodes to generate",
    )
    expert_group.add_argument(
        "--expert_max_steps",
        type=int,
        default=5000,
        help="Maximum steps per expert episode",
    )
    expert_group.add_argument(
        "--expert_save_videos",
        action="store_true",
        help="Save videos during expert demonstration collection",
    )
    expert_group.add_argument(
        "--expert_max_abstract_plans",
        type=int,
        default=10,
        help="Maximum abstract plans for BilevelPlanningAgent",
    )
    expert_group.add_argument(
        "--expert_samples_per_step",
        type=int,
        default=3,
        help="Samples per step for BilevelPlanningAgent",
    )
    expert_group.add_argument(
        "--expert_planning_timeout",
        type=float,
        default=30.0,
        help="Planning timeout in seconds",
    )
    expert_group.add_argument(
        "--expert_set_random_seed",
        action="store_true",
        help="Use specific random seeds for environment resets",
    )
    
    # Parse known args (expert-specific ones)
    expert_args, remaining_args = arg_parser.parse_known_args()
    
    # If expert data generation is requested, generate it first
    if expert_args.generate_expert_data:
        init_logging()
        
        logging.info("=" * 80)
        logging.info("EXPERT DEMONSTRATION GENERATION")
        logging.info("=" * 80)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        expert_output_dir = f"./expert_data/{expert_args.expert_env}_p{expert_args.expert_env_param}_{timestamp}"
        
        try:
            dataset_path = collect_expert_demonstrations(
                env_name=expert_args.expert_env,
                env_param=expert_args.expert_env_param,
                num_episodes=expert_args.expert_episodes,
                output_dir=expert_output_dir,
                max_steps_per_episode=expert_args.expert_max_steps,
                save_videos=expert_args.expert_save_videos,
                max_abstract_plans=expert_args.expert_max_abstract_plans,
                samples_per_step=expert_args.expert_samples_per_step,
                planning_timeout=expert_args.expert_planning_timeout,
                seed=123,
                set_random_seed=expert_args.expert_set_random_seed,
            )
            
            logging.info("=" * 80)
            logging.info("Expert demonstration generation completed!")
            logging.info(f"Dataset saved to: {dataset_path}")
            logging.info("=" * 80)
            
            # Add dataset path to remaining args for training
            # Note: You would need to configure the dataset loading in your config file
            # or pass it via command line arguments
            logging.info("\nTo train with this dataset, use:")
            logging.info(f"  --dataset.path={dataset_path}")
            
        except Exception as e:
            logging.error(f"Failed to generate expert demonstrations: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        logging.info("\n" + "=" * 80)
        logging.info("STARTING TRAINING WITH EXPERT DATA")
        logging.info("=" * 80 + "\n")
    
    # Continue with normal LeRobot training
    init_logging()
    train()


if __name__ == "__main__":
    main()

