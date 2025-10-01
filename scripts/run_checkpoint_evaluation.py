#!/usr/bin/env python3
"""Multi-checkpoint evaluation script.

This script trains a policy while saving multiple checkpoints during training, then
evaluates all checkpoints on a fixed set of test episodes to analyze performance
evolution during training.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prbench_imitation_learning import (
    evaluate_policy,
    generate_lerobot_dataset,
    get_available_environments,
    get_default_training_config,
)

# Import functions for different data types
sys.path.insert(0, str(Path(__file__).parent))
from run_diffusion_pipeline import (
    _parse_environment_name,
    collect_geom2d_demonstrations,
    load_precomputed_demonstrations,
)


def train_policy_with_checkpoints(
    dataset_path: str,
    checkpoint_dir: str,
    config: Dict[str, Any],
    policy_type: str = "behavior_cloning",
    log_dir: str = "./logs",
    checkpoint_interval: int = 5,
) -> List[str]:
    """Train policy with multiple checkpoints.

    Args:
        dataset_path: Path to the dataset
        checkpoint_dir: Directory to save checkpoints
        config: Training configuration
        policy_type: Type of policy to train (behavior_cloning, custom, lerobot)
        log_dir: Directory to save logs
        checkpoint_interval: Save checkpoint every N epochs

    Returns:
        List of checkpoint paths
    """
    import time

    import torch
    import torch.nn.functional as F
    from torch import optim
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from prbench_imitation_learning.policy import (
        BehaviorCloningPolicy,
        DiffusionPolicy,
        DiffusionPolicyDataset,
    )

    # LeRobot imports (optional)
    try:
        from lerobot.configs.policies import FeatureType, PolicyFeature
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.policies.diffusion.modeling_diffusion import (
            DiffusionPolicy as LeRobotDiffusionPolicy,)
        from torch.amp import GradScaler

        LEROBOT_AVAILABLE = True
    except ImportError:
        LEROBOT_AVAILABLE = False

    # Setup logging
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"{policy_type}_checkpoint_training.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message(f"Starting {policy_type} training with checkpoints...")
    log_message(f"Dataset: {dataset_path}")
    log_message(f"Checkpoint directory: {checkpoint_dir}")
    log_message(f"Checkpoint interval: {checkpoint_interval} epochs")
    log_message(f"Policy type: {policy_type}")
    log_message(f"Config: {config}")

    # Create checkpoint directory
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    # Set device
    if config.get("force_cpu", False):
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    # Create dataset
    dataset = DiffusionPolicyDataset(
        dataset_path=dataset_path,
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        pred_horizon=config["pred_horizon"],
    )

    # Create data loader
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )

    log_message(f"Dataset size: {len(dataset)}")
    log_message(f"Batch size: {config['batch_size']}")

    # Get dimensions from dataset
    obs_dim = dataset.obs_dim
    action_dim = dataset.action_dim

    log_message(f"Observation dimension: {obs_dim}")
    log_message(f"Action dimension: {action_dim}")

    # Create model based on policy type
    if policy_type == "behavior_cloning":
        model = BehaviorCloningPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=config["obs_horizon"],
            action_horizon=config["action_horizon"],
            hidden_dim=config.get("hidden_dim", 512),
            num_layers=config.get("num_layers", 3),
        ).to(device)
    elif policy_type == "lerobot":
        if not LEROBOT_AVAILABLE:
            raise ImportError(
                "LeRobot is not available. Please install it with: pip install lerobot"
            )

        # Create LeRobot diffusion config
        obs_feature = PolicyFeature(
            name="observation.state",
            shape=[obs_dim],
            names=None,
        )

        diffusion_config = DiffusionConfig(
            n_obs_steps=config["obs_horizon"],
            n_action_steps=config["action_horizon"],
            input_shapes={"observation.state": [obs_dim]},
            output_shapes={"action": [action_dim]},
            input_normalization_modes={"observation.state": "mean_std"},
            output_normalization_modes={"action": "mean_std"},
        )

        model = LeRobotDiffusionPolicy(diffusion_config).to(device)
    else:  # custom diffusion policy
        model = DiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=config["obs_horizon"],
            action_horizon=config["action_horizon"],
            num_diffusion_iters=config["num_diffusion_iters"],
        ).to(device)

    log_message(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-6),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["num_epochs"])

    # Setup gradient scaler for LeRobot (if using mixed precision)
    scaler = None
    if policy_type == "lerobot" and config.get("use_mixed_precision", False):
        scaler = GradScaler()

    # Training loop
    model.train()
    checkpoint_paths = []

    for epoch in range(config["num_epochs"]):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            obs_seq = batch["obs_states"].to(device)  # [B, obs_horizon, obs_dim]
            action_seq = batch["actions"].to(device)  # [B, action_horizon, action_dim]

            # Forward pass and loss computation based on policy type
            if policy_type == "behavior_cloning":
                pred_actions = model(obs_seq)
                loss = F.mse_loss(pred_actions, action_seq)
            elif policy_type == "lerobot":
                # LeRobot expects different input format
                batch_lerobot = {
                    "observation.state": obs_seq,
                    "action": action_seq,
                }
                if scaler:
                    with torch.amp.autocast(device_type=device.type):
                        output = model(batch_lerobot)
                        loss = output["loss"]
                else:
                    output = model(batch_lerobot)
                    loss = output["loss"]
            else:  # custom diffusion policy
                noise_pred, noise = model(obs_seq, action_seq)
                loss = F.mse_loss(noise_pred, noise)

            # Backward pass
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("grad_clip_norm", 1.0)
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.get("grad_clip_norm", 1.0)
                )
                optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Log progress
            if batch_idx % config.get("log_interval", 10) == 0:
                msg = (
                    f"Epoch {epoch+1}/{config['num_epochs']}, "
                    f"Batch {batch_idx+1}/{len(dataloader)}, "
                    f"Loss: {loss.item():.6f}"
                )
                log_message(msg)

        # End of epoch
        avg_loss = epoch_loss / num_batches
        scheduler.step()

        msg = f"Epoch {epoch+1} completed. Average loss: {avg_loss:.6f}"
        log_message(msg)

        # Save checkpoint at regular intervals
        if (epoch + 1) % checkpoint_interval == 0 or epoch == config["num_epochs"] - 1:
            checkpoint_path = checkpoint_dir_path / f"checkpoint_epoch_{epoch+1}.pth"

            save_config = config.copy()
            save_config.update(
                {
                    "obs_dim": obs_dim,
                    "action_dim": action_dim,
                    "image_shape": list(dataset.image_shape),
                    "policy_type": policy_type,  # Use actual policy type
                }
            )

            # Prepare checkpoint data
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "config": save_config,
            }

            # Add LeRobot-specific config if needed
            if policy_type == "lerobot":
                checkpoint_data["diffusion_config"] = model.config

            torch.save(checkpoint_data, checkpoint_path)

            checkpoint_paths.append(str(checkpoint_path))
            log_message(f"Checkpoint saved: {checkpoint_path}")

    log_message("Training completed!")
    log_message(f"Total checkpoints saved: {len(checkpoint_paths)}")

    return checkpoint_paths


def evaluate_all_checkpoints(
    checkpoint_paths: List[str],
    env_id: str,
    num_episodes: int = 10,
    output_dir: str = "./checkpoint_evaluation",
    log_dir: str = "./logs",
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate all checkpoints on the same set of test episodes.

    Args:
        checkpoint_paths: List of checkpoint file paths
        env_id: Environment ID for evaluation
        num_episodes: Number of episodes to evaluate on
        output_dir: Directory to save evaluation results
        log_dir: Directory for logs
        seed: Random seed for reproducible evaluation

    Returns:
        Dictionary containing evaluation results for all checkpoints
    """
    # Setup logging
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / "checkpoint_evaluation.log"

    def log_message(message: str):
        """Log message to both console and file."""
        print(message)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

    log_message("Starting checkpoint evaluation...")
    log_message(f"Environment: {env_id}")
    log_message(f"Number of episodes: {num_episodes}")
    log_message(f"Number of checkpoints: {len(checkpoint_paths)}")
    log_message(f"Random seed: {seed}")

    # Create output directory
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for i, checkpoint_path in enumerate(checkpoint_paths):
        checkpoint_name = Path(checkpoint_path).stem
        log_message(
            f"\nEvaluating checkpoint {i+1}/{len(checkpoint_paths)}: {checkpoint_name}"
        )

        try:
            # Create checkpoint-specific output directory
            checkpoint_output_dir = output_dir_path / checkpoint_name
            checkpoint_output_dir.mkdir(exist_ok=True)

            # Evaluate this checkpoint
            results = evaluate_policy(
                model_path=checkpoint_path,
                env_id=env_id,
                num_episodes=num_episodes,
                output_dir=str(checkpoint_output_dir),
                render=False,
                save_videos=False,
                save_plots=True,
                log_dir=str(log_dir),
                max_episode_steps=400,
                set_random_seed=False,
                seed=seed,  # Use same seed for all evaluations
            )

            # Extract epoch number from checkpoint name
            epoch_num = int(checkpoint_name.split("_")[-1])

            # Store results
            all_results[checkpoint_name] = {
                "epoch": epoch_num,
                "checkpoint_path": checkpoint_path,
                "mean_return": results["mean_return"],
                "std_return": results["std_return"],
                "success_rate": results["success_rate"],
                "mean_length": results["mean_length"],
                "std_length": results["std_length"],
                "individual_returns": results["episode_returns"],
                "individual_lengths": results["episode_lengths"],
                "individual_successes": results["success_rates"],
            }

            log_message(
                f"  Mean return: {results['mean_return']:.3f} ± {results['std_return']:.3f}"
            )
            log_message(f"  Success rate: {results['success_rate']:.1%}")
            log_message(f"  Mean length: {results['mean_length']:.1f}")

        except Exception as e:
            log_message(f"  Error evaluating {checkpoint_name}: {e}")
            continue

    log_message(f"\nCompleted evaluation of {len(all_results)} checkpoints")

    # Save consolidated results
    results_file = output_dir_path / "all_checkpoint_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    log_message(f"Consolidated results saved to: {results_file}")

    return all_results


def create_checkpoint_analysis_plots(
    results: Dict[str, Any],
    save_dir: str,
    show_plots: bool = False,
) -> None:
    """Create plots showing performance evolution across checkpoints.

    Args:
        results: Dictionary containing evaluation results for all checkpoints
        save_dir: Directory to save plots
        show_plots: Whether to display plots
    """
    if not results:
        print("No results to plot!")
        return

    # Convert results to DataFrame for easier plotting
    plot_data = []
    for checkpoint_name, checkpoint_results in results.items():
        plot_data.append(
            {
                "checkpoint": checkpoint_name,
                "epoch": checkpoint_results["epoch"],
                "mean_return": checkpoint_results["mean_return"],
                "std_return": checkpoint_results["std_return"],
                "success_rate": checkpoint_results["success_rate"],
                "mean_length": checkpoint_results["mean_length"],
            }
        )

    df = pd.DataFrame(plot_data)
    df = df.sort_values("epoch")

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Policy Performance Evolution During Training", fontsize=16, fontweight="bold"
    )

    # Plot 1: Mean Return vs Epoch
    ax1.errorbar(
        df["epoch"],
        df["mean_return"],
        yerr=df["std_return"],
        fmt="bo-",
        linewidth=2,
        markersize=8,
        capsize=5,
    )
    ax1.set_xlabel("Training Epoch")
    ax1.set_ylabel("Mean Return")
    ax1.set_title("Mean Return vs Training Progress")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Success Rate vs Epoch
    ax2.plot(df["epoch"], df["success_rate"] * 100, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Training Epoch")
    ax2.set_ylabel("Success Rate (%)")
    ax2.set_title("Success Rate vs Training Progress")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    # Add value labels on success rate points
    for _, row in df.iterrows():
        ax2.annotate(
            f'{row["success_rate"]*100:.1f}%',
            (row["epoch"], row["success_rate"] * 100),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # Plot 3: Mean Episode Length vs Epoch
    ax3.plot(df["epoch"], df["mean_length"], "go-", linewidth=2, markersize=8)
    ax3.set_xlabel("Training Epoch")
    ax3.set_ylabel("Mean Episode Length")
    ax3.set_title("Episode Length vs Training Progress")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Performance Summary Table
    ax4.axis("tight")
    ax4.axis("off")

    # Create table data
    table_data = []
    for _, row in df.iterrows():
        table_data.append(
            [
                f"Epoch {row['epoch']}",
                f"{row['mean_return']:.2f}",
                f"{row['success_rate']*100:.1f}%",
                f"{row['mean_length']:.1f}",
            ]
        )

    table = ax4.table(
        cellText=table_data,
        colLabels=["Checkpoint", "Mean Return", "Success Rate", "Mean Length"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax4.set_title("Performance Summary", fontweight="bold")

    plt.tight_layout()

    # Save plot
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)
    plot_path = save_dir_path / "checkpoint_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Analysis plot saved to: {plot_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()

    # Print summary statistics
    print("\n📊 CHECKPOINT ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"{'Epoch':<8} {'Return':<12} {'Success Rate':<15} {'Episode Length':<15}")
    print("=" * 60)

    for _, row in df.iterrows():
        print(
            f"{row['epoch']:<8} {row['mean_return']:<12.2f} "
            f"{row['success_rate']*100:<15.1f} {row['mean_length']:<15.1f}"
        )

    # Find best performing checkpoint
    best_return_idx = df["mean_return"].idxmax()
    best_success_idx = df["success_rate"].idxmax()

    # Extract values explicitly for type safety
    best_return_epoch = int(df.loc[best_return_idx, "epoch"])
    best_return_value = float(df.loc[best_return_idx, "mean_return"])
    best_success_epoch = int(df.loc[best_success_idx, "epoch"])
    best_success_value = float(df.loc[best_success_idx, "success_rate"])

    print(f"\n🏆 BEST PERFORMANCE:")
    print(f"Best Return: Epoch {best_return_epoch} ({best_return_value:.3f})")
    print(f"Best Success Rate: Epoch {best_success_epoch} ({best_success_value*100:.1f}%)")


def main():
    """Main function to run checkpoint training and evaluation."""
    # Get available environments
    try:
        available_envs = get_available_environments()
        env_choices = list(available_envs.keys())
        default_env = "motion2d-p1" if "motion2d-p1" in env_choices else env_choices[0]
    except Exception as e:
        print(f"Warning: Could not load environments dynamically: {e}")
        env_choices = ["motion2d-p1", "motion2d-p2", "stickbutton2d-b2"]
        default_env = "motion2d-p1"

    parser = argparse.ArgumentParser(
        description="Train policy with checkpoints and evaluate performance evolution"
    )

    # Environment and data options
    parser.add_argument(
        "--env",
        type=str,
        default=default_env,
        choices=env_choices,
        help=f"Environment name. Available: {', '.join(env_choices[:5])}{'...' if len(env_choices) > 5 else ''}",
    )
    parser.add_argument(
        "--data-episodes",
        type=int,
        default=10,
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
        "--precomputed-demos-dir",
        type=str,
        help="Directory containing precomputed demonstrations (for data-type=precomputed)",
    )

    # Expert demonstration specific options
    parser.add_argument(
        "--env-param",
        type=int,
        help="Generic environment parameter (passages, buttons, obstructions, etc.)",
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
        help="Planning timeout in seconds for BilevelPlanningAgent (only for expert data)",
    )
    parser.add_argument(
        "--set-random-seed",
        action="store_true",
        help="Use specific random seeds for environment resets (for reproducibility)",
    )

    # Training options
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="behavior_cloning",
        choices=["behavior_cloning", "custom", "lerobot"],
        help="Type of policy to train",
    )

    # Evaluation options
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation of each checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-data",
        action="store_true",
        help="Skip data generation (use existing dataset)",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to existing dataset (if skipping data generation)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoint_experiment_results",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots instead of just saving them",
    )

    args = parser.parse_args()

    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = time.strftime("%Y-%m-%d-%H%M%S")
        args.experiment_name = (
            f"checkpoint_exp_{args.env}_{args.policy_type}_{timestamp}"
        )

    # Setup directories
    output_dir = Path(args.output_dir) / args.experiment_name
    dataset_dir = output_dir / "datasets"
    checkpoint_dir = output_dir / "checkpoints"
    eval_dir = output_dir / "evaluation"
    log_dir = output_dir / "logs"

    for dir_path in [output_dir, dataset_dir, checkpoint_dir, eval_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("🧪 CHECKPOINT EVALUATION EXPERIMENT")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Environment: {args.env}")
    print(f"Data type: {args.data_type}")
    print(f"Policy type: {args.policy_type}")
    print(f"Training epochs: {args.train_epochs}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Evaluation episodes: {args.eval_episodes}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    try:
        # Step 1: Data Generation (if not skipped)
        if not args.skip_data:
            print(f"\n🔄 STEP 1: Generating {args.data_type} data for {args.env}")
            dataset_name = f"{args.env}_{args.data_type}_{args.data_episodes}ep"

            if args.data_type == "expert":
                # Use expert demonstration collection
                env_type, env_param = _parse_environment_name(args.env)

                # Use env_param argument if provided, otherwise use parsed value
                if args.env_param is not None:
                    env_param = args.env_param

                print("Using BilevelPlanningAgent for expert demonstrations")
                dataset_path = collect_geom2d_demonstrations(
                    env_name=env_type,
                    env_param=env_param,
                    num_episodes=args.data_episodes,
                    output_dir=str(dataset_dir / dataset_name),
                    max_steps_per_episode=1000,
                    save_videos=True,
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

                print(
                    f"Loading precomputed demonstrations from: {args.precomputed_demos_dir}"
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
                    save_videos=False,
                )

            print(f"✅ Data generation completed: {dataset_path}")
        else:
            dataset_path = args.dataset_path
            if not dataset_path:
                raise ValueError(
                    "Must provide --dataset-path when skipping data generation"
                )
            print(f"⏭️  Skipping data generation, using: {dataset_path}")

        # Step 2: Training with Checkpoints
        print(f"\n🔄 STEP 2: Training with checkpoints")

        # Get default config and update with user settings
        train_config = get_default_training_config()
        train_config.update(
            {
                "batch_size": args.batch_size,
                "num_epochs": args.train_epochs,
                "learning_rate": args.learning_rate,
            }
        )

        checkpoint_paths = train_policy_with_checkpoints(
            dataset_path=dataset_path,
            checkpoint_dir=str(checkpoint_dir),
            config=train_config,
            policy_type=args.policy_type,
            log_dir=str(log_dir),
            checkpoint_interval=args.checkpoint_interval,
        )

        print(f"✅ Training completed with {len(checkpoint_paths)} checkpoints")

        # Step 3: Evaluate All Checkpoints
        print(f"\n🔄 STEP 3: Evaluating all checkpoints")

        # Get environment ID
        try:
            available_envs = get_available_environments()
            env_id = available_envs.get(args.env, args.env)
        except:
            env_id = args.env

        all_results = evaluate_all_checkpoints(
            checkpoint_paths=checkpoint_paths,
            env_id=env_id,
            num_episodes=args.eval_episodes,
            output_dir=str(eval_dir),
            log_dir=str(log_dir),
            seed=args.seed,
        )

        print(f"✅ Evaluation completed for {len(all_results)} checkpoints")

        # Step 4: Create Analysis Plots
        print(f"\n🔄 STEP 4: Creating analysis plots")

        create_checkpoint_analysis_plots(
            results=all_results,
            save_dir=str(output_dir / "analysis"),
            show_plots=args.show_plots,
        )

        print(f"✅ Analysis plots created")

        # Save experiment summary
        summary = {
            "experiment_name": args.experiment_name,
            "environment": args.env,
            "data_type": args.data_type,
            "policy_type": args.policy_type,
            "data_episodes": args.data_episodes,
            "train_epochs": args.train_epochs,
            "checkpoint_interval": args.checkpoint_interval,
            "eval_episodes": args.eval_episodes,
            "num_checkpoints": len(checkpoint_paths),
            "dataset_path": dataset_path,
            "checkpoint_paths": checkpoint_paths,
            "completed_at": time.time(),
        }

        summary_path = output_dir / "experiment_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("🎉 CHECKPOINT EXPERIMENT COMPLETED!")
        print("=" * 60)
        print(f"Experiment: {args.experiment_name}")
        print(f"Output directory: {output_dir}")
        print(f"Checkpoints: {len(checkpoint_paths)}")
        print(f"Analysis plots: {output_dir / 'analysis'}")
        print(f"Summary: {summary_path}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ EXPERIMENT FAILED: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
