#!/usr/bin/env python3
"""Complete pipeline script for generating data, training, and evaluating diffusion
policies on geom2d environments."""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
from prbench_imitation_learning import (
    evaluate_policy,
    generate_lerobot_dataset,
    get_available_environments,
    get_default_training_config,
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

# Import expert demonstration collection
try:
    from collect_motion2d_demonstrations import collect_motion2d_demonstrations

    EXPERT_COLLECTION_AVAILABLE = True
except ImportError:
    EXPERT_COLLECTION_AVAILABLE = False


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
        choices=["random", "expert"],
        help="Type of data to collect (random or expert demonstrations)",
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
        choices=["custom", "lerobot"],
        help="Type of diffusion policy to use (custom implementation or LeRobot)",
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
        timestamp = int(time.time())
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
                # Use expert demonstration collection for Motion2D
                if not EXPERT_COLLECTION_AVAILABLE:
                    raise ImportError(
                        "Expert demonstration collection not available. "
                        "Make sure collect_motion2d_demonstrations.py is in the same "
                        "directory and bilevel planning third-party module is available."
                    )

                if not args.env.startswith("motion2d"):
                    raise ValueError(
                        "Expert demonstrations are currently only supported for "
                        f"Motion2D environments. Got environment: {args.env}"
                    )

                log_message("Using BilevelPlanningAgent for expert demonstrations")
                log_message("Using bilevel planning from third-party submodule")

                dataset_path = collect_motion2d_demonstrations(
                    num_passages=args.num_passages,
                    num_episodes=args.data_episodes,
                    output_dir=str(dataset_dir / dataset_name),
                    max_steps_per_episode=1000,  # Reasonable default
                    save_videos=args.save_demo_videos,
                    max_abstract_plans=args.max_abstract_plans,
                    samples_per_step=args.samples_per_step,
                    planning_timeout=args.planning_timeout,
                    seed=123,  # Fixed seed for reproducibility
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
