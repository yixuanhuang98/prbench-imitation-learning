#!/usr/bin/env python3
"""Example script for training and evaluating policies on the lerobot PushT dataset.

This script demonstrates how to:
1. Load the lerobot PushT dataset
2. Train different policy types on it
3. Evaluate the trained policies in the PushT environment

Policy Types:
    - diffusion: Custom diffusion policy implementation
    - lerobot: LeRobot's diffusion policy implementation
    - behavior_cloning: Behavior cloning baseline
    - all: Train all three policy types

Usage:
    python run_pusht_example.py --policy-type diffusion --epochs 50
    python run_pusht_example.py --policy-type lerobot --epochs 50
    python run_pusht_example.py --policy-type behavior_cloning --epochs 50
    python run_pusht_example.py --policy-type all --epochs 100 --save-videos
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
from prbench_imitation_learning import (
    evaluate_policy,
    get_default_training_config,
    train_behavior_cloning_policy,
    train_diffusion_policy,
    train_lerobot_diffusion_policy,
)

try:
    # Import to verify lerobot is available
    import lerobot

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("ERROR: lerobot is not installed. Install with: pip install lerobot")
    sys.exit(1)


def main():
    """Main function to run PushT training and evaluation."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate policies on lerobot PushT dataset"
    )

    # Training options
    parser.add_argument(
        "--policy-type",
        type=str,
        default="lerobot",
        choices=["diffusion", "lerobot", "behavior_cloning", "all"],
        help="Type of policy to train (diffusion=custom implementation, lerobot=LeRobot implementation)",
    )
    parser.add_argument(
        "--steps", type=int, default=200000, help="Number of training steps (default: 200K for full training)"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test with 5K steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--seed", type=int, default=100000, help="Random seed (default: 100000)"
    )

    # Evaluation options
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--save-videos", action="store_true", help="Save evaluation videos"
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=300,
        help="Maximum steps per evaluation episode",
    )

    # Pipeline control
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training (use existing model)",
    )
    parser.add_argument(
        "--skip-evaluation", action="store_true", help="Skip evaluation"
    )
    parser.add_argument(
        "--model-path", type=str, help="Path to existing model (if skipping training)"
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./pusht_results",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--experiment-name", type=str, help="Experiment name (auto-generated if not provided)"
    )

    args = parser.parse_args()

    # Check lerobot availability
    if not LEROBOT_AVAILABLE:
        print("ERROR: lerobot is required but not installed")
        print("Install with: pip install lerobot")
        sys.exit(1)

    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"pusht_{args.policy_type}_{timestamp}"

    # Setup directories
    output_dir = Path(args.output_dir) / args.experiment_name
    model_dir = output_dir / "models"
    eval_dir = output_dir / "evaluation"
    log_dir = output_dir / "logs"

    for dir_path in [output_dir, model_dir, eval_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PUSHT DATASET TRAINING PIPELINE")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Policy type: {args.policy_type}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Use PushT dataset from HuggingFace
    print("\n📦 Using PushT dataset from HuggingFace")
    print("Note: Dataset will be downloaded on first use (~100MB)")
    dataset_path = "lerobot/pusht"  # Use repo_id directly
    print(f"✅ Dataset: {dataset_path}")

    # Save experiment configuration
    experiment_config = {
        "experiment_name": args.experiment_name,
        "dataset": "lerobot/pusht",
        "policy_type": args.policy_type,
        "training_steps": 5000 if args.quick_test else args.steps,
        "quick_test": args.quick_test,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "eval_episodes": args.eval_episodes,
        "timestamp": time.time(),
    }

    config_path = output_dir / "experiment_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Experiment config saved to: {config_path}")

    # Training configuration - use optimized LeRobot configuration
    train_config = get_default_training_config()
    
    # Override with command-line arguments
    training_steps = 5000 if args.quick_test else args.steps
    train_config.update({
        "batch_size": args.batch_size,
        "training_steps": training_steps,
        "learning_rate": args.learning_rate,
        "use_wandb": args.use_wandb,
        "seed": args.seed,
        "num_workers": 4,
        # Logging and checkpointing
        "log_interval": 200,
        "save_freq": 25000 if not args.quick_test else 1000,
        "eval_freq": 25000 if not args.quick_test else 1000,
    })
    
    if args.quick_test:
        print("🚀 Quick test mode: Training for 5K steps")
    else:
        print(f"📊 Full training mode: {training_steps} steps")

    results = {}

    # Train policies
    if not args.skip_training:
        policies_to_train = []
        if args.policy_type == "all":
            policies_to_train = ["diffusion", "lerobot", "behavior_cloning"]
        else:
            policies_to_train = [args.policy_type]

        for policy_type in policies_to_train:
            print(f"\n🔄 Training {policy_type} policy...")
            model_path = model_dir / f"pusht_{policy_type}_model.pth"

            try:
                if policy_type == "diffusion":
                    print("Using custom diffusion policy implementation")
                    train_diffusion_policy(
                        dataset_path=str(dataset_path),
                        model_save_path=str(model_path),
                        config=train_config,
                        log_dir=str(log_dir / policy_type),
                    )
                elif policy_type == "lerobot":
                    print("Using LeRobot diffusion policy implementation")
                    train_lerobot_diffusion_policy(
                        dataset_path=str(dataset_path),
                        model_save_path=str(model_path),
                        config=train_config,
                        log_dir=str(log_dir / policy_type),
                        use_original_lerobot_params=True,  # Use original LeRobot parameters for better performance
                    )
                else:  # behavior_cloning
                    train_behavior_cloning_policy(
                        dataset_path=str(dataset_path),
                        model_save_path=str(model_path),
                        config=train_config,
                        log_dir=str(log_dir / policy_type),
                    )

                print(f"✅ Training completed: {model_path}")
                results[f"{policy_type}_model_path"] = str(model_path)

            except Exception as e:
                print(f"❌ Training failed for {policy_type}: {e}")
                import traceback

                traceback.print_exc()
                continue

    else:
        if not args.model_path:
            print("ERROR: --model-path required when skipping training")
            sys.exit(1)
        print(f"⏭️  Skipping training, using model: {args.model_path}")
        results["model_path"] = args.model_path

    # Evaluate policies
    if not args.skip_evaluation:
        models_to_eval = []

        if args.skip_training:
            # Use provided model
            policy_name = "loaded_model"
            models_to_eval = [(policy_name, args.model_path)]
        else:
            # Evaluate trained models
            if args.policy_type == "all":
                models_to_eval = [
                    ("diffusion", results.get("diffusion_model_path")),
                    ("lerobot", results.get("lerobot_model_path")),
                    ("behavior_cloning", results.get("behavior_cloning_model_path")),
                ]
            else:
                models_to_eval = [
                    (args.policy_type, results.get(f"{args.policy_type}_model_path"))
                ]

        # Filter out None paths (failed training)
        models_to_eval = [(name, path) for name, path in models_to_eval if path]

        eval_results = {}
        
        # Import gym_pusht to register the environment
        import gym_pusht  # pylint: disable=import-outside-toplevel,unused-import
        pusht_env_id = "gym_pusht/PushT-v0"
        
        for policy_name, model_path in models_to_eval:
            print(f"\n🔄 Evaluating {policy_name} policy...")

            try:
                eval_result = evaluate_policy(
                    model_path=str(model_path),
                    env_id=pusht_env_id,
                    num_episodes=args.eval_episodes,
                    output_dir=str(eval_dir / policy_name),
                    render=False,
                    save_videos=args.save_videos,
                    save_plots=True,
                    log_dir=str(log_dir / f"eval_{policy_name}"),
                    max_episode_steps=args.max_episode_steps,
                )

                eval_results[policy_name] = eval_result

                print(f"✅ Evaluation completed for {policy_name}")
                print(
                    f"   Mean return: {eval_result['mean_return']:.2f} "
                    f"± {eval_result['std_return']:.2f}"
                )
                print(f"   Success rate: {eval_result['success_rate']:.2%}")

            except Exception as e:
                print(f"❌ Evaluation failed for {policy_name}: {e}")
                import traceback

                traceback.print_exc()
                continue

        # Save evaluation results
        if eval_results:
            eval_results_path = output_dir / "evaluation_results.json"
            with open(eval_results_path, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2)
            print(f"\nEvaluation results saved to: {eval_results_path}")

        # Print comparison if multiple policies
        if len(eval_results) > 1:
            print("\n" + "=" * 80)
            print("POLICY COMPARISON")
            print("=" * 80)
            for policy_name, eval_result in eval_results.items():
                print(f"{policy_name.upper()}:")
                print(
                    f"  Mean return: {eval_result['mean_return']:.2f} "
                    f"± {eval_result['std_return']:.2f}"
                )
                print(f"  Success rate: {eval_result['success_rate']:.2%}")
                print()
            print("=" * 80)

    else:
        print("⏭️  Skipping evaluation")

    # Create summary
    summary = {
        "experiment_name": args.experiment_name,
        "status": "completed",
        "results": results,
        "completed_at": time.time(),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Output directory: {output_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
