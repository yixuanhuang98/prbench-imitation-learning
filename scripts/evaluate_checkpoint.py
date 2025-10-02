#!/usr/bin/env python3
"""Evaluate a specific checkpoint script.

This script evaluates a single trained checkpoint on a specified environment without
needing to run the full training pipeline.
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

# Add src to path to import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
from prbench_imitation_learning import (
    evaluate_policy,
    get_available_environments,
)

# pylint: enable=wrong-import-position


def evaluate_single_checkpoint(
    checkpoint_path: str,
    env_id: str,
    num_episodes: int = 10,
    output_dir: str = None,
    render: bool = False,
    save_videos: bool = False,
    save_plots: bool = True,
    max_episode_steps: int = 400,
    seed: int = 42,
    set_random_seed: bool = True,
) -> Dict[str, Any]:
    """Evaluate a single checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        env_id: Environment ID for evaluation
        num_episodes: Number of episodes to evaluate
        output_dir: Directory to save results (auto-generated if None)
        render: Whether to render during evaluation
        save_videos: Whether to save videos
        save_plots: Whether to save plots
        max_episode_steps: Maximum steps per episode
        seed: Random seed for evaluation
        set_random_seed: Whether to use fixed seed

    Returns:
        Dictionary containing evaluation results
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    # Generate output directory if not provided
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_name = checkpoint_path.stem
        # Use diffusion_pipeline_results directory
        output_dir = (
            f"diffusion_pipeline_results/"
            f"single_checkpoint_eval_{checkpoint_name}_{timestamp}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 SINGLE CHECKPOINT EVALUATION")
    print("=" * 50)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {num_episodes}")
    print(f"Max steps: {max_episode_steps}")
    print(f"Random seed: {seed}")
    print(f"Output: {output_dir}")
    print("=" * 50)

    # Evaluate the checkpoint
    results = evaluate_policy(
        model_path=str(checkpoint_path),
        env_id=env_id,
        num_episodes=num_episodes,
        output_dir=str(output_dir),
        render=render,
        save_videos=save_videos,
        save_plots=save_plots,
        max_episode_steps=max_episode_steps,
        set_random_seed=set_random_seed,
        seed=seed,
    )

    print("\n📊 EVALUATION RESULTS")
    print("=" * 30)
    print(f"Mean Return: {results['mean_return']:.3f} ± {results['std_return']:.3f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print(f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 30)

    # Save a summary file
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "environment": env_id,
        "num_episodes": num_episodes,
        "max_episode_steps": max_episode_steps,
        "seed": seed,
        "evaluation_results": results,
        "evaluated_at": time.time(),
    }

    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: {summary_path}")

    return results


def main():
    """Main function to evaluate a specific checkpoint."""
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
        description="Evaluate a specific trained checkpoint"
    )

    # Required arguments
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint file (.pth)",
    )

    # Environment options
    parser.add_argument(
        "--env",
        type=str,
        default=default_env,
        choices=env_choices,
        help=(
            f"Environment name. Available: {', '.join(env_choices[:5])}"
            f"{'...' if len(env_choices) > 5 else ''}"
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
        "--max-episode-steps",
        type=int,
        default=400,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible evaluation",
    )
    parser.add_argument(
        "--no-fixed-seed",
        action="store_true",
        help="Don't use fixed seed (allow random evaluation)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        help=(
            "Output directory for results (auto-generated in "
            "diffusion_pipeline_results/ if not provided)"
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render during evaluation",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save videos of evaluation episodes",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't save evaluation plots",
    )

    args = parser.parse_args()

    try:
        # Get environment ID
        try:
            available_envs = get_available_environments()
            env_id = available_envs.get(args.env, args.env)
        except Exception:  # pylint: disable=broad-except
            env_id = args.env

        # Evaluate the checkpoint
        evaluate_single_checkpoint(
            checkpoint_path=args.checkpoint_path,
            env_id=env_id,
            num_episodes=args.eval_episodes,
            output_dir=args.output_dir,
            render=args.render,
            save_videos=args.save_videos,
            save_plots=not args.no_plots,
            max_episode_steps=args.max_episode_steps,
            seed=args.seed,
            set_random_seed=not args.no_fixed_seed,
        )

        print("\n🎉 EVALUATION COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"\n❌ EVALUATION FAILED: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
