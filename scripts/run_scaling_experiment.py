#!/usr/bin/env python3
"""
Scaling experiment script: Evaluate behavior cloning performance vs number of demonstrations.

This script runs experiments with different numbers of demonstrations and generates
a figure showing the relationship between success rate and number of demonstrations.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def run_experiment(
    env: str,
    policy_type: str,
    data_episodes: int,
    train_epochs: int = 1,
    eval_episodes: int = 1,
    experiment_prefix: str = "scaling_exp",
) -> Dict[str, Any]:
    """Run a single experiment with specified parameters.

    Args:
        env: Environment name (e.g., 'motion2d-p1')
        policy_type: Policy type (e.g., 'behavior_cloning')
        data_episodes: Number of demonstration episodes
        train_epochs: Number of training epochs
        eval_episodes: Number of evaluation episodes
        experiment_prefix: Prefix for experiment name

    Returns:
        Dictionary containing experiment results
    """
    experiment_name = f"{experiment_prefix}_{env}_{policy_type}_{data_episodes}demo"

    print(f"\n{'='*80}")
    print(f"🧪 RUNNING EXPERIMENT: {experiment_name}")
    print(f"{'='*80}")
    print(f"Environment: {env}")
    print(f"Policy: {policy_type}")
    print(f"Demonstrations: {data_episodes}")
    print(f"Training epochs: {train_epochs}")
    print(f"Evaluation episodes: {eval_episodes}")
    print(f"{'='*80}")

    # Build command
    cmd = [
        "python",
        "scripts/run_diffusion_pipeline.py",
        "--env",
        env,
        "--policy-type",
        policy_type,
        "--data-type",
        "expert",
        "--data-episodes",
        str(data_episodes),
        "--train-epochs",
        str(train_epochs),
        "--eval-episodes",
        str(eval_episodes),
        "--experiment-name",
        experiment_name,
    ]

    try:
        # Run the experiment
        start_time = time.time()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600  # 1 hour timeout
        )
        end_time = time.time()

        if result.returncode != 0:
            print(f"❌ Experiment failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr}")
            return {
                "experiment_name": experiment_name,
                "env": env,
                "policy_type": policy_type,
                "data_episodes": data_episodes,
                "train_epochs": train_epochs,
                "eval_episodes": eval_episodes,
                "success": False,
                "error": result.stderr,
                "runtime": end_time - start_time,
            }

        # Parse results
        results_dir = Path("diffusion_pipeline_results") / experiment_name
        eval_results_path = results_dir / "evaluation" / "results.json"
        pipeline_summary_path = results_dir / "pipeline_summary.json"

        if not eval_results_path.exists() or not pipeline_summary_path.exists():
            print(f"❌ Results files not found for {experiment_name}")
            return {
                "experiment_name": experiment_name,
                "env": env,
                "policy_type": policy_type,
                "data_episodes": data_episodes,
                "train_epochs": train_epochs,
                "eval_episodes": eval_episodes,
                "success": False,
                "error": "Results files not found",
                "runtime": end_time - start_time,
            }

        # Load results
        with open(eval_results_path, "r") as f:
            eval_results = json.load(f)

        with open(pipeline_summary_path, "r") as f:
            pipeline_summary = json.load(f)

        # Load dataset metadata for expert statistics
        dataset_metadata_path = (
            results_dir
            / "datasets"
            / f"{env}_expert_{data_episodes}ep"
            / "metadata.json"
        )
        expert_stats = {}
        if dataset_metadata_path.exists():
            with open(dataset_metadata_path, "r") as f:
                expert_stats = json.load(f)

        print(f"✅ Experiment completed successfully!")
        print(f"📊 Results:")
        print(f"   - Mean Return: {eval_results['mean_return']:.2f}")
        print(f"   - Success Rate: {eval_results['success_rate']:.2f}%")
        print(
            f"   - Expert Success Rate: {expert_stats.get('success_rate', 0.0) * 100:.2f}%"
        )
        print(f"   - Runtime: {end_time - start_time:.1f}s")

        return {
            "experiment_name": experiment_name,
            "env": env,
            "policy_type": policy_type,
            "data_episodes": data_episodes,
            "train_epochs": train_epochs,
            "eval_episodes": eval_episodes,
            "success": True,
            "mean_return": eval_results["mean_return"],
            "std_return": eval_results["std_return"],
            "success_rate": eval_results["success_rate"],
            "mean_length": eval_results["mean_length"],
            "expert_success_rate": expert_stats.get("success_rate", 0.0)
            * 100,  # Convert to percentage
            "expert_avg_reward": expert_stats.get("average_reward", 0.0),
            "total_expert_episodes": expert_stats.get("num_episodes", 0),
            "successful_expert_episodes": expert_stats.get("successful_episodes", 0),
            "runtime": end_time - start_time,
            "results_dir": str(results_dir),
        }

    except subprocess.TimeoutExpired:
        print(f"❌ Experiment timed out after 1 hour")
        return {
            "experiment_name": experiment_name,
            "env": env,
            "policy_type": policy_type,
            "data_episodes": data_episodes,
            "train_epochs": train_epochs,
            "eval_episodes": eval_episodes,
            "success": False,
            "error": "Timeout",
            "runtime": 3600,
        }
    except Exception as e:
        print(f"❌ Experiment failed with exception: {e}")
        return {
            "experiment_name": experiment_name,
            "env": env,
            "policy_type": policy_type,
            "data_episodes": data_episodes,
            "train_epochs": train_epochs,
            "eval_episodes": eval_episodes,
            "success": False,
            "error": str(e),
            "runtime": 0,
        }


def run_scaling_experiments(
    env: str = "motion2d-p1",
    policy_type: str = "behavior_cloning",
    demo_counts: List[int] = None,
    train_epochs: int = 1,
    eval_episodes: int = 1,
    experiment_dir: str = None,
) -> List[Dict[str, Any]]:
    """Run scaling experiments with different numbers of demonstrations.

    Args:
        env: Environment name
        policy_type: Policy type
        demo_counts: List of demonstration counts to test
        train_epochs: Number of training epochs per experiment
        eval_episodes: Number of evaluation episodes per experiment
        experiment_dir: Directory to save intermediate results (optional)

    Returns:
        List of experiment results
    """
    if demo_counts is None:
        demo_counts = [1, 2, 5, 10, 20, 50]

    print(f"\n{'='*100}")
    print(f"🚀 STARTING SCALING EXPERIMENT")
    print(f"{'='*100}")
    print(f"Environment: {env}")
    print(f"Policy: {policy_type}")
    print(f"Demonstration counts: {demo_counts}")
    print(f"Training epochs per experiment: {train_epochs}")
    print(f"Evaluation episodes per experiment: {eval_episodes}")
    print(f"Total experiments: {len(demo_counts)}")
    print(f"{'='*100}")

    results = []

    for i, demo_count in enumerate(demo_counts):
        print(f"\n📈 Progress: {i+1}/{len(demo_counts)} experiments")

        result = run_experiment(
            env=env,
            policy_type=policy_type,
            data_episodes=demo_count,
            train_epochs=train_epochs,
            eval_episodes=eval_episodes,
            experiment_prefix="scaling_exp",
        )

        results.append(result)

        # Save intermediate results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_dir:
            # Ensure the directory exists
            os.makedirs(experiment_dir, exist_ok=True)
            results_file = (
                f"{experiment_dir}/scaling_results_{env}_{policy_type}_{timestamp}.json"
            )
        else:
            results_file = f"scaling_results_{env}_{policy_type}_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"💾 Intermediate results saved to: {results_file}")

    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(
        f"Total successful experiments: {sum(1 for r in results if r['success'])}/{len(results)}"
    )

    return results


def create_scaling_figure(
    results: List[Dict[str, Any]], save_path: str = None, show_plot: bool = True
) -> None:
    """Create a figure showing success rate vs number of demonstrations.

    Args:
        results: List of experiment results
        save_path: Path to save the figure (optional)
        show_plot: Whether to display the plot
    """
    # Filter successful experiments
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        print("❌ No successful experiments to plot!")
        return

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(successful_results)
    df = df.sort_values("data_episodes")

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'Behavior Cloning Scaling Analysis: {df.iloc[0]["env"]}',
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Success Rate vs Demonstrations
    ax1.plot(df["data_episodes"], df["success_rate"], "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Demonstrations")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_title("Policy Success Rate vs Demonstrations")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-5, 105)

    # Add value labels on points
    for _, row in df.iterrows():
        ax1.annotate(
            f'{row["success_rate"]:.1f}%',
            (row["data_episodes"], row["success_rate"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Plot 2: Mean Return vs Demonstrations
    ax2.errorbar(
        df["data_episodes"],
        df["mean_return"],
        yerr=df["std_return"],
        fmt="ro-",
        linewidth=2,
        markersize=8,
        capsize=5,
    )
    ax2.set_xlabel("Number of Demonstrations")
    ax2.set_ylabel("Mean Return")
    ax2.set_title("Mean Return vs Demonstrations")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Expert Success Rate vs Demonstrations
    ax3.plot(
        df["data_episodes"], df["expert_success_rate"], "go-", linewidth=2, markersize=8
    )
    ax3.set_xlabel("Number of Demonstrations")
    ax3.set_ylabel("Expert Success Rate (%)")
    ax3.set_title("Expert Demonstration Quality")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-5, 105)

    # Plot 4: Episode Length vs Demonstrations
    ax4.plot(df["data_episodes"], df["mean_length"], "mo-", linewidth=2, markersize=8)
    ax4.set_xlabel("Number of Demonstrations")
    ax4.set_ylabel("Mean Episode Length")
    ax4.set_title("Episode Length vs Demonstrations")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"📊 Figure saved to: {save_path}")

    # Show plot if requested
    if show_plot:
        plt.show()

    # Print summary statistics
    print(f"\n📊 SCALING ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(
        f"{'Demos':<8} {'Success Rate':<12} {'Mean Return':<12} {'Expert Success':<15}"
    )
    print(f"{'='*50}")

    for _, row in df.iterrows():
        print(
            f"{row['data_episodes']:<8} {row['success_rate']:<12.1f} {row['mean_return']:<12.2f} {row['expert_success_rate']:<15.1f}"
        )

    # Calculate correlations
    if len(df) > 1:
        success_corr = np.corrcoef(df["data_episodes"], df["success_rate"])[0, 1]
        return_corr = np.corrcoef(df["data_episodes"], df["mean_return"])[0, 1]

        print(f"\n📈 CORRELATIONS:")
        print(f"Success Rate vs Demonstrations: {success_corr:.3f}")
        print(f"Mean Return vs Demonstrations: {return_corr:.3f}")


def main():
    """Main function to run the scaling experiment."""
    print("🧪 BEHAVIOR CLONING SCALING EXPERIMENT")
    print("=" * 60)

    # Configuration
    env = "motion2d-p1"
    policy_type = "behavior_cloning"
    demo_counts = [1, 2, 5, 10, 20]  # Start with smaller numbers for testing
    train_epochs = 1  # Fast for testing
    eval_episodes = 1  # Fast for testing

    print(f"Configuration:")
    print(f"  Environment: {env}")
    print(f"  Policy: {policy_type}")
    print(f"  Demo counts: {demo_counts}")
    print(f"  Train epochs: {train_epochs}")
    print(f"  Eval episodes: {eval_episodes}")

    # Run experiments
    results = run_scaling_experiments(
        env=env,
        policy_type=policy_type,
        demo_counts=demo_counts,
        train_epochs=train_epochs,
        eval_episodes=eval_episodes,
    )

    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"final_scaling_results_{env}_{policy_type}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Final results saved to: {results_file}")

    # Create visualization
    figure_path = f"scaling_analysis_{env}_{policy_type}_{timestamp}.png"
    create_scaling_figure(results, save_path=figure_path, show_plot=False)

    print(f"\n🎉 EXPERIMENT COMPLETED!")
    print(f"Results file: {results_file}")
    print(f"Figure file: {figure_path}")


if __name__ == "__main__":
    main()
