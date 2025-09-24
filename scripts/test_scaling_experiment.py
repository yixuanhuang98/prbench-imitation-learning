#!/usr/bin/env python3
"""Quick test of the scaling experiment with minimal parameters."""

import matplotlib
from run_scaling_experiment import create_scaling_figure, run_scaling_experiments

matplotlib.use("Agg")  # Use non-interactive backend


def main():
    """Test the scaling experiment with minimal parameters."""
    print("🧪 TESTING SCALING EXPERIMENT")
    print("=" * 50)

    # Very minimal test configuration
    env = "motion2d-p1"
    policy_type = "behavior_cloning"
    demo_counts = [1, 2]  # Just 2 experiments for testing
    train_epochs = 1
    eval_episodes = 1

    print("Test configuration:")
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

    # Create visualization (save only, don't show)
    if results:
        create_scaling_figure(
            results, save_path="test_scaling_plot.png", show_plot=False
        )
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed - no results generated")


if __name__ == "__main__":
    main()
