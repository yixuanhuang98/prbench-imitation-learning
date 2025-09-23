#!/usr/bin/env python3
"""
Comprehensive scaling experiment for behavior cloning with motion2d-p1 environment.

This script runs a full scaling study with reasonable parameters for meaningful results.
"""

from run_scaling_experiment import run_scaling_experiments, create_scaling_figure
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from datetime import datetime

def main():
    """Run comprehensive scaling experiment."""
    print("🚀 COMPREHENSIVE BEHAVIOR CLONING SCALING EXPERIMENT")
    print("="*70)
    
    # Configuration for comprehensive experiment
    env = "motion2d-p1"
    policy_type = "behavior_cloning"
    demo_counts = [1, 2, 5, 10, 20, 50]  # Full range
    train_epochs = 10000  # More epochs for better training
    eval_episodes = 10  # More episodes for better statistics
    
    print(f"Comprehensive configuration:")
    print(f"  Environment: {env}")
    print(f"  Policy: {policy_type}")
    print(f"  Demo counts: {demo_counts}")
    print(f"  Train epochs: {train_epochs}")
    print(f"  Eval episodes: {eval_episodes}")
    print(f"  Estimated total runtime: ~{len(demo_counts) * 5} minutes")
    print()
    
    # Ask for confirmation
    response = input("🤔 This will take some time. Continue? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Experiment cancelled.")
        return
    
    # Run experiments
    print("\n🚀 Starting comprehensive scaling experiment...")
    results = run_scaling_experiments(
        env=env,
        policy_type=policy_type,
        demo_counts=demo_counts,
        train_epochs=train_epochs,
        eval_episodes=eval_episodes
    )
    
    # Save final results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_scaling_results_{env}_{policy_type}_{timestamp}.json"
    
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Final results saved to: {results_file}")
    
    # Create comprehensive visualization
    figure_path = f"comprehensive_scaling_analysis_{env}_{policy_type}_{timestamp}.png"
    create_scaling_figure(results, save_path=figure_path, show_plot=False)
    
    print(f"\n🎉 COMPREHENSIVE EXPERIMENT COMPLETED!")
    print(f"Results file: {results_file}")
    print(f"Figure file: {figure_path}")
    
    # Print summary
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n📊 FINAL SUMMARY:")
        print(f"Successful experiments: {len(successful_results)}/{len(results)}")
        print(f"Best policy success rate: {max(r['success_rate'] for r in successful_results):.1f}%")
        print(f"Expert success rate: {successful_results[0]['expert_success_rate']:.1f}%")
        
        # Check if there's improvement with more demonstrations
        if len(successful_results) > 1:
            first_success = successful_results[0]['success_rate']
            last_success = successful_results[-1]['success_rate']
            if last_success > first_success:
                print(f"✅ Improvement observed: {first_success:.1f}% → {last_success:.1f}%")
            else:
                print(f"⚠️  No clear improvement: {first_success:.1f}% → {last_success:.1f}%")

if __name__ == "__main__":
    main()
