#!/usr/bin/env python3
"""
Complete pipeline script for generating data, training, and evaluating diffusion policies on geom2d environments.
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path
import time


def run_command(command: str, description: str = "", check: bool = True):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description or command}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    result = subprocess.run(command, shell=True, check=check, 
                          capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Command failed with return code {result.returncode}")
        if check:
            sys.exit(1)
    else:
        print(f"SUCCESS: {description or 'Command completed'}")
    
    return result


def generate_data(env_name: str, dataset_name: str, num_episodes: int, data_type: str, output_dir: str):
    """Generate LeRobot dataset."""
    command = f"""
    python generate_lerobot_data.py \\
        --env {env_name} \\
        --dataset-name {dataset_name} \\
        --num-episodes {num_episodes} \\
        --data-type {data_type} \\
        --output-dir {output_dir}
    """
    
    run_command(command.strip(), f"Generating {data_type} dataset for {env_name}")
    
    return str(Path(output_dir) / dataset_name)


def train_model(dataset_path: str, model_save_path: str, config: dict):
    """Train diffusion policy."""
    
    # Save config to temporary file
    config_path = "/tmp/diffusion_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    command = f"""
    python train_diffusion_policy.py \\
        --dataset-path {dataset_path} \\
        --model-save-path {model_save_path} \\
        --config {config_path} \\
        --batch-size {config['batch_size']} \\
        --num-epochs {config['num_epochs']} \\
        --learning-rate {config['learning_rate']}
    """
    
    if config.get('use_wandb', False):
        command += " --use-wandb"
    
    run_command(command.strip(), "Training diffusion policy")
    
    # Clean up config file
    os.remove(config_path)


def evaluate_model(model_path: str, env_id: str, num_episodes: int, output_dir: str, 
                  render: bool = False, save_videos: bool = False, save_plots: bool = True):
    """Evaluate trained model."""
    
    command_parts = [
        # "CUDA_VISIBLE_DEVICES=''",
        "python evaluate_diffusion_policy.py",
        f"--model-path {model_path}",
        f"--env {env_id}",
        f"--num-episodes {num_episodes}",
        f"--output-dir {output_dir}",
        "--device cpu"
    ]
    
    if render:
        command_parts.append("--render")
    if save_videos:
        command_parts.append("--save-videos")
    if save_plots:
        command_parts.append("--save-plots")
    
    command = " ".join(command_parts)
    
    run_command(command.strip(), "Evaluating trained model")


def get_env_id(env_name: str) -> str:
    """Get full environment ID from short name."""
    env_map = {
        "motion2d": "prbench/Motion2D-p2-v0",
        "pushpullhook2d": "prbench/PushPullHook2D-v0", 
        "stickbutton2d": "prbench/StickButton2D-b2-v0",
        "clutteredretrieval2d": "prbench/ClutteredRetrieval2D-o10-v0",
        "clutteredstorage2d": "prbench/ClutteredStorage2D-b3-v0",
        "obstruction2d": "prbench/Obstruction2D-o2-v0"
    }
    return env_map.get(env_name, env_name)


def main():
    parser = argparse.ArgumentParser(description="Complete diffusion policy pipeline for geom2d environments")
    
    # Environment and data options
    parser.add_argument("--env", type=str, default="motion2d",
                       choices=["motion2d", "pushpullhook2d", "stickbutton2d", 
                               "clutteredretrieval2d", "clutteredstorage2d", "obstruction2d"],
                       help="Environment name")
    parser.add_argument("--data-episodes", type=int, default=20,
                       help="Number of episodes for data collection")
    parser.add_argument("--data-type", type=str, default="expert", choices=["random", "expert"],
                       help="Type of data to collect")
    
    # Training options
    parser.add_argument("--train-epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    
    # Evaluation options
    parser.add_argument("--eval-episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--render", action="store_true",
                       help="Render during evaluation")
    parser.add_argument("--save-videos", action="store_true",
                       help="Save evaluation videos")
    
    # Pipeline control
    parser.add_argument("--skip-data", action="store_true",
                       help="Skip data generation (use existing dataset)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (use existing model)")
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip evaluation")
    parser.add_argument("--dataset-path", type=str,
                       help="Path to existing dataset (if skipping data generation)")
    parser.add_argument("--model-path", type=str,
                       help="Path to existing model (if skipping training)")
    
    # Output options
    parser.add_argument("--output-dir", type=str, default="./diffusion_pipeline_results",
                       help="Output directory for all results")
    parser.add_argument("--experiment-name", type=str,
                       help="Experiment name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if not args.experiment_name:
        timestamp = int(time.time())
        args.experiment_name = f"{args.env}_{args.data_type}_{timestamp}"
    
    # Setup directories
    output_dir = Path(args.output_dir) / args.experiment_name
    dataset_dir = output_dir / "datasets"
    model_dir = output_dir / "models"
    eval_dir = output_dir / "evaluation"
    
    for dir_path in [output_dir, dataset_dir, model_dir, eval_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("DIFFUSION POLICY PIPELINE FOR GEOM2D ENVIRONMENTS")
    print("="*80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Environment: {args.env}")
    print(f"Output directory: {output_dir}")
    print(f"Steps: {'Data' if not args.skip_data else 'Skip Data'} -> "
          f"{'Train' if not args.skip_training else 'Skip Train'} -> "
          f"{'Eval' if not args.skip_evaluation else 'Skip Eval'}")
    print("="*80)
    
    # Save experiment configuration
    experiment_config = {
        "experiment_name": args.experiment_name,
        "environment": args.env,
        "data_episodes": args.data_episodes,
        "data_type": args.data_type,
        "train_epochs": args.train_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "eval_episodes": args.eval_episodes,
        "timestamp": time.time(),
    }
    
    config_path = output_dir / "experiment_config.json"
    with open(config_path, 'w') as f:
        json.dump(experiment_config, f, indent=2)
    print(f"Experiment config saved to: {config_path}")
    
    dataset_path = None
    model_path = None
    
    try:
        # Step 1: Data Generation
        if not args.skip_data:
            print(f"\n🔄 STEP 1: Generating {args.data_type} data for {args.env}")
            dataset_name = f"{args.env}_{args.data_type}_{args.data_episodes}ep"
            dataset_path = generate_data(
                env_name=args.env,
                dataset_name=dataset_name,
                num_episodes=args.data_episodes,
                data_type=args.data_type,
                output_dir=str(dataset_dir)
            )
            print(f"✅ Data generation completed: {dataset_path}")
        else:
            dataset_path = args.dataset_path
            if not dataset_path:
                raise ValueError("Must provide --dataset-path when skipping data generation")
            print(f"⏭️  Skipping data generation, using: {dataset_path}")
        
        # Step 2: Training
        if not args.skip_training:
            print(f"\n🔄 STEP 2: Training diffusion policy")
            
            # Training configuration
            train_config = {
                "obs_horizon": 2,
                "action_horizon": 8,
                "pred_horizon": 8,
                "num_diffusion_iters": 100,
                "batch_size": args.batch_size,
                "num_epochs": args.train_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": 1e-6,
                "grad_clip_norm": 1.0,
                "num_workers": 4,
                "log_interval": 10,
                "use_wandb": args.use_wandb,
            }
            
            model_path = str(model_dir / f"{args.experiment_name}_model.pth")
            train_model(dataset_path, model_path, train_config)
            print(f"✅ Training completed: {model_path}")
        else:
            model_path = args.model_path
            if not model_path:
                raise ValueError("Must provide --model-path when skipping training")
            print(f"⏭️  Skipping training, using: {model_path}")
        
        # Step 3: Evaluation
        if not args.skip_evaluation:
            print(f"\n🔄 STEP 3: Evaluating trained policy")
            env_id = get_env_id(args.env)
            evaluate_model(
                model_path=model_path,
                env_id=env_id,
                num_episodes=args.eval_episodes,
                output_dir=str(eval_dir),
                render=args.render,
                save_videos=args.save_videos,
                save_plots=True
            )
            print(f"✅ Evaluation completed: {eval_dir}")
        else:
            print("⏭️  Skipping evaluation")
        
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
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Experiment: {args.experiment_name}")
        print(f"Output directory: {output_dir}")
        if dataset_path:
            print(f"Dataset: {dataset_path}")
        if model_path:
            print(f"Model: {model_path}")
        if not args.skip_evaluation:
            print(f"Evaluation: {eval_dir}")
        print(f"Summary: {summary_path}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Save error info
        error_info = {
            "experiment_name": args.experiment_name,
            "status": "failed",
            "error": str(e),
            "failed_at": time.time(),
        }
        
        error_path = output_dir / "pipeline_error.json"
        with open(error_path, 'w') as f:
            json.dump(error_info, f, indent=2)
        
        sys.exit(1)


if __name__ == "__main__":
    main()
