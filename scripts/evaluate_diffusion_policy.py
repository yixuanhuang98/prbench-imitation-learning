#!/usr/bin/env python3
"""
Script to evaluate a trained diffusion policy on geom2d environments.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import gymnasium as gym

# Add the prbench module to path
sys.path.append('/home/yixuan/prbench_dir/prbench/src')
import prbench

# Import the policy from training script
from train_diffusion_policy import DiffusionPolicy


class PolicyEvaluator:
    """Evaluator for trained diffusion policies."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run evaluation on ("cpu", "cuda", or "auto")
        """
        self.model_path = model_path
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model, self.config = self._load_model()
        self.model.eval()
        
        # Initialize observation buffer
        self.obs_buffer = deque(maxlen=self.config["obs_horizon"])
        
    def _load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint["config"]
        
        # Get model dimensions from config or checkpoint
        # Note: These should be saved in the checkpoint, but we'll use defaults if not
        obs_dim = config.get("obs_dim", 4)  # Default for geom2d environments
        action_dim = config.get("action_dim", 5)  # dx, dy, dtheta, darm, vacuum
        
        model = DiffusionPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            obs_horizon=config["obs_horizon"],
            action_horizon=config["action_horizon"],
            num_diffusion_iters=config["num_diffusion_iters"]
        ).to(self.device)
        
        model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"Model loaded successfully. Loss: {checkpoint['loss']:.6f}")
        print(f"Model config: {config}")
        
        return model, config
    
    def reset_observation_buffer(self, initial_obs: np.ndarray):
        """Reset observation buffer with initial observation."""
        self.obs_buffer.clear()
        # Fill buffer with initial observation
        for _ in range(self.config["obs_horizon"]):
            self.obs_buffer.append(initial_obs.copy())
    
    def predict_action(self, obs: np.ndarray, env: gym.Env = None) -> np.ndarray:
        """Predict action given current observation."""
        # Add observation to buffer
        self.obs_buffer.append(obs.copy())
        
        # Stack observations
        obs_seq = np.stack(list(self.obs_buffer), axis=0)
        obs_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)
        
        # Predict actions
        with torch.no_grad():
            action_seq = self.model(obs_tensor)
        
        # Return first action from sequence
        action = action_seq[0, 0].cpu().numpy()
        
        # Clip action to valid range if environment provided
        if env is not None:
            action = np.clip(action, env.action_space.low, env.action_space.high)
        
        return action
    
    def evaluate_episodes(self, env_id: str, num_episodes: int = 10, max_steps: int = 200, 
                         render: bool = False, save_videos: bool = False) -> Dict[str, Any]:
        """Evaluate policy on multiple episodes."""
        
        # Setup environment
        prbench.register_all_environments()
        env = gym.make(env_id, render_mode="rgb_array" if render or save_videos else None)
        
        results = {
            "episode_returns": [],
            "episode_lengths": [],
            "success_rate": 0.0,
            "average_return": 0.0,
            "average_length": 0.0,
        }
        
        videos = []
        
        for episode in range(num_episodes):
            print(f"Evaluating episode {episode + 1}/{num_episodes}")
            
            obs, info = env.reset()
            self.reset_observation_buffer(obs)
            
            episode_return = 0.0
            episode_length = 0
            episode_frames = []
            done = False
            
            while not done and episode_length < max_steps:
                # Predict action
                action = self.predict_action(obs, env)
                
                # Take step
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_return += reward
                episode_length += 1
                
                # Render if requested
                if render or save_videos:
                    frame = env.render()
                    episode_frames.append(frame)
                
                # Print progress
                if episode_length % 50 == 0:
                    print(f"  Step {episode_length}, Reward: {reward:.3f}, Total: {episode_return:.3f}")
            
            # Store results
            results["episode_returns"].append(episode_return)
            results["episode_lengths"].append(episode_length)
            
            if save_videos:
                videos.append(episode_frames)
            
            print(f"Episode {episode + 1} completed: Return={episode_return:.3f}, Length={episode_length}")
        
        # Compute summary statistics
        returns = results["episode_returns"]
        lengths = results["episode_lengths"]
        
        results["average_return"] = np.mean(returns)
        results["std_return"] = np.std(returns)
        results["average_length"] = np.mean(lengths)
        results["std_length"] = np.std(lengths)
        
        # Success rate (assuming positive reward indicates success)
        results["success_rate"] = np.mean([r > 0 for r in returns])
        
        env.close()
        
        if save_videos:
            results["videos"] = videos
        
        return results
    
    def evaluate_single_episode(self, env_id: str, max_steps: int = 200, render: bool = True, 
                               save_trajectory: bool = False) -> Dict[str, Any]:
        """Evaluate policy on a single episode with detailed logging."""
        
        # Setup environment
        prbench.register_all_environments()
        env = gym.make(env_id, render_mode="rgb_array")
        
        obs, info = env.reset()
        self.reset_observation_buffer(obs)
        
        trajectory = {
            "observations": [obs.copy()],
            "actions": [],
            "rewards": [],
            "frames": [],
        }
        
        episode_return = 0.0
        episode_length = 0
        done = False
        
        print("Starting single episode evaluation...")
        
        while not done and episode_length < max_steps:
            # Predict action
            action = self.predict_action(obs, env)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
            
            # Store trajectory data
            if save_trajectory:
                trajectory["observations"].append(obs.copy())
                trajectory["actions"].append(action.copy())
                trajectory["rewards"].append(reward)
            
            # Render
            if render:
                frame = env.render()
                trajectory["frames"].append(frame)
            
            # Print detailed progress
            print(f"Step {episode_length}: Action={action}, Reward={reward:.3f}, Obs={obs[:4]}")
        
        env.close()
        
        results = {
            "total_return": episode_return,
            "episode_length": episode_length,
            "success": episode_return > 0,
            "trajectory": trajectory if save_trajectory else None,
            "frames": trajectory["frames"] if render else None,
        }
        
        print(f"Episode completed: Return={episode_return:.3f}, Length={episode_length}, Success={results['success']}")
        
        return results


def create_evaluation_video(frames: List[np.ndarray], save_path: str, fps: int = 10):
    """Create video from episode frames."""
    if not frames:
        print("No frames to create video")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Display first frame
    im = ax.imshow(frames[0])
    
    def animate(frame_idx):
        im.set_array(frames[frame_idx])
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(frames), 
                                 interval=1000//fps, blit=True, repeat=True)
    
    # Save video
    print(f"Saving video to {save_path}")
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close(fig)


def plot_evaluation_results(results: Dict[str, Any], save_path: str = None):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode returns
    axes[0, 0].plot(results["episode_returns"])
    axes[0, 0].axhline(y=results["average_return"], color='r', linestyle='--', 
                      label=f'Average: {results["average_return"]:.3f}')
    axes[0, 0].set_title("Episode Returns")
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(results["episode_lengths"])
    axes[0, 1].axhline(y=results["average_length"], color='r', linestyle='--',
                      label=f'Average: {results["average_length"]:.1f}')
    axes[0, 1].set_title("Episode Lengths")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Length")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Return histogram
    axes[1, 0].hist(results["episode_returns"], bins=10, alpha=0.7)
    axes[1, 0].axvline(x=results["average_return"], color='r', linestyle='--',
                      label=f'Average: {results["average_return"]:.3f}')
    axes[1, 0].set_title("Return Distribution")
    axes[1, 0].set_xlabel("Return")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Summary statistics
    stats_text = f"""
    Success Rate: {results["success_rate"]:.1%}
    Average Return: {results["average_return"]:.3f} ± {results["std_return"]:.3f}
    Average Length: {results["average_length"]:.1f} ± {results["std_length"]:.1f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
    axes[1, 1].set_title("Summary Statistics")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained diffusion policy")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--env", type=str, default="prbench/Motion2D-p2-v0",
                       help="Environment ID to evaluate on")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--max-steps", type=int, default=200,
                       help="Maximum steps per episode")
    parser.add_argument("--single-episode", action="store_true",
                       help="Run single episode with detailed logging")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--save-videos", action="store_true",
                       help="Save episode videos")
    parser.add_argument("--save-plots", action="store_true",
                       help="Save evaluation plots")
    parser.add_argument("--output-dir", type=str, default="./evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                       help="Device to run evaluation on")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Diffusion Policy Evaluation")
    print("="*50)
    print(f"Model path: {args.model_path}")
    print(f"Environment: {args.env}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Device: {args.device}")
    print("="*50)
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator
        evaluator = PolicyEvaluator(args.model_path, args.device)
        
        if args.single_episode:
            # Single episode evaluation
            print("\nRunning single episode evaluation...")
            results = evaluator.evaluate_single_episode(
                env_id=args.env,
                max_steps=args.max_steps,
                render=args.render,
                save_trajectory=True
            )
            
            print(f"\nSingle Episode Results:")
            print(f"Total Return: {results['total_return']:.3f}")
            print(f"Episode Length: {results['episode_length']}")
            print(f"Success: {results['success']}")
            
            # Save video if frames available
            if results["frames"] and args.save_videos:
                video_path = output_dir / "single_episode.gif"
                create_evaluation_video(results["frames"], str(video_path))
            
            # Save trajectory
            if results["trajectory"]:
                trajectory_path = output_dir / "single_episode_trajectory.json"
                trajectory_data = {
                    "observations": [obs.tolist() for obs in results["trajectory"]["observations"]],
                    "actions": [action.tolist() for action in results["trajectory"]["actions"]],
                    "rewards": results["trajectory"]["rewards"],
                }
                with open(trajectory_path, 'w') as f:
                    json.dump(trajectory_data, f, indent=2)
                print(f"Trajectory saved to {trajectory_path}")
        
        else:
            # Multi-episode evaluation
            print(f"\nRunning {args.num_episodes} episode evaluation...")
            results = evaluator.evaluate_episodes(
                env_id=args.env,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                render=args.render,
                save_videos=args.save_videos
            )
            
            print(f"\nEvaluation Results:")
            print(f"Success Rate: {results['success_rate']:.1%}")
            print(f"Average Return: {results['average_return']:.3f} ± {results['std_return']:.3f}")
            print(f"Average Length: {results['average_length']:.1f} ± {results['std_length']:.1f}")
            
            # Save results
            results_path = output_dir / "evaluation_results.json"
            results_to_save = {k: v for k, v in results.items() if k != "videos"}
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2)
            print(f"Results saved to {results_path}")
            
            # Create plots
            if args.save_plots:
                plot_path = output_dir / "evaluation_plots.png"
                plot_evaluation_results(results, str(plot_path))
            else:
                plot_evaluation_results(results)
            
            # Save videos
            if args.save_videos and "videos" in results:
                for i, video_frames in enumerate(results["videos"]):
                    video_path = output_dir / f"episode_{i+1}.gif"
                    create_evaluation_video(video_frames, str(video_path))
        
        print("\n" + "="*50)
        print("SUCCESS: Evaluation completed!")
        print(f"Results saved to: {output_dir}")
        print("="*50)
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
