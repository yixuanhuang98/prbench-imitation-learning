"""Evaluation functionality for trained diffusion policies."""

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import prbench
import torch

from .policy import DiffusionPolicy


class PolicyEvaluator:
    """Evaluator for trained diffusion policies."""

    def __init__(self, model_path: str, device: str = "auto", log_dir: str = "./logs"):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run evaluation on ("cpu", "cuda", or "auto")
            log_dir: Directory to save logs
        """
        self.model_path = model_path
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Load model
        self.model, self.config = self._load_model()

        # Setup observation history buffer
        self.obs_history = deque(maxlen=self.config["obs_horizon"])

    def _load_model(self) -> Tuple[DiffusionPolicy, Dict[str, Any]]:
        """Load the trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        config = checkpoint["config"]

        # Create model
        model = DiffusionPolicy(
            obs_dim=config["obs_dim"],
            action_dim=config["action_dim"],
            obs_horizon=config["obs_horizon"],
            action_horizon=config["action_horizon"],
            num_diffusion_iters=config["num_diffusion_iters"],
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        print(
            f"Model loaded successfully (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})"
        )
        return model, config

    def predict_action(self, observation) -> np.ndarray:
        """Predict action for a single observation.

        Args:
            observation: Dictionary containing 'state' and optionally 'image', or numpy array

        Returns:
            Predicted action array
        """
        # Handle different observation formats
        if isinstance(observation, dict):
            obs_state = observation["state"]
        else:
            # Assume observation is the state directly
            obs_state = observation

        self.obs_history.append(obs_state)

        # If we don't have enough history, pad with the current observation
        while len(self.obs_history) < self.config["obs_horizon"]:
            self.obs_history.append(obs_state)

        # Prepare observation sequence
        obs_seq = np.stack(list(self.obs_history))
        obs_seq_tensor = torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)

        # Predict action sequence
        with torch.no_grad():
            action_seq = self.model(obs_seq_tensor)

        # Return first action
        predicted_action = action_seq[0, 0].cpu().numpy()

        # Ensure action is finite and not NaN
        if not np.isfinite(predicted_action).all():
            print(f"Warning: Non-finite action predicted: {predicted_action}")
            predicted_action = np.nan_to_num(
                predicted_action, nan=0.0, posinf=1.0, neginf=-1.0
            )

        return predicted_action

    def reset(self):
        """Reset the observation history."""
        self.obs_history.clear()

    def evaluate_policy(
        self,
        env_id: str,
        num_episodes: int = 10,
        render: bool = False,
        save_videos: bool = False,
        save_plots: bool = True,
        output_dir: str = None,
        max_episode_steps: int = 1000,
    ) -> Dict[str, Any]:
        """Evaluate the policy on the environment.

        Args:
            env_id: Environment ID to evaluate on
            num_episodes: Number of episodes to run
            render: Whether to render during evaluation
            save_videos: Whether to save video recordings
            save_plots: Whether to save evaluation plots
            output_dir: Directory to save results
            max_episode_steps: Maximum steps per episode to prevent infinite loops

        Returns:
            Dictionary containing evaluation results
        """
        if output_dir is None:
            output_dir = self.log_dir / "evaluation"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup environment
        try:
            prbench.register_all_environments()

            env = gym.make(env_id)
        except Exception as e:
            print(f"Failed to create environment {env_id}: {e}")
            raise

        print(f"Evaluating on environment: {env_id}")
        print(f"Number of episodes: {num_episodes}")

        # Evaluation metrics
        episode_returns = []
        episode_lengths = []
        success_rates = []
        all_trajectories = []

        # Setup logging
        eval_log_path = output_dir / "evaluation.log"

        def log_message(message: str):
            """Log message to both console and file."""
            print(message)
            with open(eval_log_path, "a") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

        log_message(f"Starting evaluation on {env_id}")
        log_message(f"Model: {self.model_path}")
        log_message(f"Config: {self.config}")

        for episode in range(num_episodes):
            log_message(f"Starting episode {episode+1}/{num_episodes}")
            self.reset()
            obs, info = env.reset()

            episode_return = 0.0
            episode_length = 0
            trajectory = []

            if save_videos:
                frames = []

            done = False
            while not done and episode_length < max_episode_steps:
                # Get action from policy
                action = self.predict_action(obs)

                # Clip action to valid action space
                if hasattr(env.action_space, "low") and hasattr(
                    env.action_space, "high"
                ):
                    action = np.clip(
                        action, env.action_space.low, env.action_space.high
                    )

                # Take step in environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Record data
                episode_return += reward
                episode_length += 1

                # Log progress every 100 steps to show activity
                if episode_length % 100 == 0:
                    log_message(
                        f"  Episode {episode+1}, Step {episode_length}, Reward: {episode_return:.2f}"
                    )

                trajectory.append(
                    {
                        "obs": obs.copy() if hasattr(obs, "copy") else obs,
                        "action": action.copy() if hasattr(action, "copy") else action,
                        "reward": reward,
                        "next_obs": (
                            next_obs.copy() if hasattr(next_obs, "copy") else next_obs
                        ),
                        "done": done,
                        "info": info.copy() if hasattr(info, "copy") else info,
                    }
                )

                if save_videos and render:
                    try:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    except:
                        pass  # Skip if rendering fails

                obs = next_obs

            # Check if episode was truncated due to max steps
            if episode_length >= max_episode_steps and not done:
                log_message(
                    f"  Episode {episode+1} truncated at {episode_length} steps"
                )

            # Record episode metrics
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)

            # Check for success (if info contains success flag)
            success = info.get("success", episode_return > 0)  # Fallback heuristic
            success_rates.append(float(success))
            all_trajectories.append(trajectory)

            log_message(
                f"Episode {episode+1}/{num_episodes}: "
                f"Return={episode_return:.2f}, Length={episode_length}, Success={success}"
            )

            # Save video if requested
            if save_videos and "frames" in locals() and frames:
                self._save_video(frames, output_dir / f"episode_{episode}.mp4")

        env.close()

        # Compute summary statistics
        results = {
            "num_episodes": num_episodes,
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            "mean_length": np.mean(episode_lengths),
            "std_length": np.std(episode_lengths),
            "success_rate": np.mean(success_rates),
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "success_rates": success_rates,
            "model_path": str(self.model_path),
            "env_id": env_id,
            "config": self.config,
        }

        # Log summary
        log_message("Evaluation Summary:")
        log_message(
            f"Mean Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}"
        )
        log_message(
            f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}"
        )
        log_message(f"Success Rate: {results['success_rate']:.2%}")

        # Save results
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        log_message(f"Results saved to {results_path}")

        # Save plots if requested
        if save_plots:
            self._save_plots(results, output_dir)
            log_message(f"Plots saved to {output_dir}")

        return results

    def _save_video(self, frames: List[np.ndarray], video_path: Path):
        """Save video from frames."""
        try:
            fig, ax = plt.subplots()
            ax.axis("off")

            def animate(frame_idx):
                ax.clear()
                ax.imshow(frames[frame_idx])
                ax.axis("off")
                return []

            anim = animation.FuncAnimation(
                fig, animate, frames=len(frames), interval=50, blit=False
            )
            anim.save(str(video_path), writer="pillow", fps=20)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to save video: {e}")

    def _save_plots(self, results: Dict[str, Any], output_dir: Path):
        """Save evaluation plots."""
        try:
            # Episode returns plot
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(results["episode_returns"], "b-", alpha=0.7)
            plt.axhline(
                y=results["mean_return"],
                color="r",
                linestyle="--",
                label=f"Mean: {results['mean_return']:.2f}",
            )
            plt.xlabel("Episode")
            plt.ylabel("Return")
            plt.title("Episode Returns")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 3, 2)
            plt.plot(results["episode_lengths"], "g-", alpha=0.7)
            plt.axhline(
                y=results["mean_length"],
                color="r",
                linestyle="--",
                label=f"Mean: {results['mean_length']:.1f}",
            )
            plt.xlabel("Episode")
            plt.ylabel("Length")
            plt.title("Episode Lengths")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 3, 3)
            success_cumulative = np.cumsum(results["success_rates"]) / np.arange(
                1, len(results["success_rates"]) + 1
            )
            plt.plot(success_cumulative, "m-", alpha=0.7)
            plt.axhline(
                y=results["success_rate"],
                color="r",
                linestyle="--",
                label=f"Final: {results['success_rate']:.2%}",
            )
            plt.xlabel("Episode")
            plt.ylabel("Cumulative Success Rate")
            plt.title("Success Rate")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(
                output_dir / "evaluation_plots.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

            # Histogram of returns
            plt.figure(figsize=(8, 6))
            plt.hist(
                results["episode_returns"],
                bins=min(20, len(results["episode_returns"])),
                alpha=0.7,
                edgecolor="black",
            )
            plt.axvline(
                x=results["mean_return"],
                color="r",
                linestyle="--",
                label=f"Mean: {results['mean_return']:.2f}",
            )
            plt.xlabel("Episode Return")
            plt.ylabel("Frequency")
            plt.title("Distribution of Episode Returns")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(
                output_dir / "return_distribution.png", dpi=150, bbox_inches="tight"
            )
            plt.close()

        except Exception as e:
            print(f"Failed to save plots: {e}")


def evaluate_policy(
    model_path: str,
    env_id: str,
    num_episodes: int = 10,
    device: str = "auto",
    output_dir: str = None,
    render: bool = False,
    save_videos: bool = False,
    save_plots: bool = True,
    log_dir: str = "./logs",
    max_episode_steps: int = 1000,
) -> Dict[str, Any]:
    """Convenience function to evaluate a policy.

    Args:
        model_path: Path to the trained model
        env_id: Environment ID to evaluate on
        num_episodes: Number of episodes to run
        device: Device to use for evaluation
        output_dir: Directory to save results
        render: Whether to render during evaluation
        save_videos: Whether to save videos
        save_plots: Whether to save plots
        log_dir: Directory for logs
        max_episode_steps: Maximum steps per episode

    Returns:
        Evaluation results dictionary
    """
    evaluator = PolicyEvaluator(model_path, device=device, log_dir=log_dir)
    return evaluator.evaluate_policy(
        env_id=env_id,
        num_episodes=num_episodes,
        render=render,
        save_videos=save_videos,
        save_plots=save_plots,
        output_dir=output_dir,
        max_episode_steps=max_episode_steps,
    )
