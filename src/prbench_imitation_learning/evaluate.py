"""Evaluation functionality for trained diffusion policies."""

import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import prbench
import torch
from matplotlib import animation

from .policy import BehaviorCloningPolicy, DiffusionPolicy

# LeRobot imports
try:
    # isort: off
    from lerobot.policies.diffusion.modeling_diffusion import (
        DiffusionPolicy as LeRobotDiffusionPolicy,
    )

    # isort: on

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False


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
        self.image_history = deque(maxlen=self.config["obs_horizon"])

    def _load_model(self) -> Tuple[DiffusionPolicy, Dict[str, Any]]:
        """Load the trained model from checkpoint."""
        print(f"Loading model from {self.model_path}")

        # Use weights_only=False for LeRobot models to handle custom classes
        try:
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=True
            )
        except Exception:
            # If weights_only=True fails, try with weights_only=False for LeRobot models
            checkpoint = torch.load(
                self.model_path, map_location=self.device, weights_only=False
            )
        config = checkpoint["config"]

        # Check policy type
        policy_type = config.get("policy_type", "custom")

        if policy_type == "behavior_cloning":
            # Create behavior cloning model
            model = BehaviorCloningPolicy(
                obs_dim=config["obs_dim"],
                action_dim=config["action_dim"],
                obs_horizon=config["obs_horizon"],
                action_horizon=config["action_horizon"],
                hidden_dim=config.get("hidden_dim", 1024),
                num_layers=config.get("num_layers", 5),
            )

            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            print(
                f"Behavior Cloning model loaded successfully "
                f"(epoch {checkpoint['epoch']}, "
                f"loss: {checkpoint['loss']:.6f})"
            )
        elif policy_type == "lerobot":
            if not LEROBOT_AVAILABLE:
                raise ImportError(
                    "LeRobot is not available but model was trained with "
                    "LeRobot policy. Please install it with: pip install lerobot"
                )

            # Load LeRobot diffusion config and dataset stats
            diffusion_config = checkpoint["diffusion_config"]
            # Create LeRobot model (stats are handled by preprocessors)
            model = LeRobotDiffusionPolicy(diffusion_config)

            # Load state dict
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()

            # Workaround: in some LeRobot versions, the FiLM cond encoder
            # receives a longer vector.
            # Trim linear inputs to expected in_features to avoid matmul
            # shape errors at inference.

            def _trim_input_pre_hook(module, inputs):
                x = inputs[0]
                if hasattr(module, "in_features") and x.shape[-1] != module.in_features:
                    return (x[..., : module.in_features],)
                return inputs

            # pylint: disable=import-outside-toplevel
            from torch import nn

            for m in model.modules():
                if isinstance(m, nn.Linear):
                    m.register_forward_pre_hook(_trim_input_pre_hook)

            # Handle both step-based and epoch-based checkpoints
            if 'step' in checkpoint:
                progress_info = f"step {checkpoint['step']}"
            elif 'epoch' in checkpoint:
                progress_info = f"epoch {checkpoint['epoch']}"
            else:
                progress_info = "unknown progress"
            
            print(
                f"LeRobot model loaded successfully "
                f"({progress_info}, "
                f"loss: {checkpoint['loss']:.6f})"
            )
        else:
            # Create custom model
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

            # Handle both step-based and epoch-based checkpoints
            if 'step' in checkpoint:
                progress_info = f"step {checkpoint['step']}"
            elif 'epoch' in checkpoint:
                progress_info = f"epoch {checkpoint['epoch']}"
            else:
                progress_info = "unknown progress"
            
            print(
                f"Custom model loaded successfully "
                f"({progress_info}, "
                f"loss: {checkpoint['loss']:.6f})"
            )

        return model, config  # type: ignore

    def predict_action(self, observation) -> np.ndarray:
        """Predict action for a single observation.

        Args:
            observation: Dictionary containing 'state' and optionally 'image',
                         or numpy array

        Returns:
            Predicted action array
        """
        # Handle different observation formats
        if isinstance(observation, dict):
            obs_state = observation["state"]
            obs_image = observation.get("image", None)
        else:
            # Assume observation is the state directly
            obs_state = observation
            obs_image = None

        # Handle dimension mismatch (e.g., PushT environment has 5D obs but dataset has 2D)
        if hasattr(self.model, 'obs_dim') and len(obs_state) > self.model.obs_dim:
            # Trim observation to match model's expected dimensions
            obs_state = obs_state[:self.model.obs_dim]
        elif self.config.get("policy_type") == "lerobot" and self.config.get("obs_dim") and len(obs_state) > self.config["obs_dim"]:
            # Also handle for LeRobot models using config
            obs_state = obs_state[:self.config["obs_dim"]]

        self.obs_history.append(obs_state)
        
        # Also append image to history if available
        if obs_image is not None:
            self.image_history.append(obs_image)

        # If we don't have enough history, pad with the current observation
        while len(self.obs_history) < self.config["obs_horizon"]:
            self.obs_history.append(obs_state)
        
        # Pad image history if needed
        if obs_image is not None:
            while len(self.image_history) < self.config["obs_horizon"]:
                self.image_history.append(obs_image)

        # Check policy type
        policy_type = self.config.get("policy_type", "custom")

        if policy_type == "behavior_cloning":
            # Behavior cloning expects flattened observation sequence
            obs_seq = np.stack(list(self.obs_history))
            obs_seq_tensor = (
                torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)
            )

            # Predict action sequence
            with torch.no_grad():
                action_seq = self.model(obs_seq_tensor)

            # Return first action
            predicted_action = action_seq[0, 0].cpu().numpy()
        elif policy_type == "lerobot":
            # LeRobot's select_action expects a SINGLE timestep WITHOUT time dimension
            # The queue system will handle the time dimension
            # Convert current observation to tensor [batch_size, state_dim]
            obs_state_tensor = (
                torch.from_numpy(obs_state).float().unsqueeze(0).to(self.device)
            )  # Shape: [1, state_dim]

            # Use actual robot state from observations (matches training setup)
            robot_state = obs_state_tensor  # Shape: [1, state_dim]
            
            # Empty environment state to match training setup
            env_state = torch.zeros(1, 0, device=self.device)

            batch = {
                "observation.state": robot_state,
                "observation.environment_state": env_state,
            }

            # Check if model expects images (was trained with images)
            if self.config.get("image_shape") is not None and obs_image is not None:
                # Model was trained with images, provide single timestep image
                C, H_expected, W_expected = self.config["image_shape"]
                
                if len(obs_image.shape) == 3:  # H, W, C format
                    img_tensor = torch.from_numpy(obs_image).float().permute(2, 0, 1) / 255.0
                else:  # Already C, H, W
                    img_tensor = torch.from_numpy(obs_image).float() / 255.0
                
                # Resize if needed to match expected dimensions
                if img_tensor.shape[1] != H_expected or img_tensor.shape[2] != W_expected:
                    import torchvision.transforms.functional as TF
                    img_tensor = TF.resize(img_tensor, (H_expected, W_expected), antialias=True)
                
                # Add only batch dimension: [1, C, H, W]
                image_tensor = img_tensor.unsqueeze(0).to(self.device)
                batch["observation.image"] = image_tensor

            # Predict action sequence using LeRobot policy
            with torch.no_grad():
                # Debug shapes to ensure conditioning dims match
                try:
                    cfg = self.model.diffusion.config  # type: ignore
                    robot_dim = (
                        cfg.robot_state_feature.shape[0]  # type: ignore
                        if cfg.robot_state_feature is not None  # type: ignore
                        else 0
                    )
                    env_dim = (
                        cfg.env_state_feature.shape[0]  # type: ignore
                        if cfg.env_state_feature is not None  # type: ignore
                        else 0
                    )
                    n_steps = cfg.n_obs_steps  # type: ignore
                    step_embed = cfg.diffusion_step_embed_dim  # type: ignore
                    
                    
                    # Compute expected cond dim
                    expected_global = (robot_dim + env_dim) * n_steps
                    expected_cond = step_embed + expected_global
                    # Compute candidate global cond from batch
                    gc = torch.cat(
                        [
                            batch["observation.state"],
                            batch.get(
                                "observation.environment_state",
                                torch.empty(1, n_steps, 0, device=self.device),
                            ),
                        ],
                        dim=-1,
                    )
                    gc_flat = gc.flatten(start_dim=1)
                    
                    # Ask model to build the actual global_cond and print shape
                    # pylint: disable=protected-access
                    true_gc = self.model.diffusion._prepare_global_conditioning(batch)  # type: ignore # pylint: disable=line-too-long
                    # And the final cond that will be used inside the first
                    # residual block
                    t_embed = self.model.diffusion.unet.diffusion_step_encoder(  # type: ignore # pylint: disable=line-too-long
                        torch.zeros(1, dtype=torch.long, device=self.device)
                    )
                    final_cond = torch.cat([t_embed, true_gc], dim=-1)
                except Exception:
                    pass
                # Use select_action method for inference
                # (not forward which is for training)
                predicted_action_tensor = self.model.select_action(batch)
                predicted_action = predicted_action_tensor.cpu().numpy()

            # Take first action from the sequence
            if predicted_action.ndim > 2:
                predicted_action = predicted_action[
                    0, 0
                ]  # [batch, time, action_dim] -> [action_dim]
            elif predicted_action.ndim > 1:
                predicted_action = predicted_action[
                    0
                ]  # [batch, action_dim] -> [action_dim]
        else:
            # Custom policy expects flattened observation sequence
            obs_seq = np.stack(list(self.obs_history))
            obs_seq_tensor = (
                torch.from_numpy(obs_seq).float().unsqueeze(0).to(self.device)
            )

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
        self.image_history.clear()

    def evaluate_policy(
        self,
        env_id: str,
        num_episodes: int = 10,
        render: bool = False,  # pylint: disable=unused-argument
        save_videos: bool = False,
        save_plots: bool = True,
        output_dir: str = None,
        max_episode_steps: int = 1000,
        set_random_seed: bool = False,
        seed: int = 123,
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
            set_random_seed: Whether to use specific random seeds for environment resets
            seed: Random seed to use when set_random_seed is True

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

            # Register gym_pusht if needed
            if "gym_pusht" in env_id or "pusht" in env_id.lower():
                try:
                    import gym_pusht  # pylint: disable=import-outside-toplevel,unused-import
                except ImportError:
                    print("Warning: gym_pusht not installed. Install with: pip install gym-pusht")

            # Enable rendering if saving videos or if model uses images
            needs_rendering = save_videos or (self.config.get("policy_type") == "lerobot" and self.config.get("image_shape") is not None)
            render_mode = "rgb_array" if needs_rendering else None
            env = gym.make(env_id, render_mode=render_mode)
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
            with open(eval_log_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

        log_message(f"Starting evaluation on {env_id}")
        log_message(f"Model: {self.model_path}")
        log_message(f"Config: {self.config}")

        for episode in range(num_episodes):
            log_message(f"Starting episode {episode+1}/{num_episodes}")
            self.reset()
            # Use specific seed if set_random_seed is True, otherwise use random seed
            reset_seed = seed if set_random_seed else np.random.randint(0, 1000000)
            obs, info = env.reset(seed=reset_seed)

            episode_return = 0.0
            episode_length = 0
            trajectory = []

            frames = [] if save_videos else None

            done = False
            while not done and episode_length < max_episode_steps:
                # Get image observation if the model needs it
                obs_dict = obs
                if self.config.get("policy_type") == "lerobot" and self.config.get("image_shape") is not None:
                    # Get image from environment render
                    try:
                        image = env.render()
                        if image is not None:
                            obs_dict = {"state": obs, "image": image}
                    except Exception:
                        obs_dict = obs
                
                # Get action from policy
                action = self.predict_action(obs_dict)

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
                        f"  Episode {episode+1}, Step {episode_length}, "
                        f"Reward: {episode_return:.2f}"
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

                if save_videos and frames is not None:
                    try:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    except Exception:  # pylint: disable=broad-except
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
            # If no explicit success flag, consider it successful if episode ended
            # before max steps
            if "success" in info:
                success = info["success"]
            else:
                # Success if episode terminated naturally (not truncated due to max
                # steps)
                success = episode_length < max_episode_steps and done
            success_rates.append(float(success))
            all_trajectories.append(trajectory)

            log_message(
                f"Episode {episode+1}/{num_episodes}: "
                f"Return={episode_return:.2f}, Length={episode_length}, "
                f"Success={success}"
            )

            # Save video if requested
            if save_videos and "frames" in locals() and frames:
                self._save_video(frames, output_dir / f"episode_{episode}.gif")

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
        with open(results_path, "w", encoding="utf-8") as f:
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
    set_random_seed: bool = False,
    seed: int = 123,
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
        set_random_seed: Whether to use specific random seeds for environment resets
        seed: Random seed to use when set_random_seed is True

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
        set_random_seed=set_random_seed,
        seed=seed,
    )
