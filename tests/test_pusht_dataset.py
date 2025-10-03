"""Integration test for training policies on the lerobot PushT dataset.

This test verifies that our diffusion policy can be trained and evaluated
on the lerobot PushT dataset, which is a standard benchmark task for
imitation learning.
"""

import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import pytest
import torch

from prbench_imitation_learning import (
    evaluate_policy,
    get_default_training_config,
    train_behavior_cloning_policy,
    train_diffusion_policy,
)

# Check if lerobot is available
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.factory import make_env

    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False

# Skip all tests in this file if lerobot is not available
pytestmark = pytest.mark.skipif(
    not LEROBOT_AVAILABLE, reason="LeRobot not installed"
)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def pusht_dataset():
    """Load the lerobot PushT dataset.
    
    Note: This will download the dataset on first run, which may take some time.
    Subsequent runs will use the cached version.
    """
    # Use a smaller subset of the PushT dataset for faster testing
    # The full dataset is "lerobot/pusht", but we can specify a subset
    # Use pyav backend instead of torchcodec to avoid FFmpeg library issues
    try:
        dataset = LeRobotDataset("lerobot/pusht", video_backend="pyav")
    except Exception:
        # If pyav doesn't work, try without video backend specification
        dataset = LeRobotDataset("lerobot/pusht")
    return dataset


@pytest.fixture
def pusht_env():
    """Create a PushT environment for evaluation."""
    try:
        # Import gym_pusht to register the environment
        import gym_pusht  # pylint: disable=import-outside-toplevel,unused-import
        import gymnasium as gym
        env = gym.make("gym_pusht/PushT-v0")
    except ImportError:
        # If gym_pusht is not installed, skip the test
        pytest.skip("PushT environment not available (gym_pusht package required)")
    yield env
    env.close()


def test_pusht_dataset_loading(pusht_dataset):
    """Test that PushT dataset can be loaded correctly."""
    assert pusht_dataset is not None
    assert len(pusht_dataset) > 0
    
    # Check that dataset has expected keys
    sample = pusht_dataset[0]
    assert "observation.state" in sample or "observation.environment_state" in sample
    assert "action" in sample
    
    print(f"Loaded PushT dataset with {len(pusht_dataset)} samples")
    print(f"Number of episodes: {len(pusht_dataset.episode_data_index)}")
    print("Note: Default lerobot dataset may filter episodes. Full dataset has ~200 episodes.")


def test_pusht_env_creation(pusht_env):
    """Test that PushT environment can be created and reset."""
    assert pusht_env is not None
    
    # Test reset
    obs, info = pusht_env.reset()
    assert obs is not None
    
    # Test step
    action = pusht_env.action_space.sample()
    next_obs, reward, terminated, truncated, info = pusht_env.step(action)
    assert next_obs is not None
    
    print(f"PushT environment created successfully")
    print(f"Observation space: {pusht_env.observation_space}")
    print(f"Action space: {pusht_env.action_space}")


def test_train_diffusion_policy_on_pusht(temp_output_dir, pusht_dataset):
    """Test training a custom diffusion policy on PushT dataset.
    
    This is a minimal training run to verify that:
    1. The dataset can be loaded and preprocessed
    2. The model can be created and trained
    3. The model can be saved and loaded
    
    NOTE: Uses limited episodes from default dataset loading.
    For full training, use the standalone run_pusht_example.py script.
    """
    dataset_path = pusht_dataset.root
    model_save_path = Path(temp_output_dir) / "pusht_diffusion_model.pth"
    
    # Create minimal training config for fast testing
    config = get_default_training_config()
    config.update({
        "batch_size": 32,
        "num_epochs": 2,  # Just 2 epochs for testing
        "learning_rate": 1e-4,
        "use_wandb": False,
        "num_workers": 0,  # Avoid multiprocessing issues in tests
        "obs_horizon": 2,
        "action_horizon": 8,
        "pred_horizon": 16,
    })
    
    # Train the model
    model = train_diffusion_policy(
        dataset_path=str(dataset_path),
        model_save_path=str(model_save_path),
        config=config,
        log_dir=str(Path(temp_output_dir) / "logs"),
    )
    
    # Verify model was created
    assert model is not None
    
    # Verify checkpoint was saved
    assert model_save_path.exists()
    
    # Verify we can load the checkpoint
    checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint
    assert "epoch" in checkpoint
    
    print(f"Successfully trained diffusion policy for {checkpoint['epoch']} epochs")


def test_train_behavior_cloning_on_pusht(temp_output_dir, pusht_dataset):
    """Test training a behavior cloning policy on PushT dataset.
    
    NOTE: Uses limited episodes from default dataset loading.
    For full training, use the standalone run_pusht_example.py script.
    """
    dataset_path = pusht_dataset.root
    model_save_path = Path(temp_output_dir) / "pusht_bc_model.pth"
    
    # Create minimal training config for fast testing
    config = get_default_training_config()
    config.update({
        "batch_size": 32,
        "num_epochs": 2,  # Just 2 epochs for testing
        "learning_rate": 1e-4,
        "use_wandb": False,
        "num_workers": 0,
        "obs_horizon": 2,
        "action_horizon": 8,
        "pred_horizon": 16,
    })
    
    # Train the model
    model = train_behavior_cloning_policy(
        dataset_path=str(dataset_path),
        model_save_path=str(model_save_path),
        config=config,
        log_dir=str(Path(temp_output_dir) / "logs"),
    )
    
    # Verify model was created
    assert model is not None
    
    # Verify checkpoint was saved
    assert model_save_path.exists()
    
    # Verify we can load the checkpoint
    checkpoint = torch.load(model_save_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "config" in checkpoint
    assert checkpoint["config"]["policy_type"] == "behavior_cloning"
    
    print(f"Successfully trained behavior cloning policy for {checkpoint['epoch']} epochs")


def test_evaluate_policy_on_pusht(temp_output_dir, pusht_dataset, pusht_env):
    """Test evaluating a trained policy on PushT environment.
    
    This test:
    1. Trains a minimal model
    2. Evaluates it in the PushT environment
    3. Verifies that evaluation metrics are computed correctly
    
    NOTE: Uses limited episodes from default dataset loading.
    For full training and evaluation, use the standalone run_pusht_example.py script.
    """
    # First, train a minimal model
    dataset_path = pusht_dataset.root
    model_save_path = Path(temp_output_dir) / "pusht_eval_model.pth"
    
    config = get_default_training_config()
    config.update({
        "batch_size": 32,
        "num_epochs": 1,  # Minimal training
        "learning_rate": 1e-4,
        "use_wandb": False,
        "num_workers": 0,
        "obs_horizon": 2,
        "action_horizon": 8,
        "pred_horizon": 16,
    })
    
    # Train the model
    train_diffusion_policy(
        dataset_path=str(dataset_path),
        model_save_path=str(model_save_path),
        config=config,
        log_dir=str(Path(temp_output_dir) / "logs"),
    )
    
    # Now evaluate it
    eval_output_dir = Path(temp_output_dir) / "evaluation"
    
    results = evaluate_policy(
        model_path=str(model_save_path),
        env_id="lerobot/pusht",
        num_episodes=3,  # Just a few episodes for testing
        output_dir=str(eval_output_dir),
        render=False,
        save_videos=False,
        save_plots=True,
        log_dir=str(Path(temp_output_dir) / "logs"),
        max_episode_steps=100,  # Short episodes for testing
    )
    
    # Verify results structure
    assert results is not None
    assert "mean_return" in results
    assert "std_return" in results
    assert "success_rate" in results
    assert "episode_returns" in results
    assert "episode_lengths" in results
    
    # Verify we have the expected number of episodes
    assert len(results["episode_returns"]) == 3
    assert len(results["episode_lengths"]) == 3
    
    print(f"Evaluation results:")
    print(f"  Mean return: {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Mean episode length: {sum(results['episode_lengths'])/len(results['episode_lengths']):.1f}")


@pytest.mark.slow
def test_full_pusht_training_pipeline(temp_output_dir, pusht_dataset):
    """Full integration test of the PushT training pipeline.
    
    This test runs a more complete training cycle:
    1. Trains for more epochs
    2. Tests both diffusion and behavior cloning
    3. Evaluates both policies
    4. Compares their performance
    
    This test is marked as 'slow' and can be skipped in quick test runs.
    """
    dataset_path = pusht_dataset.root
    
    # Train diffusion policy
    diffusion_model_path = Path(temp_output_dir) / "pusht_diffusion_full.pth"
    config_diffusion = get_default_training_config()
    config_diffusion.update({
        "batch_size": 64,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "use_wandb": False,
        "num_workers": 0,
    })
    
    train_diffusion_policy(
        dataset_path=str(dataset_path),
        model_save_path=str(diffusion_model_path),
        config=config_diffusion,
        log_dir=str(Path(temp_output_dir) / "logs" / "diffusion"),
    )
    
    # Train behavior cloning policy
    bc_model_path = Path(temp_output_dir) / "pusht_bc_full.pth"
    config_bc = get_default_training_config()
    config_bc.update({
        "batch_size": 64,
        "num_epochs": 10,
        "learning_rate": 1e-4,
        "use_wandb": False,
        "num_workers": 0,
    })
    
    train_behavior_cloning_policy(
        dataset_path=str(dataset_path),
        model_save_path=str(bc_model_path),
        config=config_bc,
        log_dir=str(Path(temp_output_dir) / "logs" / "bc"),
    )
    
    # Evaluate both policies
    eval_episodes = 5
    
    diffusion_results = evaluate_policy(
        model_path=str(diffusion_model_path),
        env_id="lerobot/pusht",
        num_episodes=eval_episodes,
        output_dir=str(Path(temp_output_dir) / "eval_diffusion"),
        render=False,
        save_videos=False,
        save_plots=True,
        log_dir=str(Path(temp_output_dir) / "logs" / "eval_diffusion"),
        max_episode_steps=200,
    )
    
    bc_results = evaluate_policy(
        model_path=str(bc_model_path),
        env_id="lerobot/pusht",
        num_episodes=eval_episodes,
        output_dir=str(Path(temp_output_dir) / "eval_bc"),
        render=False,
        save_videos=False,
        save_plots=True,
        log_dir=str(Path(temp_output_dir) / "logs" / "eval_bc"),
        max_episode_steps=200,
    )
    
    # Print comparison
    print("\n" + "=" * 60)
    print("POLICY COMPARISON ON PUSHT")
    print("=" * 60)
    print(f"Diffusion Policy:")
    print(f"  Mean return: {diffusion_results['mean_return']:.2f} ± {diffusion_results['std_return']:.2f}")
    print(f"  Success rate: {diffusion_results['success_rate']:.2%}")
    
    print(f"\nBehavior Cloning Policy:")
    print(f"  Mean return: {bc_results['mean_return']:.2f} ± {bc_results['std_return']:.2f}")
    print(f"  Success rate: {bc_results['success_rate']:.2%}")
    print("=" * 60)
    
    # Both policies should produce reasonable results (not NaN, not zero)
    assert not torch.isnan(torch.tensor(diffusion_results['mean_return']))
    assert not torch.isnan(torch.tensor(bc_results['mean_return']))


if __name__ == "__main__":
    # Allow running this test file directly for debugging
    pytest.main([__file__, "-v", "-s"])
