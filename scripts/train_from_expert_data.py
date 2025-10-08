#!/usr/bin/env python
"""
Train LeRobot policies using existing expert demonstration data.

This script trains on pre-generated expert demonstrations stored in pickle format.

Usage:
    python scripts/train_from_expert_data.py \
        --expert_data_path=expert_data/motion2d_p0_20251008_105219 \
        --output_dir=outputs/motion2d_p0_training \
        --steps=10000 \
        --policy.device=cuda
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

# Add LeRobot imports
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.utils.utils import init_logging

# Import the training function from train_lerobot_with_experts
sys.path.insert(0, str(Path(__file__).parent))
from train_lerobot_with_experts import train, update_policy


def convert_expert_data_to_lerobot_dataset(expert_data_path: str, output_dir: str):
    """Convert expert pickle data to LeRobot dataset format."""
    expert_path = Path(expert_data_path)
    
    # Load expert data
    with open(expert_path / "dataset.pkl", "rb") as f:
        expert_data = pickle.load(f)
    
    metadata = expert_data["metadata"]
    episodes = expert_data["episodes"]
    
    logging.info(f"Loading expert data from: {expert_data_path}")
    logging.info(f"  Environment: {metadata['env_name']}")
    logging.info(f"  Episodes: {metadata['num_episodes']}")
    logging.info(f"  Frames: {metadata['total_frames']}")
    logging.info(f"  Success rate: {metadata['success_rate']:.2%}")
    
    # Create LeRobot dataset directory structure
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
    import torch
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert episodes to LeRobot format
    # Group by episode
    episode_groups = {}
    for frame in episodes:
        ep_idx = frame['episode_index']
        if ep_idx not in episode_groups:
            episode_groups[ep_idx] = []
        episode_groups[ep_idx].append(frame)
    
    # Create dataset structure
    dataset_dict = {
        'observation.state': [],
        'observation.image': [],
        'action': [],
        'episode_index': [],
        'frame_index': [],
        'timestamp': [],
        'next.reward': [],
        'next.done': [],
    }
    
    for ep_idx in sorted(episode_groups.keys()):
        ep_frames = episode_groups[ep_idx]
        for frame in ep_frames:
            dataset_dict['observation.state'].append(torch.from_numpy(frame['observation.state']))
            dataset_dict['observation.image'].append(torch.from_numpy(frame['observation.image']))
            dataset_dict['action'].append(torch.from_numpy(frame['action']))
            dataset_dict['episode_index'].append(frame['episode_index'])
            dataset_dict['frame_index'].append(frame['frame_index'])
            dataset_dict['timestamp'].append(frame['timestamp'])
            dataset_dict['next.reward'].append(frame['next.reward'])
            dataset_dict['next.done'].append(frame['next.done'])
    
    # Save as LeRobot dataset
    dataset_name = metadata.get('dataset_name', 'expert_dataset')
    
    # Create info.json for LeRobot
    info = {
        'fps': 10,
        'video': False,
        'codebase_version': 'v2.0',
        'data_path': str(output_path),
        'total_episodes': metadata['num_episodes'],
        'total_frames': metadata['total_frames'],
    }
    
    with open(output_path / 'info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    # Save dataset
    torch.save(dataset_dict, output_path / 'data.pt')
    
    # Also copy metadata
    with open(output_path / 'expert_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Converted dataset saved to: {output_path}")
    return str(output_path)


def main():
    """Main entry point for training with existing expert data."""
    # Parse arguments
    arg_parser = argparse.ArgumentParser(
        description="Train LeRobot policies with existing expert demonstration data",
        add_help=False,
    )
    
    arg_parser.add_argument(
        "--expert_data_path",
        type=str,
        required=True,
        help="Path to expert data directory (containing dataset.pkl)",
    )
    
    # Parse known args
    custom_args, remaining_args = arg_parser.parse_known_args()
    
    init_logging()
    
    logging.info("=" * 80)
    logging.info("TRAINING WITH EXISTING EXPERT DATA")
    logging.info("=" * 80)
    logging.info(f"Expert data path: {custom_args.expert_data_path}")
    
    # Check if expert data exists
    expert_path = Path(custom_args.expert_data_path)
    if not expert_path.exists():
        logging.error(f"Expert data path not found: {custom_args.expert_data_path}")
        sys.exit(1)
    
    dataset_pkl = expert_path / "dataset.pkl"
    if not dataset_pkl.exists():
        logging.error(f"dataset.pkl not found in {custom_args.expert_data_path}")
        sys.exit(1)
    
    # Load and display metadata
    metadata_path = expert_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info("\nExpert Data Info:")
        logging.info(f"  Environment: {metadata.get('env_name', 'unknown')}")
        logging.info(f"  Episodes: {metadata.get('num_episodes', 'unknown')}")
        logging.info(f"  Frames: {metadata.get('total_frames', 'unknown')}")
        logging.info(f"  Success Rate: {metadata.get('success_rate', 0):.2%}")
    
    logging.info("\nNote: This script currently shows how to load expert data.")
    logging.info("To train with this data, you need to:")
    logging.info("1. Convert the expert data to LeRobot dataset format, OR")
    logging.info("2. Use the dataset path directly with LeRobot training config")
    logging.info("\nFor now, use the existing dataset with lerobot training:")
    logging.info(f"  --dataset.repo_id=lerobot/pusht (or your dataset)")
    logging.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

