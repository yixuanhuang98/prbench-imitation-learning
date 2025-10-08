#!/usr/bin/env python
"""
Upload LeRobot dataset to HuggingFace Hub.

Usage:
    # First login to HuggingFace:
    huggingface-cli login
    
    # Then upload:
    python scripts/upload_dataset_to_hub.py \
        --dataset_path=datasets/motion2d_lerobot \
        --repo_id=yixuanh/motion2d_expert \
        --private
"""

import argparse
import shutil
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo


def upload_dataset_to_hub(
    dataset_path: str,
    repo_id: str,
    private: bool = True,
):
    """Upload a LeRobot dataset to HuggingFace Hub."""
    
    dataset_path = Path(dataset_path)
    
    print("=" * 80)
    print("UPLOADING DATASET TO HUGGINGFACE HUB")
    print("=" * 80)
    print(f"\nDataset path: {dataset_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    
    # Load the dataset
    print("\nLoading dataset...")
    data_path = dataset_path / "data" / "chunk-000"
    dataset = load_from_disk(str(data_path))
    
    print(f"Dataset loaded: {dataset}")
    print(f"Features: {list(dataset['train'].features.keys())}")
    print(f"Number of samples: {len(dataset['train'])}")
    
    # Create repository
    print(f"\nCreating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository created/verified: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error creating repo: {e}")
        return
    
    # Upload the dataset
    print("\nUploading dataset to Hub...")
    dataset.push_to_hub(
        repo_id=repo_id,
        private=private,
    )
    print("✓ Dataset uploaded successfully!")
    
    # Upload metadata files
    print("\nUploading metadata files...")
    api = HfApi()
    
    # Upload info.json
    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        api.upload_file(
            path_or_fileobj=str(info_path),
            path_in_repo="meta/info.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✓ Uploaded meta/info.json")
    
    # Upload episodes.parquet
    episodes_path = dataset_path / "meta" / "episodes" / "episodes.parquet"
    if episodes_path.exists():
        api.upload_file(
            path_or_fileobj=str(episodes_path),
            path_in_repo="meta/episodes/episodes.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✓ Uploaded meta/episodes/episodes.parquet")
    
    # Upload tasks.parquet
    tasks_path = dataset_path / "meta" / "tasks" / "tasks.parquet"
    if tasks_path.exists():
        api.upload_file(
            path_or_fileobj=str(tasks_path),
            path_in_repo="meta/tasks/tasks.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✓ Uploaded meta/tasks/tasks.parquet")
    
    # Upload expert metadata if exists
    expert_metadata_path = dataset_path / "expert_metadata.json"
    if expert_metadata_path.exists():
        api.upload_file(
            path_or_fileobj=str(expert_metadata_path),
            path_in_repo="expert_metadata.json",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("✓ Uploaded expert_metadata.json")
    
    print("\n" + "=" * 80)
    print("✅ UPLOAD COMPLETE!")
    print("=" * 80)
    print(f"\nYour dataset is now available at:")
    print(f"  https://huggingface.co/datasets/{repo_id}")
    print(f"\nTo train with this dataset, use:")
    print(f"""
python scripts/train_lerobot_direct.py \\
    --dataset.repo_id={repo_id} \\
    --policy.type=diffusion \\
    --policy.repo_id=yixuanh/motion2d_policy \\
    --output_dir=outputs/expert_training \\
    --steps=50000 \\
    --eval_freq=10000 \\
    --save_freq=10000 \\
    --policy.device=cuda \\
    --policy.push_to_hub=false
    """)


def main():
    parser = argparse.ArgumentParser(
        description="Upload LeRobot dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (username/dataset_name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private"
    )
    
    args = parser.parse_args()
    
    upload_dataset_to_hub(
        dataset_path=args.dataset_path,
        repo_id=args.repo_id,
        private=args.private,
    )


if __name__ == "__main__":
    main()

