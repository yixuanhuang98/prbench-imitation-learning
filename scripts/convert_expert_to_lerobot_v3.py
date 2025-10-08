#!/usr/bin/env python
"""
Convert expert pickle data to a LeRobot v3.0 dataset (file-based Parquet) using
LeRobot's dataset API, mirroring PushT's structure so it works with
train_lerobot_direct.py locally (no Hub required).

This script will:
- Create `meta/info.json` with features + defaults
- Write `data/chunk-000/file-000.parquet` with frames (images embedded)
- Write `meta/tasks.parquet` (index = task name, column = task_index)
- Write `meta/episodes/chunk-000/file-000.parquet` with episode ranges and data file refs
- Write `meta/stats.json`

Usage:
  python scripts/convert_expert_to_lerobot_v3.py \
      --expert_data_dir expert_data/motion2d_p0_20251008_105219 \
      --output_dir datasets/motion2d_lerobot_v3 \
      --repo_id motion2d_expert \
      --fps 10
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image as PILImage

# Import LeRobot APIs
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts, hw_to_dataset_features


def load_expert_pickle(expert_data_dir: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    import pickle

    pkl_path = expert_data_dir / "dataset.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Expected keys based on prior usage
    metadata = data.get("metadata", {})
    episodes_or_frames = data.get("episodes")
    if episodes_or_frames is None:
        # fallback
        episodes_or_frames = data.get("frames")
    if episodes_or_frames is None:
        raise ValueError("dataset.pkl missing 'episodes' or 'frames' key")

    return metadata, episodes_or_frames


def to_pil(img: np.ndarray) -> PILImage:
    if isinstance(img, PILImage):
        return img
    if img.dtype != np.uint8:
        # clip + convert
        arr = np.clip(img, 0, 255).astype(np.uint8)
    else:
        arr = img
    return PILImage.fromarray(arr)


def infer_shapes(frames: List[Dict[str, Any]]) -> Tuple[int, int, Tuple[int, int, int]]:
    # Assume frames contain np arrays
    for fr in frames:
        if (
            "observation.state" in fr
            and "action" in fr
            and "observation.image" in fr
        ):
            state_dim = int(np.array(fr["observation.state"]).shape[0])
            action_dim = int(np.array(fr["action"]).shape[0])
            img_shape = tuple(np.array(fr["observation.image"]).shape)
            return state_dim, action_dim, img_shape  # (H, W, C)
    raise ValueError("Could not infer shapes from frames; expected keys missing.")


def build_features(state_dim: int, action_dim: int, img_shape: Tuple[int, int, int]) -> Dict[str, Dict]:
    # Build observation features (state + image), using images (not videos) for simplicity
    obs_hw = {f"s{i}": float for i in range(state_dim)}
    # Add a single camera
    obs_hw.update({"cam0": img_shape})
    obs_feats = hw_to_dataset_features(obs_hw, prefix="observation", use_video=False)

    # Build action features
    act_hw = {f"a{i}": float for i in range(action_dim)}
    act_feats = hw_to_dataset_features(act_hw, prefix="action")

    features = combine_feature_dicts(obs_feats, act_feats)
    return features


def group_by_episode(frames: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    buckets: Dict[int, List[Dict[str, Any]]] = {}
    for fr in frames:
        ep_idx = int(fr.get("episode_index", 0))
        buckets.setdefault(ep_idx, []).append(fr)
    # sort frames within episode by frame_index if present
    for ep_idx in buckets:
        buckets[ep_idx].sort(key=lambda x: int(x.get("frame_index", 0)))
    return dict(sorted(buckets.items(), key=lambda kv: kv[0]))


def convert(
    expert_data_dir: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
) -> None:
    metadata, frames = load_expert_pickle(expert_data_dir)

    # Infer shapes
    state_dim, action_dim, img_shape = infer_shapes(frames)

    # Build features dict (image-based)
    features = build_features(state_dim, action_dim, img_shape)

    # Create dataset structure using LeRobot API (ensures perfect v3.0 compliance)
    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=metadata.get("env_name", metadata.get("env_type", "geom2d")),
        use_videos=False,
    )

    # Map frames by episode
    episodes = group_by_episode(frames)

    # Derive a task name
    env_name = str(metadata.get("env_name", "motion2d")).lower()
    task_name = env_name.replace("/", "_")
    if not task_name:
        task_name = "geom2d_task"

    # Write episodes
    total_frames = 0
    for ep_idx, ep_frames in episodes.items():
        # For each frame in the episode, add to buffer
        for i, fr in enumerate(ep_frames):
            obs_state = np.array(fr["observation.state"], dtype=np.float32)
            action = np.array(fr["action"], dtype=np.float32)
            image = fr["observation.image"]
            # image can be np array (H,W,C) uint8; pass PIL or numpy
            if isinstance(image, np.ndarray):
                img_val = image  # LeRobot accepts np ndarray; will be embedded later
            else:
                img_val = image

            frame = {
                # special field required (not in features)
                "task": task_name,
                # features
                "observation.state": obs_state,
                "observation.images.cam0": img_val,
                "action": action,
                # Do NOT include 'timestamp' here; LeRobot will infer it automatically
            }
            ds.add_frame(frame)

        # save episode (writes data parquet, updates meta, tasks, stats, episodes)
        ds.save_episode()
        total_frames += len(ep_frames)

    # Write a minimal README on the Hub card structure (optional locally)
    # Not needed for local training.

    print("\nConversion complete!")
    print(f"Output root: {output_dir}")
    print("Structure:")
    print(f"  - {output_dir}/meta/info.json")
    print(f"  - {output_dir}/meta/tasks.parquet")
    print(f"  - {output_dir}/meta/episodes/chunk-000/file-000.parquet (and possibly more)")
    print(f"  - {output_dir}/data/chunk-000/file-000.parquet (and possibly more)")


def main():
    parser = argparse.ArgumentParser(description="Convert expert pickle to LeRobot v3.0 file-based dataset")
    parser.add_argument("--expert_data_dir", type=str, required=True, help="Directory containing dataset.pkl")
    parser.add_argument("--output_dir", type=str, required=True, help="Output dataset root directory")
    parser.add_argument("--repo_id", type=str, default="motion2d_expert", help="Local dataset repo_id")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for timestamps")
    args = parser.parse_args()

    expert_dir = Path(args.expert_data_dir)
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        # Avoid accidental overwrite of existing datasets
        raise FileExistsError(f"Output directory already exists: {out_dir}")

    convert(expert_dir, out_dir, repo_id=args.repo_id, fps=args.fps)

    print("\nTo train locally with this dataset, run:")
    print(
        " ".join(
            [
                "python scripts/train_lerobot_direct.py",
                f"--dataset.repo_id={args.repo_id}",
                f"--dataset.root={args.output_dir}",
                "--policy.type=diffusion",
                "--policy.repo_id=yixuanh/motion2d_policy",
                "--output_dir=outputs/expert_training",
                "--steps=50000",
                "--eval_freq=10000",
                "--save_freq=10000",
                "--policy.device=cuda",
                "--policy.push_to_hub=false",
            ]
        )
    )


if __name__ == "__main__":
    main()


