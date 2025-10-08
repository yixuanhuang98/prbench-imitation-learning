#!/bin/bash
# Test training script for train_lerobot_direct.py

conda activate prbenchIL1

python train_lerobot_direct.py \
    --env.type=pusht \
    --policy.type=diffusion \
    --policy.repo_id=test_pusht_model \
    --dataset.repo_id=lerobot/pusht \
    --dataset.video_backend=pyav \
    --batch_size=64 \
    --steps=10000 \
    --eval_freq=2000 \
    --save_freq=2000 \
    --output_dir=../test_results/pusht_test_$(date +%Y%m%d_%H%M%S) \
    --wandb.enable=false

echo "Training completed!"
