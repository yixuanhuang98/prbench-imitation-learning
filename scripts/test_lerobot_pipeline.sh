#!/bin/bash
# Test script for LeRobot diffusion policy pipeline
# This tests the exact command the user wants to run

set -e  # Exit on error

echo "Activating prbenchIL1 conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate prbenchIL1

echo "Running LeRobot diffusion policy pipeline..."
cd /home/yixuan/prbench_dir/prbench-imitation-learning/scripts

python run_diffusion_pipeline.py \
    --env motion2d-p0 \
    --policy-type lerobot \
    --data-type expert \
    --data-episodes 1 \
    --train-epochs 10000 \
    --save-demo-videos \
    --eval-episodes 5 \
    --save-videos \
    --set-random-seed

echo "Pipeline completed successfully!"

