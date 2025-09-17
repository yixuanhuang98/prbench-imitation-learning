# Expert Demonstration Pipeline for Motion2D

This directory contains scripts for collecting expert demonstrations from Motion2D environments using BilevelPlanningAgent and training diffusion policies on them.

## Files

- `collect_motion2d_demonstrations.py`: Collects expert demonstrations using BilevelPlanningAgent
- `run_diffusion_pipeline.py`: Complete pipeline for data generation, training, and evaluation (now supports expert data)
- `run_expert_pipeline.sh`: Wrapper script that handles conda environment switching automatically

## Prerequisites

Make sure you have both required conda environments set up:

1. `pr_planning`: For bilevel planning and expert demonstration collection  
2. `prbenchIL1`: For imitation learning training and evaluation

**Note**: Currently, expert demonstration collection requires the `pr_planning` environment due to bilevel planning dependencies, while training requires the `prbenchIL1` environment.

## Usage

### Option 1: Automatic Environment Switching (Recommended)

Use the wrapper script that automatically handles environment switching:

```bash
# Basic expert pipeline for Motion2D with 2 passages
./run_expert_pipeline.sh --env motion2d-p2 --data-type expert --data-episodes 20

# With custom parameters
./run_expert_pipeline.sh \
  --env motion2d-p2 \
  --data-type expert \
  --data-episodes 50 \
  --num-passages 2 \
  --max-abstract-plans 15 \
  --samples-per-step 5 \
  --planning-timeout 60.0 \
  --train-epochs 100 \
  --save-demo-videos
```

### Option 2: Manual Environment Management

If you prefer to manage environments manually:

```bash
# Step 1: Collect expert demonstrations (requires pr_planning environment)
conda activate pr_planning
python run_diffusion_pipeline.py \
  --env motion2d-p2 \
  --data-type expert \
  --data-episodes 20 \
  --num-passages 2 \
  --skip-training \
  --skip-evaluation

# Step 2: Train and evaluate (requires prbenchIL1 environment)  
conda activate prbenchIL1
python run_diffusion_pipeline.py \
  --env motion2d-p2 \
  --data-type expert \
  --data-episodes 20 \
  --skip-data \
  --dataset-path ./diffusion_pipeline_results/motion2d-p2_expert_20ep_*/datasets/motion2d-p2_expert_20ep
```

### Option 3: Standalone Expert Collection

Just collect demonstrations without training:

```bash
conda activate pr_planning
python collect_motion2d_demonstrations.py \
  --num_episodes 50 \
  --num_passages 2 \
  --output_dir ./my_expert_data \
  --save_videos
```

## Parameters for Expert Data Collection

### Environment Parameters
- `--num-passages`: Number of passages in Motion2D (1, 2, or 3)
- `--data-episodes`: Number of episodes to collect

### BilevelPlanningAgent Parameters
- `--max-abstract-plans`: Maximum number of abstract plans to try (default: 10)
- `--samples-per-step`: Number of samples per planning step (default: 3) 
- `--planning-timeout`: Timeout for motion planning in seconds (default: 30.0)

### Other Options
- `--save-demo-videos`: Save videos of demonstration trajectories
- `--output-dir`: Output directory for results
- `--experiment-name`: Custom experiment name

## Example Workflows

### Quick Test
```bash
./run_expert_pipeline.sh --env motion2d-p2 --data-type expert --data-episodes 5 --train-epochs 10
```

### Full Training Run
```bash
./run_expert_pipeline.sh \
  --env motion2d-p2 \
  --data-type expert \
  --data-episodes 100 \
  --num-passages 2 \
  --train-epochs 200 \
  --batch-size 64 \
  --learning-rate 1e-4 \
  --eval-episodes 20 \
  --save-demo-videos \
  --save-videos \
  --use-wandb
```

### Compare Expert vs Random
```bash
# Collect expert data
./run_expert_pipeline.sh --env motion2d-p2 --data-type expert --data-episodes 50 --experiment-name expert_baseline

# Collect random data  
conda activate prbenchIL1
python run_diffusion_pipeline.py --env motion2d-p2 --data-type random --data-episodes 50 --experiment-name random_baseline
```

## Output Structure

```
diffusion_pipeline_results/
└── {experiment_name}/
    ├── datasets/
    │   └── {dataset_name}/
    │       ├── dataset.pkl
    │       ├── metadata.json
    │       └── videos/ (if --save-demo-videos)
    ├── models/
    │   └── {experiment_name}_model.pth
    ├── evaluation/
    │   ├── evaluation_results.json
    │   ├── plots/
    │   └── videos/ (if --save-videos)
    ├── experiment_config.json
    └── pipeline_summary.json
```

## Troubleshooting

### Environment Issues
- Make sure both `pr_planning` and `prbenchIL1` environments are properly set up
- The wrapper script will check for environment existence before running

### Import Errors
- Ensure `collect_motion2d_demonstrations.py` is in the same directory as `run_diffusion_pipeline.py`
- Check that all required packages are installed in the respective environments

### Planning Failures
- Increase `--planning-timeout` if the agent times out frequently
- Try reducing `--max-abstract-plans` or `--samples-per-step` for faster but potentially lower quality planning
- Motion2D with more passages is more challenging - start with `--num-passages 1` for testing

### Memory Issues
- Reduce `--batch-size` if you run out of GPU memory during training
- Reduce `--data-episodes` for initial testing
