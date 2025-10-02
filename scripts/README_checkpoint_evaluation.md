# Checkpoint Evaluation Script

## Overview

The `run_checkpoint_evaluation.py` script trains policies (behavior cloning, custom diffusion, or LeRobot diffusion) while saving multiple checkpoints during training, then evaluates all checkpoints on a fixed set of test episodes to analyze performance evolution during training. It supports multiple data types (random, expert, precomputed) and policy architectures.

## Features

- **Multi-checkpoint Training**: Saves model checkpoints at regular intervals during training
- **Multiple Policy Types**: Supports behavior cloning, custom diffusion, and LeRobot diffusion policies
- **Multiple Data Types**: Supports random, expert (using BilevelPlanningAgent), and precomputed demonstrations
- **Reproducible Evaluation**: Evaluates all checkpoints on the same set of episodes using fixed random seeds
- **Performance Analysis**: Creates visualizations showing how performance evolves during training
- **Comprehensive Logging**: Detailed logs for all phases of the experiment
- **Structured Output**: Organized directory structure with results, plots, and checkpoints

## Quick Start

```bash
# Activate the conda environment
conda activate prbenchIL1

# Run with default settings (10 episodes of training data, 20 training epochs, checkpoints every 5 epochs)
python scripts/run_checkpoint_evaluation.py

# Run with custom parameters - Behavior Cloning with expert data
python scripts/run_checkpoint_evaluation.py \
    --env motion2d-p1 \
    --data-episodes 20 \
    --train-epochs 30 \
    --checkpoint-interval 5 \
    --eval-episodes 10 \
    --data-type expert \
    --policy-type behavior_cloning

# Run with Custom Diffusion Policy
python scripts/run_checkpoint_evaluation.py \
    --env motion2d-p1 \
    --data-episodes 20 \
    --train-epochs 50 \
    --checkpoint-interval 10 \
    --eval-episodes 10 \
    --data-type random \
    --policy-type custom

# Run with LeRobot Diffusion Policy
python scripts/run_checkpoint_evaluation.py \
    --env motion2d-p1 \
    --data-episodes 20 \
    --train-epochs 50 \
    --checkpoint-interval 10 \
    --eval-episodes 10 \
    --data-type expert \
    --policy-type lerobot
```

## Key Parameters

### Core Parameters
- `--env`: Environment to use (e.g., motion2d-p1, stickbutton2d-b2)
- `--data-episodes`: Number of episodes for training data collection
- `--train-epochs`: Total number of training epochs
- `--checkpoint-interval`: Save checkpoint every N epochs
- `--eval-episodes`: Number of episodes to evaluate each checkpoint on
- `--seed`: Random seed for reproducible evaluation

### Data Type Options
- `--data-type`: Type of training data
  - `random`: Random agent demonstrations
  - `expert`: Expert demonstrations using BilevelPlanningAgent
  - `precomputed`: Load existing demonstration files
- `--precomputed-demos-dir`: Directory with precomputed demos (when using `--data-type precomputed`)

### Expert Data Parameters (when using `--data-type expert`)
- `--env-param`: Environment parameter (passages, buttons, obstructions, etc.)
- `--max-abstract-plans`: Maximum abstract plans for BilevelPlanningAgent
- `--samples-per-step`: Samples per planning step
- `--planning-timeout`: Planning timeout in seconds
- `--set-random-seed`: Use specific random seeds for reproducibility

### Policy Type Options
- `--policy-type`: Type of policy to train
  - `behavior_cloning`: Standard behavior cloning with MLP
  - `custom`: Custom diffusion policy implementation
  - `lerobot`: LeRobot diffusion policy (requires LeRobot installation)

## Output Structure

The script creates a structured output directory:

```
checkpoint_experiment_results/
└── {experiment_name}/
    ├── analysis/
    │   └── checkpoint_analysis.png      # Performance evolution plots
    ├── checkpoints/
    │   ├── checkpoint_epoch_5.pth       # Model checkpoints
    │   └── checkpoint_epoch_10.pth
    ├── datasets/
    │   └── {dataset_name}/              # Training data
    ├── evaluation/
    │   ├── all_checkpoint_results.json  # Consolidated results
    │   ├── checkpoint_epoch_5/          # Individual evaluation results
    │   └── checkpoint_epoch_6/
    ├── experiment_summary.json          # Experiment metadata
    └── logs/                            # Detailed logs
```

## Analysis Outputs

1. **Performance Evolution Plots**: Shows how mean return, success rate, and episode length change during training
2. **Performance Summary Table**: Tabular view of all checkpoint results
3. **Best Performance Identification**: Automatically identifies best-performing checkpoints
4. **Individual Episode Data**: Detailed results for each evaluation episode

## Example Results

The script will output analysis like:

```
📊 CHECKPOINT ANALYSIS SUMMARY
============================================================
Epoch    Return       Success Rate    Episode Length 
============================================================
5        -350.25      15.0            380.5          
10       -275.80      35.0            342.1          
15       -198.45      60.0            285.3          

🏆 BEST PERFORMANCE:
Best Return: Epoch 15 (-198.450)
Best Success Rate: Epoch 15 (60.0%)
```

## Use Cases

- **Training Analysis**: Understand how policy performance evolves during training
- **Overfitting Detection**: Identify if the model starts overfitting at later epochs
- **Hyperparameter Tuning**: Compare different training configurations
- **Early Stopping**: Determine optimal training duration
- **Policy Comparison**: Compare behavior cloning vs diffusion policies
- **Data Type Analysis**: Compare performance with random vs expert demonstrations
- **Architecture Evaluation**: Compare custom vs LeRobot diffusion implementations

## Requirements

- Python 3.11+
- PyTorch
- The prbenchIL1 conda environment
- PRBench environments
- At least 1GB of disk space for checkpoints and results
