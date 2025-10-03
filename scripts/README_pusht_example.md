# PushT Dataset Training Example

This script demonstrates training and evaluating diffusion policies on the standard [lerobot PushT benchmark dataset](https://huggingface.co/datasets/lerobot/pusht).

## Overview

The PushT task is a standard 2D manipulation benchmark where an agent must push a T-shaped block to a target location. This script provides an end-to-end example of:

1. Loading the pre-collected lerobot PushT dataset
2. Training diffusion or behavior cloning policies
3. Evaluating trained policies in the PushT environment

## Quick Start

### Basic Usage

Train a diffusion policy:
```bash
cd scripts
python run_pusht_example.py --policy-type diffusion --epochs 50
```

Train a behavior cloning policy:
```bash
python run_pusht_example.py --policy-type behavior_cloning --epochs 50
```

Train both and compare:
```bash
python run_pusht_example.py --policy-type both --epochs 100
```

### With Evaluation Videos

```bash
python run_pusht_example.py --policy-type diffusion --epochs 50 --save-videos
```

**Note:** Due to a dependency conflict (see Troubleshooting section):
- Training on PushT dataset works perfectly ✅
- Evaluation on PushT environment is skipped (dependency conflict) ⚠️
- The script will complete successfully with clear messaging

### With W&B Logging

```bash
python run_pusht_example.py --policy-type diffusion --epochs 100 --use-wandb
```

## Dataset

The script automatically downloads the PushT dataset from HuggingFace on first run:
- **Dataset**: `lerobot/pusht`
- **Size**: ~100MB
- **Location**: `~/.cache/huggingface/lerobot/pusht/`
- **Episodes**: ~500 expert demonstrations
- **Task**: Push T-shaped block to target position

The dataset only needs to be downloaded once and will be cached for future runs.

## Command-Line Arguments

### Training Options

- `--policy-type {diffusion,behavior_cloning,both}`: Type of policy to train (default: `diffusion`)
- `--epochs N`: Number of training epochs (default: `50`)
- `--batch-size N`: Training batch size (default: `64`)
- `--learning-rate LR`: Learning rate (default: `1e-4`)
- `--use-wandb`: Enable Weights & Biases logging

### Evaluation Options

- `--eval-episodes N`: Number of episodes to evaluate (default: `10`)
- `--save-videos`: Save videos of evaluation rollouts
- `--max-episode-steps N`: Maximum steps per episode (default: `300`)

### Pipeline Control

- `--skip-training`: Skip training and use existing model
- `--skip-evaluation`: Skip evaluation step
- `--model-path PATH`: Path to existing model (required if `--skip-training`)

### Output Options

- `--output-dir DIR`: Output directory for results (default: `./pusht_results`)
- `--experiment-name NAME`: Custom experiment name (auto-generated if not provided)

## Example Workflows

### Quick Test Run

Fast training for testing the pipeline:
```bash
python run_pusht_example.py \
    --policy-type diffusion \
    --epochs 10 \
    --batch-size 32 \
    --eval-episodes 5
```

Expected time: ~5-10 minutes

### Full Training

Recommended settings for good performance:
```bash
python run_pusht_example.py \
    --policy-type diffusion \
    --epochs 200 \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --eval-episodes 20 \
    --save-videos \
    --use-wandb
```

Expected time: ~1-2 hours

### Compare Policies

Train and compare both diffusion and behavior cloning:
```bash
python run_pusht_example.py \
    --policy-type both \
    --epochs 100 \
    --eval-episodes 20 \
    --save-videos
```

This will output a comparison table showing performance metrics for both policies.

### Evaluate Existing Model

Evaluate a previously trained model without retraining:
```bash
python run_pusht_example.py \
    --skip-training \
    --model-path ./pusht_results/my_experiment/models/pusht_diffusion_model.pth \
    --eval-episodes 50 \
    --save-videos
```

## Output Structure

Results are organized in the output directory:

```
pusht_results/
└── pusht_diffusion_20241003_143022/
    ├── experiment_config.json       # Configuration used
    ├── summary.json                 # Pipeline summary
    ├── models/
    │   └── pusht_diffusion_model.pth    # Trained model checkpoint
    ├── evaluation/
    │   └── diffusion/
    │       ├── evaluation_results.json  # Detailed metrics
    │       ├── episode_returns.png      # Return plot
    │       └── videos/                  # Rollout videos (if --save-videos)
    └── logs/
        ├── diffusion/               # Training logs
        └── eval_diffusion/          # Evaluation logs
```

## Expected Performance

### After 50 Epochs
- Mean return: ~200-300 (baseline random: ~0-50)
- Some task understanding
- Occasional successful pushes

### After 200 Epochs
- Mean return: ~400-600
- Consistent task understanding
- Frequent successful pushes to target

### State-of-the-Art
- Published results on PushT: ~600-800 mean return
- May require 500+ epochs and hyperparameter tuning

Note: Performance varies based on random seed, hyperparameters, and training hardware.

## Hardware Requirements

### Minimum
- CPU: Any modern CPU
- RAM: 8GB
- GPU: Not required (will use CPU)
- Disk: ~500MB for dataset and models

### Recommended
- CPU: 4+ cores
- RAM: 16GB
- GPU: NVIDIA GPU with 4GB+ VRAM
- Disk: ~2GB for multiple experiments

GPU training is 5-10x faster than CPU.

## Troubleshooting

### Import Errors
If you see `ImportError: No module named 'lerobot'`:
```bash
pip install lerobot
```

### Evaluation Skipped

**Dependency Conflict Issue:**

There is a known dependency conflict:
- `gym-pusht` requires `pymunk < 7.0.0`
- `prbench` requires `pymunk == 7.1.0`

This means:
- ✅ **Training on PushT dataset works perfectly** (only needs lerobot)
- ⚠️ **Evaluation on PushT environment is not available** (requires incompatible pymunk version)

The script will detect this and show:
```
⚠️  Warning: gym-pusht has dependency conflict with current pymunk version.
   Evaluation will be skipped.
   Note: PRBench requires pymunk 7.1.0, but gym-pusht requires pymunk <7.0.0
   Training on PushT dataset works, but environment evaluation is not available.
```

**Workaround for Evaluation:**
If you specifically need PushT environment evaluation, you can:
1. Create a separate conda environment for PushT evaluation only
2. Use the trained models from this environment
3. Evaluate them in the PushT-specific environment

**Note:** This limitation only affects PushT. All PRBench environments work normally for both training and evaluation.

### Dataset Download Issues
If behind a firewall or having connection issues:
1. Download manually from [HuggingFace](https://huggingface.co/datasets/lerobot/pusht)
2. Place in `~/.cache/huggingface/lerobot/pusht/`

### CUDA Out of Memory
Reduce batch size:
```bash
python run_pusht_example.py --batch-size 16
```

### Slow Training
- Enable GPU if available
- Increase batch size (if memory allows)
- Reduce `--num-workers` if data loading is slow

## Comparison with PRBench Environments

This script works with the lerobot PushT dataset, while `run_diffusion_pipeline.py` works with PRBench environments (Motion2D, etc.). Key differences:

| Feature | PushT (this script) | PRBench Environments |
|---------|---------------------|----------------------|
| Dataset | Pre-collected from HuggingFace | Generate on-the-fly |
| Data source | Expert demonstrations | Random/expert policy |
| Environment | Standard benchmark | Research environments |
| Training | Ready to train immediately | Requires data collection |
| Use case | Benchmarking, testing | Research, scaling studies |

## Related Files

- **Test suite**: `tests/test_pusht_dataset.py` - Integration tests for PushT
- **Test docs**: `tests/README_PUSHT_TESTS.md` - How to run the tests
- **Pipeline script**: `run_diffusion_pipeline.py` - For PRBench environments

## References

- [LeRobot PushT Dataset](https://huggingface.co/datasets/lerobot/pusht)
- [LeRobot Library](https://github.com/huggingface/lerobot)
- [Diffusion Policy Paper](https://diffusion-policy.cs.columbia.edu/)

## Tips for Best Results

1. **Start small**: Run a quick test (10 epochs) to verify everything works
2. **Use W&B**: Track experiments with `--use-wandb` for easier comparison
3. **Save videos**: Use `--save-videos` to visually inspect policy behavior
4. **Try both**: Compare diffusion vs behavior cloning with `--policy-type both`
5. **Tune hyperparameters**: Learning rate and batch size significantly impact performance
6. **Train longer**: For publication-quality results, train 200+ epochs
