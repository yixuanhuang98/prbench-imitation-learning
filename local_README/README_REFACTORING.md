# Diffusion Policy Refactoring Summary

## Overview
The scripts directory has been refactored to improve code organization and maintainability. The core functionality has been moved into separate modules in the `src/` directory, with only the main pipeline script remaining in `scripts/`.

## New Structure

### `src/prbench_imitation_learning/`
Contains the core library modules:

- **`policy.py`**: Core diffusion policy models and dataset classes
  - `DiffusionPolicy`: Main diffusion policy model
  - `DiffusionPolicyDataset`: Dataset wrapper for LeRobot datasets
  - `ConditionalUNet1D`: 1D U-Net for noise prediction

- **`train.py`**: Training functionality
  - `train_diffusion_policy()`: Main training function
  - `get_default_training_config()`: Default configuration
  - Enhanced logging to files and console

- **`evaluate.py`**: Evaluation functionality
  - `PolicyEvaluator`: Class for evaluating trained policies
  - `evaluate_policy()`: Convenience function for evaluation
  - Automatic plot generation and video saving

- **`data_generation.py`**: Data generation functionality
  - `generate_lerobot_dataset()`: Main data generation function
  - Support for expert and random policies
  - Automatic metadata saving

- **`__init__.py`**: Package initialization with clean imports

### `scripts/`
Contains only the main pipeline script:

- **`run_diffusion_pipeline.py`**: Updated to use the new modules
  - Cleaner code using the refactored functions
  - Better error handling and logging
  - All functionality preserved

### `logs/`
New directory for storing all logs:
- Training logs
- Evaluation logs
- Pipeline execution logs
- Centralized logging location

## Benefits

1. **Better Organization**: Code is now organized into logical modules
2. **Reusability**: Functions can be imported and used independently
3. **Maintainability**: Easier to modify and extend individual components
4. **Testing**: Each module can be tested independently
5. **Documentation**: Cleaner API with proper docstrings
6. **Logging**: Centralized logging to `logs/` directory

## Usage

The main pipeline script works exactly the same as before:

```bash
cd scripts/
python run_diffusion_pipeline.py --env motion2d --data-episodes 20 --train-epochs 50
```

But now you can also import and use individual components:

```python
from prbench_imitation_learning import (
    generate_lerobot_dataset,
    train_diffusion_policy,
    evaluate_policy
)

# Generate data
dataset_path = generate_lerobot_dataset("motion2d", "my_dataset", 50, "expert", "./data")

# Train model
train_diffusion_policy(dataset_path, "./model.pth", config)

# Evaluate
results = evaluate_policy("./model.pth", "prbench/Motion2D-p2-v0", 10)
```

## Migration Notes

- Old script files have been removed from `scripts/`
- All functionality is preserved in the new modules
- Logging now goes to `logs/` directory instead of being scattered
- The main pipeline script interface remains unchanged
