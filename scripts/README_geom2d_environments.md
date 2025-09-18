# Geom2D Environments Support

The diffusion pipeline now supports expert demonstration collection and training for all geom2d environments using bilevel planning agents.

## Supported Environments

| Environment | Parameter | Example Usage | Status |
|-------------|-----------|---------------|--------|
| **Motion2D** | `p` (passages) | `motion2d-p1`, `motion2d-p2`, `motion2d-p3` | ✅ **Working** |
| **Obstruction2D** | `o` (obstructions) | `obstruction2d-o0`, `obstruction2d-o1`, `obstruction2d-o2` | ✅ **Working** |
| **ClutteredRetrieval2D** | `o` (obstructions) | `clutteredretrieval2d-o1`, `clutteredretrieval2d-o3` | ✅ **Working** |
| **ClutteredStorage2D** | `b` (blocks) | `clutteredstorage2d-b1`, `clutteredstorage2d-b7` | ✅ **Working** |
| **StickButton2D** | `b` (buttons) | `stickbutton2d-b1`, `stickbutton2d-b2`, `stickbutton2d-b5` | ⚠️ Planning Issues |

✅ **Fixed**: Import issues have been resolved! **4 out of 5** environments are now fully functional for demonstration collection and training.

## Usage Examples

### 1. Motion2D Environment (Backward Compatible)
```bash
conda activate prbenchIL1
python run_diffusion_pipeline.py \
    --env motion2d-p2 \
    --data-type expert \
    --data-episodes 20 \
    --train-epochs 100 \
    --experiment-name motion2d_p2_expert
```

### 2. StickButton2D Environment
```bash
conda activate prbenchIL1
python run_diffusion_pipeline.py \
    --env stickbutton2d-b3 \
    --data-type expert \
    --data-episodes 15 \
    --train-epochs 80 \
    --experiment-name stickbutton2d_b3_expert
```

### 3. Obstruction2D Environment
```bash
conda activate prbenchIL1
python run_diffusion_pipeline.py \
    --env obstruction2d-o1 \
    --data-type expert \
    --data-episodes 25 \
    --train-epochs 120 \
    --experiment-name obstruction2d_o1_expert
```

### 4. ClutteredStorage2D Environment
```bash
conda activate prbenchIL1
python run_diffusion_pipeline.py \
    --env clutteredstorage2d-b5 \
    --data-type expert \
    --data-episodes 30 \
    --train-epochs 150 \
    --experiment-name cluttered_storage_b5_expert
```

### 5. ClutteredRetrieval2D Environment
```bash
conda activate prbenchIL1
python run_diffusion_pipeline.py \
    --env clutteredretrieval2d-o2 \
    --data-type expert \
    --data-episodes 20 \
    --train-epochs 100 \
    --experiment-name cluttered_retrieval_o2_expert
```

## Advanced Usage

### Custom Environment Parameters
You can override the environment parameter using the `--env-param` flag:

```bash
python run_diffusion_pipeline.py \
    --env stickbutton2d-b1 \
    --env-param 5 \
    --data-type expert \
    --experiment-name custom_param_example
```

### Bilevel Planning Configuration
Fine-tune the bilevel planning agent:

```bash
python run_diffusion_pipeline.py \
    --env obstruction2d-o1 \
    --data-type expert \
    --max-abstract-plans 15 \
    --samples-per-step 5 \
    --planning-timeout 45.0 \
    --experiment-name fine_tuned_planning
```

### Complete Pipeline with Evaluation
```bash
python run_diffusion_pipeline.py \
    --env motion2d-p1 \
    --data-type expert \
    --data-episodes 50 \
    --train-epochs 200 \
    --eval-episodes 20 \
    --save-videos \
    --experiment-name complete_motion2d_pipeline
```

## Key Features

### 🔄 **Automatic Environment Detection**
- The pipeline automatically detects environment type and parameters from the environment name
- Supports both explicit parameter specification and automatic parsing

### 🎯 **Bilevel Planning Integration** 
- Uses expert bilevel planning agents for high-quality demonstrations
- Configurable planning parameters for different environments
- Automatic environment model creation and agent initialization

### 📊 **Comprehensive Logging**
- Detailed logs for each environment type
- Success rate and reward tracking
- Episode-by-episode progress monitoring

### 🔧 **Flexible Configuration**
- All standard diffusion policy hyperparameters supported
- Environment-specific parameter overrides
- Video recording and evaluation options

## Environment-Specific Notes

### Motion2D
- **Parameter**: Number of passages (1-3)
- **Difficulty**: Increases with more passages
- **Typical Episodes**: 20-50 for good performance

### StickButton2D  
- **Parameter**: Number of buttons (1-5+)
- **Complexity**: Increases exponentially with buttons
- **Typical Episodes**: 15-30 depending on button count

### Obstruction2D
- **Parameter**: Number of obstructions (0-2+)
- **Challenge**: Navigation around obstacles
- **Typical Episodes**: 20-40 for robust policies

### ClutteredStorage2D
- **Parameter**: Number of blocks (1-7+)
- **Task**: Organize blocks on shelves
- **Typical Episodes**: 25-50 for complex arrangements

### ClutteredRetrieval2D
- **Parameter**: Number of obstructions (1-3+)
- **Task**: Retrieve target objects
- **Typical Episodes**: 20-35 for reliable retrieval

## Troubleshooting

### Import Errors
Some environments may have missing dependencies in the bilevel planning models. If you encounter import errors:

1. Try with a simpler environment first (e.g., `obstruction2d-o1`)
2. Check that all third-party dependencies are properly installed
3. Verify that the prbenchIL1 conda environment is activated

### Low Success Rates
Expert agents may not always succeed, especially in complex environments:

1. Increase planning timeout: `--planning-timeout 60.0`
2. Increase abstract plans: `--max-abstract-plans 20`
3. Try simpler environment variants (fewer objects/obstructions)

### Memory Issues
For environments with many objects:

1. Reduce batch size: `--batch-size 16`
2. Reduce number of episodes: `--data-episodes 15`
3. Use CPU training if GPU memory is limited: add force_cpu config

## Performance Tips

1. **Start Simple**: Begin with basic environment variants (few objects)
2. **Collect More Data**: Complex environments benefit from 30+ episodes
3. **Longer Training**: Use 100-200 epochs for better convergence
4. **Monitor Success**: Check demonstration success rates in logs
5. **Save Videos**: Use `--save-videos` to visualize agent behavior

## Example Output Structure

```
diffusion_pipeline_results/
└── obstruction2d-o1_expert_2025-09-17-112030/
    ├── datasets/
    │   └── obstruction2d-o1_expert_25ep/
    │       ├── dataset.pkl
    │       └── metadata.json
    ├── models/
    │   └── obstruction2d-o1_expert_2025-09-17-112030_custom_model.pth
    ├── evaluation/
    │   ├── results.json
    │   └── plots/
    ├── experiment_config.json
    └── pipeline_summary.json
```

This comprehensive support enables training diffusion policies on all major geom2d environments with expert demonstrations! 🚀
