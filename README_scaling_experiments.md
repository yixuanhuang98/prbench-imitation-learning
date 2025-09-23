# Behavior Cloning Scaling Experiments

This directory contains scripts for running comprehensive scaling experiments to evaluate how behavior cloning performance varies with the number of expert demonstrations.

## 🔧 Fixed Issues

### Success Detection Bug Fix
We identified and fixed a critical bug where expert demonstrations were incorrectly marked as failures even when they completed successfully. The issue was in the success determination logic in `run_diffusion_pipeline.py`:

**Problem**: Episodes that terminated early (before max steps) were marked as failures because the code only checked for explicit `success` flags from the environment.

**Solution**: Updated the logic to recognize that early termination (`terminated=True, truncated=False`) before reaching max steps indicates successful task completion.

**Result**: Expert demonstrations now correctly show 100% success rate instead of 0%.

## 📁 Files

### Core Scripts

1. **`run_scaling_experiment.py`** - Main scaling experiment framework
   - Runs experiments with different numbers of demonstrations
   - Generates comprehensive visualizations
   - Saves detailed results in JSON format

2. **`run_comprehensive_scaling_experiment.py`** - Full scaling study
   - Uses reasonable parameters for meaningful results
   - Includes user confirmation for long-running experiments
   - Generates publication-quality figures

### Pipeline Script
- **`scripts/run_diffusion_pipeline.py`** - Updated with fixed success detection

## 🚀 Usage

### Quick Test (Fast)
```bash
python run_scaling_experiment.py
```
- Uses minimal parameters (1 epoch, 1 eval episode)
- Tests demo counts: [1, 2, 5, 10, 20]
- Takes ~10 minutes

### Comprehensive Study (Recommended)
```bash
python run_comprehensive_scaling_experiment.py
```
- Uses proper parameters (100 epochs, 10 eval episodes)
- Tests demo counts: [1, 2, 5, 10, 20, 50]
- Takes ~30-60 minutes
- Generates high-quality results

### Custom Configuration
```python
from run_scaling_experiment import run_scaling_experiments, create_scaling_figure

results = run_scaling_experiments(
    env="motion2d-p1",
    policy_type="behavior_cloning", 
    demo_counts=[1, 5, 10, 20],
    train_epochs=100,
    eval_episodes=10
)

create_scaling_figure(results, save_path="my_results.png")
```

## 📊 Output Files

Each experiment generates:
- **Results JSON**: Detailed experimental data
- **Visualization PNG**: 4-panel figure showing:
  - Policy success rate vs demonstrations
  - Mean return vs demonstrations  
  - Expert demonstration quality
  - Episode length vs demonstrations

## 🔍 Key Metrics Tracked

### Policy Performance
- **Success Rate**: Percentage of episodes completed successfully
- **Mean Return**: Average episode return
- **Episode Length**: Average number of steps per episode

### Expert Demonstration Quality
- **Expert Success Rate**: Percentage of successful expert demonstrations
- **Expert Average Reward**: Average reward from expert episodes
- **Total Expert Episodes**: Number of expert demonstrations collected

## 🧪 Expected Results

With the fixed success detection:
- **Expert Success Rate**: Should be 100% (demonstrations complete in 40-90 steps)
- **Policy Performance**: Should improve with more demonstrations (if training is sufficient)
- **Scaling Behavior**: More demonstrations → better policy performance (in theory)

## 🛠️ Troubleshooting

### Low Policy Success Rate
If trained policies show 0% success rate with -400 return:
- Increase training epochs (try 1000+ for better convergence)
- Check that expert demonstrations are truly successful
- Verify environment setup

### Expert Demonstrations Failing
If expert success rate is 0%:
- Check that the success detection fix is applied
- Try increasing planning timeout: `--planning-timeout 120`
- Verify BilevelPlanningAgent is working correctly

### Script Errors
- Ensure prbenchIL1 conda environment is activated
- Install required packages: `pip install matplotlib pandas`
- Check file permissions: `chmod +x *.py`

## 📈 Analysis Tips

1. **Correlation Analysis**: Look for positive correlation between demo count and success rate
2. **Diminishing Returns**: Performance gains may plateau at higher demo counts  
3. **Expert Quality**: Ensure expert success rate remains high across all experiments
4. **Statistical Significance**: Use multiple evaluation episodes for reliable statistics

## 🎯 Next Steps

1. Run comprehensive scaling study with motion2d-p1
2. Extend to other environments (motion2d-p2, obstruction2d, etc.)
3. Compare behavior cloning vs diffusion policy scaling
4. Investigate optimal training epoch counts for each demo count
