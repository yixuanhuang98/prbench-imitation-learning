# New Features Added to PRBench Diffusion Policy Pipeline

## ✅ All PRBench Environments Support (35 environments!)

### What was added:
- **Dynamic environment discovery**: Automatically detects all available PRBench environments
- **Full environment support**: Now supports all 35 PRBench environments instead of just 6 hardcoded ones
- **Environment listing**: `--list-envs` command to see all available environments

### Available environment categories:
- **Geom2D**: Motion2D (5 variants), ClutteredRetrieval2D (3 variants), ClutteredStorage2D (4 variants), Obstruction2D (5 variants), StickButton2D (5 variants), PushPullHook2D
- **Dynamic2D**: DynObstruction2D (4 variants)  
- **Geom3D**: Motion3D
- **Dynamic3D**: TidyBot3D (7 variants across ground, table, cupboard scenes)

### Usage:
```bash
# List all available environments
python run_diffusion_pipeline.py --list-envs

# Use any environment with short name
python run_diffusion_pipeline.py --env motion2d-p3 --data-episodes 10
python run_diffusion_pipeline.py --env tidybot3d-ground-o5 --data-episodes 5
python run_diffusion_pipeline.py --env dynobstruction2d-o2 --data-episodes 8
```

---

## ✅ Video Recording for Demonstration Quality Inspection

### What was added:
- **Video saving during data generation**: Record GIF videos of expert/random trajectories
- **Quality inspection**: Visually verify demonstration quality before training
- **Automatic organization**: Videos saved in `videos/` subdirectory within dataset folder

### Features:
- **Automatic video naming**: `episode_N_expert.gif` or `episode_N_random.gif`
- **Efficient storage**: Videos saved as GIF format for compatibility
- **Error handling**: Graceful fallback if video rendering fails
- **Progress feedback**: Console messages showing when videos are saved

### Usage:
```bash
# Generate data with video recording
python run_diffusion_pipeline.py --env motion2d-p2 --data-episodes 5 --save-demo-videos

# Skip training/eval to just generate videos for inspection
python run_diffusion_pipeline.py --env pushpullhook2d --data-episodes 3 \
    --save-demo-videos --skip-training --skip-evaluation
```

### Output structure:
```
datasets/
  └── motion2d-p2_expert_5ep/
      ├── dataset.pkl
      ├── metadata.json
      └── videos/
          ├── episode_1_expert.gif
          ├── episode_2_expert.gif
          ├── episode_3_expert.gif
          ├── episode_4_expert.gif
          └── episode_5_expert.gif
```

---

## 🎯 Combined Usage Examples

### Inspect demonstration quality for a new environment:
```bash
# Generate data with videos for quality inspection
python run_diffusion_pipeline.py --env tidybot3d-table-o3 --data-episodes 3 \
    --save-demo-videos --skip-training --skip-evaluation

# Check the videos in: diffusion_pipeline_results/*/datasets/*/videos/
# If quality looks good, proceed with training
```

### Full pipeline with video documentation:
```bash
# Complete pipeline with video recording for documentation
python run_diffusion_pipeline.py --env clutteredretrieval2d-o10 \
    --data-episodes 10 --train-epochs 20 --save-demo-videos
```

### Compare expert vs random demonstrations:
```bash
# Generate expert demonstrations with videos
python run_diffusion_pipeline.py --env motion3d --data-episodes 5 \
    --data-type expert --save-demo-videos --skip-training --skip-evaluation

# Generate random demonstrations with videos  
python run_diffusion_pipeline.py --env motion3d --data-episodes 5 \
    --data-type random --save-demo-videos --skip-training --skip-evaluation
```

---

## 🚀 Benefits

### 1. **Comprehensive Environment Coverage**
- Access to all PRBench environments (35 total)
- Automatic discovery prevents missing environments
- Easy scaling to new environments as they're added

### 2. **Quality Assurance**
- Visual verification of demonstration quality
- Early detection of poor expert policies
- Ability to debug environment/policy issues before training

### 3. **Better Workflow**
- `--list-envs` for environment discovery
- Video inspection for quality control
- Streamlined pipeline from data → training → evaluation

### 4. **Improved Debugging**
- Videos help identify why training might fail
- Visual inspection of trajectory quality
- Better understanding of environment dynamics

---

## 📊 Summary Statistics

- **Environments**: 35 total (vs 6 previously)
- **Environment categories**: 4 (Geom2D, Dynamic2D, Geom3D, Dynamic3D)
- **New features**: 2 major (all environments + video saving)
- **Video formats**: GIF (high compatibility)
- **Quality**: Production-ready with error handling

The pipeline now provides comprehensive support for the entire PRBench suite with built-in quality inspection capabilities! 🎉
