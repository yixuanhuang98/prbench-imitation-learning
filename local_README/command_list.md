evaluation: python scripts/lerobot_eval.py --policy.path=/home/yixuan/prbench_dir/prbench-imitation-learning/outputs/train/2025-10-08/18-17-40_prbench_diffusion/checkpoints/050000/pretrained_model --env.type=prbench --env.task=Motion2D-p1-v0 --eval.batch_size=20 --eval.n_episodes=20 --policy.use_amp=false --policy.device=cuda  --policy.crop_shape=[64,64]

evaluation script for multiple seeds:
python scripts/lerobot_eval_multi_seed.py     --policy.path=outputs/train/2025-11-19/18-25-35_prbench_diffusion/checkpoints/030000/pretrained_model     --env.type=prbench     --env.task=Motion2D-p0-v0     --eval.batch_size=20     --eval.n_episodes=50     --policy.use_amp=false     --policy.device=cuda     --policy.crop_shape=[64,64]     --num_seeds=5     --base_seed=0


dataset collection: 
python scripts/generate_expert_demonstrations.py     --expert_env=motion2d     --expert_env_param=2     --expert_episodes=100     --expert_save_videos

lerobot format: 
python scripts/convert_expert_to_lerobot_v3.py       --expert_data_dir expert_data/motion2d_p0_20251008_170831       --output_dir datasets/motion2d_lerobot_300       --repo_id motion2d_expert_300       --fps 10

training: 
python scripts/train_lerobot_direct.py --dataset.repo_id=motion2d_p2_v1 --dataset.root=datasets/motion2d_p2_v1 --policy.type=diffusion --policy.repo_id=yixuanh/motion2d_policy --steps=30000 --eval_freq=10000 --save_freq=10000 --policy.device=cuda --policy.push_to_hub=false --env.type=prbench --env.task=Motion2D-p2-v0 --policy.crop_shape=[64,64]

python scripts/train_lerobot_direct.py --dataset.repo_id=stick2d_b1_v0 --dataset.root=datasets/stick2d_b1_v0 --policy.type=diffusion --policy.repo_id=yixuanh/motion2d_policy --steps=300 --eval_freq=100 --save_freq=100 --policy.device=cuda --policy.push_to_hub=false --env.type=prbench --env.task=StickButton2D-b1-v0 --policy.crop_shape=[64,64]

## motion2d-p0
training: 
python scripts/train_lerobot_direct.py --dataset.repo_id=motion2d_teleop --dataset.root=datasets/motion2d_teleop_v3 --policy.type=diffusion --policy.repo_id=yixuanh/motion2d_policy --steps=50000 --eval_freq=10000 --save_freq=10000 --policy.device=cuda --policy.push_to_hub=false --env.type=prbench --env.task=Motion2D-p0-v0 --policy.crop_shape=[64,64]

## obstruction2d_p1
lerobot format: 
python scripts/convert_expert_to_lerobot_v3.py       --expert_data_dir expert_data/obstruction2d_p1_20251008_200243       --output_dir datasets/obstruction2d_p1_lerobot_180       --repo_id obstruction2d_p1_lerobot_180       --fps 10

## clutteredretrieval2d_p1
python scripts/train_lerobot_direct.py --dataset.repo_id=clutteredretrieval2d_p1_expert_300 --dataset.root=datasets/clutteredretrieval2d_p1_lerobot_300 --policy.type=diffusion --policy.repo_id=yixuanh/clutteredretrieval2d_policy --steps=100000 --eval_freq=10000 --save_freq=10000 --policy.device=cuda --policy.push_to_hub=false --env.type=prbench --env.task=ClutteredRetrieval2D-o1-v0 --policy.crop_shape=[64,64]