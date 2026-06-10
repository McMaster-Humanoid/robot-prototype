# VLA/LeRobot Setup Guide - McMaster Humanoid

## Quick Setup

```bash
# 1. Install LeRobot
python3 -m venv vla_env
source vla_env/bin/activate
pip install lerobot

# 2. Train ACT on example dataset
lerobot-train \
  --policy.type act \
  --dataset.repo_id lerobot/pusht \
  --steps 5000 \
  --output_dir outputs/test_run \
  --policy.push_to_hub false \
  --dataset.video_backend pyav
```

## Critical Issue: Video Backend

**Problem**: Training crashes with FFmpeg error

**Solution**: Always add `--dataset.video_backend pyav` to training commands

## System Info

- LeRobot: 0.4.3
- Python: 3.12.3
- OS: Ubuntu 24.04 (WSL2)
- GPU: NVIDIA RTX 4070

## Training Results

Trained ACT on PushT dataset:
- Time: 11 minutes (5000 steps)
- Loss: 6.5 → 0.4 (94% reduction)
- Model: 52M parameters
- Speed: 0.13s per step

## Common Issues

1. **CLI syntax**: Use `--policy.type act` NOT `policy=act`
2. **FFmpeg error**: Add `--dataset.video_backend pyav`
3. **Output directory**: Delete old one or use new name
4. **Hub upload**: Add `--policy.push_to_hub false`

## Next Steps for Unitree G1

1. Review LeRobot G1 docs: https://huggingface.co/docs/lerobot/unitree_g1
2. Set up MuJoCo simulation
3. Record demonstrations
4. Train ACT for manipulation tasks
5. Test in simulation then deploy to robot

## Key Takeaway

LeRobot + ACT works well on our hardware. Same workflow applies to G1 humanoid for manipulation tasks.
