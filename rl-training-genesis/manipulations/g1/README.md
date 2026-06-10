# G1 Manipulation Environment

Genesis-based manipulation environment for Unitree G1 humanoid robot, compatible with LeRobot/VLA training.

## Features

- **Genesis Physics**: Fast, GPU-accelerated simulation
- **Vision Input**: RGB camera observations (480x640)
- **Proprioceptive State**: Joint positions and velocities
- **Arm Control**: 14 DOF (7 per arm)
- **Gymnasium Interface**: Compatible with LeRobot
- **Manipulation Tasks**: Reach, pick, place, etc.

## Quick Start

### Test the Environment

```bash
cd /home/sehaj/one-leg-robot
source vla_env/bin/activate

# Run basic test
python g1_manipulation/test_env.py

# Or test directly
python g1_manipulation/env.py
```

### Use with Python

```python
from g1_manipulation import G1ManipulationEnv

# Create environment
env = G1ManipulationEnv(
    num_envs=1,
    task="reach",
    show_viewer=True
)

# Standard Gymnasium interface
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## Observation Space

```python
{
    "image": Box(0, 255, (480, 640, 3), uint8),  # RGB camera
    "state": Box(-inf, inf, (28,), float32)      # Joint pos + vel
}
```

## Action Space

```python
Box(-1, 1, (14,), float32)  # Normalized arm joint positions
```

## Integration with LeRobot

### Record Demonstrations

```bash
# Register environment
export PYTHONPATH=/home/sehaj/one-leg-robot:$PYTHONPATH

# Record with LeRobot (TODO: implement teleoperation)
lerobot-record \
  --env-name g1_manipulation:G1ManipulationEnv \
  --episodes 50 \
  --fps 30 \
  --repo-id your-username/g1-reach-dataset
```

### Train ACT Policy

```bash
lerobot-train \
  --policy.type act \
  --dataset.repo_id your-username/g1-reach-dataset \
  --steps 5000 \
  --output_dir outputs/g1_act \
  --policy.push_to_hub false \
  --dataset.video_backend pyav
```

## Environment Configuration

Key parameters in `G1ManipulationEnv.__init__()`:

- `num_envs`: Number of parallel environments (default: 1)
- `task`: Task type - "reach", "pick", "place" (default: "reach")
- `show_viewer`: Show Genesis viewer (default: True)
- `device`: "cuda" or "cpu" (default: "cuda")

## Tasks

Currently implemented:
- **reach**: Reach towards target cube

TODO:
- **pick**: Pick up cube
- **place**: Place cube at target location
- **stack**: Stack multiple cubes

## Current Limitations

1. **Teleoperation**: Not yet implemented (needed for recording demonstrations)
2. **Reward Function**: Basic placeholder (needs task-specific implementation)
3. **Multi-task**: Only single task at a time
4. **End-effector**: No gripper model yet

## Next Steps

1. Implement teleoperation interface (keyboard/gamepad)
2. Add proper task rewards
3. Record demonstration dataset
4. Train ACT policy
5. Evaluate on test tasks

## File Structure

```
g1_manipulation/
├── __init__.py       # Package init
├── env.py           # Main environment
├── test_env.py      # Test script
├── README.md        # This file
└── tasks/           # Task definitions (future)
```

## Dependencies

- genesis-world (physics simulation)
- gymnasium (RL interface)
- numpy
- torch
- LeRobot (for data collection & training)

## Credits

Based on `rl-training-genesis/go2_env.py` from McMaster Humanoid team.
Adapted for manipulation with vision for VLA training.
