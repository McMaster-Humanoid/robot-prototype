#!/usr/bin/env python3
"""Quick test of G1 manipulation environment - No viewer for speed"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_manipulation.env import G1ManipulationEnv

print("="*60)
print("G1 Manipulation Environment - Quick Test")
print("="*60)

# Create environment WITHOUT viewer (much faster)
print("\n1. Creating G1 environment (no viewer)...")
env = G1ManipulationEnv(
    num_envs=1,
    task="reach",
    show_viewer=False,  # NO VIEWER = FAST
)

print(f"✓ G1 environment created!")
print(f"  - Robot has {len(env.arm_dof_idx)} arm DOFs")
print(f"  - Observation space: {env.observation_space}")
print(f"  - Action space: {env.action_space}")

# Test reset
print("\n2. Testing reset...")
obs, info = env.reset()
print(f"✓ Reset successful")
print(f"  - Image shape: {obs['image'].shape}")
print(f"  - State shape: {obs['state'].shape}")

# Test 10 steps only (fast)
print("\n3. Testing steps (10 steps)...")
for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i == 0:
        print(f"✓ Step successful")

print(f"✓ Completed 10 steps successfully!")

# Close
env.close()
print("\n" + "="*60)
print("SUCCESS! G1 manipulation environment works!")
print("="*60)
