#!/usr/bin/env python3
"""Test G1 with visual viewer - WARNING: VERY SLOW in WSL2"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_manipulation.env import G1ManipulationEnv

print("Creating G1 environment WITH viewer (this is SLOW)...")
print("Viewer window should appear...")

env = G1ManipulationEnv(
    num_envs=1,
    task="reach",
    show_viewer=True,  # ENABLE VIEWER
)

print("✓ Environment created! Viewer should be visible.")
print("Running 50 steps...")

obs, info = env.reset()

for i in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if i % 10 == 0:
        print(f"Step {i}/50")

print("Done! Close viewer window to exit.")
env.close()
