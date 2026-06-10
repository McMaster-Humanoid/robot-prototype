#!/usr/bin/env python3
"""Simple scripted reaching behavior to test G1 manipulation"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from g1_manipulation.env import G1ManipulationEnv

print("Testing G1 reaching with simple controller...")

# Create environment
env = G1ManipulationEnv(
    num_envs=1,
    task="reach",
    show_viewer=True,  # Show the robot
)

print("✓ Environment created!")
print("\nWatching G1 reach towards cube...")
print("(Simple scripted motion - not learned policy yet)")

# Reset
obs, info = env.reset()

# Simple reaching controller
# This gradually moves arms forward (very basic!)
for i in range(100):
    # Create action: slowly extend arms forward
    # Action is 10D: left_arm(5) + right_arm(5)
    # Positive values extend forward
    
    progress = i / 100.0  # 0 to 1
    
    # Simple reaching motion
    action = np.array([
        0.3 * progress,  # left shoulder pitch
        0.0,             # left shoulder roll
        0.0,             # left shoulder yaw
        0.5 * progress,  # left elbow
        0.0,             # left wrist roll
        0.3 * progress,  # right shoulder pitch
        0.0,             # right shoulder roll
        0.0,             # right shoulder yaw
        0.5 * progress,  # right elbow
        0.0,             # right wrist roll
    ], dtype=np.float32)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    if i % 20 == 0:
        print(f"Step {i}/100 - Arms extending...")
    
    if terminated or truncated:
        obs, info = env.reset()

print("\n✓ Test complete!")
print("\nWhat you saw:")
print("- Simple scripted reaching motion")
print("- NOT a trained policy")
print("\nTo actually grasp:")
print("1. Record demonstrations with teleoperation")
print("2. Train ACT policy on demonstrations")
print("3. Deploy trained policy = intelligent grasping!")

env.close()
