#!/usr/bin/env python3
"""
Quick test script for G1 manipulation environment
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import G1ManipulationEnv

def test_basic():
    """Test basic environment functionality"""
    print("="*60)
    print("Testing G1 Manipulation Environment")
    print("="*60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = G1ManipulationEnv(
        num_envs=1,
        task="reach",
        show_viewer=False,  # No viewer for headless testing
    )
    print("✓ Environment created successfully")
    
    # Test reset
    print("\n2. Testing reset...")
    obs, info = env.reset()
    print(f"✓ Reset successful")
    print(f"  - Image shape: {obs['image'].shape}")
    print(f"  - State shape: {obs['state'].shape}")
    print(f"  - Image dtype: {obs['image'].dtype}")
    print(f"  - State dtype: {obs['state'].dtype}")
    
    # Test step
    print("\n3. Testing steps...")
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i == 0:
            print(f"✓ Step successful")
            print(f"  - Reward: {reward}")
            print(f"  - Terminated: {terminated}")
            print(f"  - Truncated: {truncated}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print(f"✓ Completed 50 steps")
    
    # Clean up
    print("\n4. Closing environment...")
    env.close()
    print("✓ Environment closed")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

if __name__ == "__main__":
    test_basic()
