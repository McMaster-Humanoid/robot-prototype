"""
Inspect the dimensions of observation and action in G1 PickApple dataset
"""
from datasets import load_dataset
import numpy as np

print("Loading dataset...")
dataset = load_dataset("unitreerobotics/G1_Brainco_PickApple_Dataset", split="train")

# Get first sample
sample = dataset[0]

print(f"\n📊 Data Dimensions:")
print(f"  observation.state: {len(sample['observation.state'])} values")
print(f"  action: {len(sample['action'])} values")
print(f"  timestamp: {sample['timestamp']}")
print(f"  episode_index: {sample['episode_index']}")

# Check a few samples to understand the structure
print(f"\n📈 First 3 observation.state values: {sample['observation.state'][:3]}")
print(f"📈 First 3 action values: {sample['action'][:3]}")

# Count unique episodes
unique_episodes = set(dataset['episode_index'])
print(f"\n📦 Total episodes: {len(unique_episodes)}")
print(f"📦 Average samples per episode: {len(dataset) / len(unique_episodes):.1f}")

# Check if there are images (sometimes in 'observation.images.cam_0' etc)
print(f"\n🔍 All feature keys:")
for key in dataset.features.keys():
    print(f"  - {key}")
