"""
Download and inspect the G1 PickApple dataset from HuggingFace
"""
from datasets import load_dataset
import numpy as np

print("Downloading G1 PickApple dataset from HuggingFace...")
print("This may take a few minutes...")

# Download the dataset
dataset = load_dataset("unitreerobotics/G1_Brainco_PickApple_Dataset", split="train")

print(f"\n✅ Dataset downloaded!")
print(f"Total episodes/samples: {len(dataset)}")
print(f"\nDataset features:")
print(dataset.features)

# Inspect first sample
if len(dataset) > 0:
    sample = dataset[0]
    print(f"\n📊 First sample keys:")
    for key in sample.keys():
        value = sample[key]
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, dict):
            print(f"  {key}: dict with keys={list(value.keys())}")
        else:
            print(f"  {key}: {type(value).__name__}")

print("\n✅ Dataset inspection complete!")
print("\n💡 Next: Adapt environment to match this format")
