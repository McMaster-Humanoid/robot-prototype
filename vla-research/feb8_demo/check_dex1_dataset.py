"""
Check if Dex1 dataset has images (required for ACT/VLA)
"""
from datasets import load_dataset

print("Checking G1_Dex1 datasets for image data...")
print("\nTrying GraspOreo dataset...")

try:
    # Try a smaller dataset first to check structure
    dataset = load_dataset("unitreerobotics/G1_Brainco_GraspOreo_Dataset", split="train", streaming=True)
    
    # Get first sample
    sample = next(iter(dataset))
    
    print(f"\n✅ Dataset loaded!")
    print(f"\n📊 Available features:")
    for key in sample.keys():
        print(f"  - {key}")
    
    # Check for image features
    image_features = [k for k in sample.keys() if 'image' in k.lower() or 'camera' in k.lower()]
    
    if image_features:
        print(f"\n🎉 FOUND IMAGES!")
        for img_key in image_features:
            print(f"  - {img_key}")
    else:
        print(f"\n❌ No image features found")
        print(f"   Dataset is state-only (joint positions)")

except Exception as e:
    print(f"\n❌ Error: {e}")

print("\n" + "="*60)
