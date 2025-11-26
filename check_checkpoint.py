"""
Check if a checkpoint exists and what loss it has
"""
import os
import torch
from pathlib import Path

checkpoint_dir = Path("checkpoints")

if not checkpoint_dir.exists():
    print(f"Checkpoint directory does not exist: {checkpoint_dir}")
    exit(0)

checkpoints = list(checkpoint_dir.glob("*.pt"))

if not checkpoints:
    print("✓ No checkpoints found - training from scratch")
    exit(0)

print("="*60)
print(f"Found {len(checkpoints)} checkpoint(s):")
print("="*60)

for cp in sorted(checkpoints, key=lambda x: x.stat().st_mtime):
    size_mb = cp.stat().st_size / 1024 / 1024
    mtime = cp.stat().st_mtime

    try:
        # Try to load checkpoint metadata
        checkpoint = torch.load(cp, map_location='cpu')
        epoch = checkpoint.get('epoch', 'unknown')
        loss = checkpoint.get('loss', 'unknown')

        print(f"\n{cp.name}:")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Epoch: {epoch}")
        print(f"  Loss: {loss}")

        # Check if model weights look trained
        if 'model_state_dict' in checkpoint:
            # Get a sample weight
            first_key = list(checkpoint['model_state_dict'].keys())[0]
            first_weight = checkpoint['model_state_dict'][first_key]
            print(f"  Sample weight mean: {first_weight.mean().item():.6f}")
            print(f"  Sample weight std:  {first_weight.std().item():.6f}")

    except Exception as e:
        print(f"\n{cp.name}: Error loading - {e}")

print("\n" + "="*60)
print("If training shows loss=0.029 with these checkpoints present,")
print("the model might be auto-loading them somehow.")
print("="*60)
