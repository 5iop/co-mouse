"""
Test loss on first batch in TRAIN mode (with dropout enabled)
to match exactly what happens in training
"""
import torch
from config import Config
from dataset import get_dataloaders
from model import ConditionalLSTMVAE, vae_loss

print("Loading data...")
config = Config()

# Set same random seed as training
torch.manual_seed(Config.SEED)

train_loader, val_loader, num_users = get_dataloaders(config)

# Create model with same seed as training
model = ConditionalLSTMVAE(config, num_users).to(config.DEVICE)

# **IMPORTANT**: Set to train mode (dropout enabled)
model.train()

print("Testing FIRST batch in TRAIN mode (with dropout)...")

for batch_idx, (sequences, conditions) in enumerate(train_loader):
    sequences = sequences.to(config.DEVICE)
    for key in conditions:
        conditions[key] = conditions[key].to(config.DEVICE)

    # Forward pass (with dropout)
    x_recon, mu, logvar = model(sequences, conditions)

    # Compute loss
    total_loss, recon_loss, kl_loss = vae_loss(x_recon, sequences, mu, logvar, kl_weight=0.0)

    print(f"\nBatch {batch_idx}:")
    print(f"  Recon Loss: {recon_loss.item():.4f}")
    print(f"  KL Loss: {kl_loss.item():.4f}")

    if batch_idx == 0:
        print("\n" + "="*60)
        print("Expected first batch loss in training:")
        print("="*60)
        print(f"  Recon Loss: {recon_loss.item():.4f}")
        print(f"\nIf training Epoch 0 shows different value:")
        print("  - Check if checkpoint is being loaded")
        print("  - Check if data order is different")
        print("="*60)
        break
