"""
Test vae_loss on actual training data to see real MSE values
"""
import torch
from config import Config
from dataset import get_dataloaders
from model import ConditionalLSTMVAE, vae_loss

print("Loading data...")
config = Config()
train_loader, val_loader, num_users = get_dataloaders(config)

# Get one batch
for sequences, conditions in train_loader:
    sequences = sequences.to(config.DEVICE)
    for key in conditions:
        conditions[key] = conditions[key].to(config.DEVICE)

    print(f"\nBatch shape: {sequences.shape}")
    print(f"Batch size: {sequences.shape[0]}, Seq len: {sequences.shape[1]}, Features: {sequences.shape[2]}")

    # Create untrained model
    model = ConditionalLSTMVAE(config, num_users).to(config.DEVICE)
    model.eval()

    with torch.no_grad():
        # Forward pass
        x_recon, mu, logvar = model(sequences, conditions)

        # Compute loss
        total_loss, recon_loss, kl_loss = vae_loss(x_recon, sequences, mu, logvar, kl_weight=0.0)

        print("\n" + "="*60)
        print("Loss on REAL training data (untrained model):")
        print("="*60)
        print(f"Recon Loss: {recon_loss.item():.4f}")
        print(f"KL Loss: {kl_loss.item():.4f}")

        # Compute per-feature MSE
        squared_error = (x_recon - sequences) ** 2
        mse_per_feature = squared_error.mean(dim=[0, 1])  # Average over batch and seq_len

        print("\n" + "="*60)
        print("Per-feature MSE on real data:")
        print("="*60)
        feature_names = ['delta_t', 'delta_x', 'delta_y', 'speed', 'accel', 'button', 'state']
        weights = [0.01, 500.0, 500.0, 0.2, 0.2, 1.0, 1.0]

        total_weighted = 0
        for i, (name, w) in enumerate(zip(feature_names, weights)):
            mse = mse_per_feature[i].item()
            weighted = w * mse
            total_weighted += weighted
            print(f"  {name:10s}: MSE={mse:.6f}, weight={w:6.2f}, contribution={weighted:.6f}")

        avg_weighted = total_weighted / 7
        print(f"\nTotal weighted: {total_weighted:.4f}")
        print(f"Average (÷7):   {avg_weighted:.4f}")
        print(f"Actual recon_loss: {recon_loss.item():.4f}")
        print(f"Match: {'✓' if abs(avg_weighted - recon_loss.item()) < 0.01 else '✗'}")

        # Compare with training loss
        print("\n" + "="*60)
        print("Comparison:")
        print("="*60)
        print(f"Expected training Epoch 0 recon loss: ~{recon_loss.item():.4f}")
        print(f"If training shows 0.029, then MSE values must be ~10× smaller")
        print(f"This would mean model is learning VERY fast in first batch")

    break  # Only test first batch

print("\n" + "="*60)
print("To verify: Check if training Epoch 0 matches this value")
print("="*60)
