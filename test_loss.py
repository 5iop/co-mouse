"""
Test script to verify feature weighting in vae_loss function
"""
import torch
from model import vae_loss
import numpy as np

# 创建测试数据（模拟diagnose.py的数值）
batch_size = 128
seq_len = 100
features = 7

# 创建假的输入
x = torch.randn(batch_size, seq_len, features)
x_recon = x.clone()

# 手动设置每个特征的MSE为diagnose.py看到的值
with torch.no_grad():
    # delta_t (feature 0): MSE = 0.0101
    x_recon[:, :, 0] = x[:, :, 0] + torch.randn_like(x[:, :, 0]) * np.sqrt(0.0101)
    # delta_x (feature 1): MSE = 0.002
    x_recon[:, :, 1] = x[:, :, 1] + torch.randn_like(x[:, :, 1]) * np.sqrt(0.002)
    # delta_y (feature 2): MSE = 0.0013
    x_recon[:, :, 2] = x[:, :, 2] + torch.randn_like(x[:, :, 2]) * np.sqrt(0.0013)
    # speed (feature 3): MSE = 0.844
    x_recon[:, :, 3] = x[:, :, 3] + torch.randn_like(x[:, :, 3]) * np.sqrt(0.844)
    # accel (feature 4): MSE = 0.285
    x_recon[:, :, 4] = x[:, :, 4] + torch.randn_like(x[:, :, 4]) * np.sqrt(0.285)
    # button (feature 5): MSE = 0.007
    x_recon[:, :, 5] = x[:, :, 5] + torch.randn_like(x[:, :, 5]) * np.sqrt(0.007)
    # state (feature 6): MSE = 0.012
    x_recon[:, :, 6] = x[:, :, 6] + torch.randn_like(x[:, :, 6]) * np.sqrt(0.012)

mu = torch.randn(batch_size, 64)
logvar = torch.randn(batch_size, 64)

# 计算loss
total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar, kl_weight=0.0)

print("="*60)
print("Testing vae_loss with feature weighting")
print("="*60)
print(f"\nRecon Loss: {recon_loss.item():.4f}")
print(f"\nExpected values:")
print(f"  With 500× weights on delta_x/delta_y: ~1.5-2.0")
print(f"  WITHOUT weighting (old version):       ~0.02-0.03")
print(f"\nResult:")

if recon_loss.item() > 1.0:
    print(f"  ✓ PASS - Feature weights ARE being applied correctly")
elif recon_loss.item() < 0.1:
    print(f"  ✗ FAIL - Feature weights are NOT applied (using old code)")
else:
    print(f"  ? UNCERTAIN - Loss in unexpected range")

# 验证per-feature MSE
mse_per_feature = ((x_recon - x) ** 2).mean(dim=[0, 1])
print("\n" + "="*60)
print("Per-feature MSE verification:")
print("="*60)
feature_names = ['delta_t', 'delta_x', 'delta_y', 'speed', 'accel', 'button', 'state']
for i, name in enumerate(feature_names):
    print(f"  {name:10s}: {mse_per_feature[i].item():.6f}")

print("\n" + "="*60)
print("Expected weighted contribution to loss:")
print("="*60)
weights = [0.01, 500.0, 500.0, 0.2, 0.2, 1.0, 1.0]
contributions = []
for i, (name, w) in enumerate(zip(feature_names, weights)):
    contrib = w * mse_per_feature[i].item()
    contributions.append(contrib)
    print(f"  {name:10s}: {w:6.2f} × {mse_per_feature[i].item():.6f} = {contrib:.4f}")

print(f"\nTotal expected: {sum(contributions):.4f}")
print(f"Actual recon_loss: {recon_loss.item():.4f}")
print(f"Match: {'✓ YES' if abs(sum(contributions) - recon_loss.item()) < 0.1 else '✗ NO'}")
print("="*60)
