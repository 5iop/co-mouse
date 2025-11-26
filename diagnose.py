"""
诊断脚本：检查特征分布和模型输出
"""

import torch
import numpy as np
from dataset import get_dataloaders
from model import ConditionalLSTMVAE
from config import Config

def check_feature_distribution(train_loader):
    """检查特征分布"""
    print("="*60)
    print("1. 检查特征分布")
    print("="*60)

    # 收集数据
    all_features = []
    for i, (sequences, _) in enumerate(train_loader):
        all_features.append(sequences.numpy())
        if i >= 10:  # 只看前10个batch
            break

    all_features = np.concatenate(all_features, axis=0)

    feature_names = ['delta_t', 'delta_x', 'delta_y', 'speed', 'accel', 'button', 'state']

    print(f"\n{'Feature':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Range':>10}")
    print("-"*70)

    for i, name in enumerate(feature_names):
        data = all_features[:, :, i]
        min_val = data.min()
        max_val = data.max()
        mean_val = data.mean()
        std_val = data.std()
        range_val = max_val - min_val

        print(f"{name:<12} {min_val:>10.4f} {max_val:>10.4f} {mean_val:>10.4f} {std_val:>10.4f} {range_val:>10.4f}")

    # 检查是否有异常值
    print("\n特征尺度分析:")
    for i, name in enumerate(feature_names):
        data = all_features[:, :, i]
        variance = np.var(data)
        print(f"  {name:<12} variance = {variance:.6f}")

    return all_features


def check_model_output(train_loader, num_users, config):
    """检查模型初始输出"""
    print("\n" + "="*60)
    print("2. 检查模型初始状态")
    print("="*60)

    # 创建模型
    model = ConditionalLSTMVAE(config, num_users).to(config.DEVICE)
    model.eval()

    # 获取一个batch
    sequences, conditions = next(iter(train_loader))
    sequences = sequences.to(config.DEVICE)
    for key in conditions:
        conditions[key] = conditions[key].to(config.DEVICE)

    with torch.no_grad():
        # 前向传播
        x_recon, mu, logvar = model(sequences, conditions)

        # 检查latent分布
        print(f"\nLatent分布统计:")
        print(f"  mu    - mean: {mu.mean().item():>10.4f}, std: {mu.std().item():>10.4f}")
        print(f"  mu    - min:  {mu.min().item():>10.4f}, max: {mu.max().item():>10.4f}")
        print(f"  logvar - mean: {logvar.mean().item():>10.4f}, std: {logvar.std().item():>10.4f}")
        print(f"  logvar - min:  {logvar.min().item():>10.4f}, max: {logvar.max().item():>10.4f}")

        # 计算KL散度
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_per_sample = kl_div / sequences.size(0)
        print(f"\nKL散度:")
        print(f"  Total KL: {kl_div.item():.4f}")
        print(f"  KL per sample: {kl_per_sample.item():.4f}")

        # 检查重建误差
        recon_error = torch.mean((x_recon - sequences) ** 2, dim=[1, 2])
        print(f"\n重建误差 (per sample):")
        print(f"  Mean: {recon_error.mean().item():.4f}")
        print(f"  Min:  {recon_error.min().item():.4f}")
        print(f"  Max:  {recon_error.max().item():.4f}")

        # 按特征分解重建误差
        print(f"\n每个特征的重建误差:")
        feature_names = ['delta_t', 'delta_x', 'delta_y', 'speed', 'accel', 'button', 'state']
        for i, name in enumerate(feature_names):
            error = torch.mean((x_recon[:, :, i] - sequences[:, :, i]) ** 2)
            print(f"  {name:<12} MSE: {error.item():.6f}")


def check_data_sanity(train_loader, val_loader, num_users):
    """检查数据完整性"""
    print("\n" + "="*60)
    print("3. 检查数据完整性")
    print("="*60)

    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"用户数量: {num_users}")
    print(f"Batch大小: {train_loader.batch_size}")
    print(f"总batch数: {len(train_loader)}")

    # 检查条件分布
    all_user_ids = []
    all_time_periods = []
    all_action_types = []

    for i, (_, conditions) in enumerate(train_loader):
        all_user_ids.append(conditions['user_id'].numpy())
        all_time_periods.append(conditions['time_period'].numpy())
        all_action_types.append(conditions['action_type'].numpy())
        if i >= 100:
            break

    all_user_ids = np.concatenate(all_user_ids)
    all_time_periods = np.concatenate(all_time_periods)
    all_action_types = np.concatenate(all_action_types)

    print(f"\n条件分布:")
    print(f"  User IDs: min={all_user_ids.min()}, max={all_user_ids.max()}, unique={len(np.unique(all_user_ids))}")
    print(f"  Time periods: {np.bincount(all_time_periods)}")
    print(f"  Action types: {np.bincount(all_action_types)}")


if __name__ == "__main__":
    print("\n开始诊断...\n")

    # 加载数据（只加载一次）
    print("加载数据...")
    config = Config()
    train_loader, val_loader, num_users = get_dataloaders(config)
    print("✓ 数据加载完成\n")

    # 1. 检查特征分布
    features = check_feature_distribution(train_loader)

    # 2. 检查模型输出
    check_model_output(train_loader, num_users, config)

    # 3. 检查数据完整性
    check_data_sanity(train_loader, val_loader, num_users)

    print("\n" + "="*60)
    print("诊断完成!")
    print("="*60)
