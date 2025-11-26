"""
使用 DataLoader 分析数据集中的 action type 分布
"""

import torch
import numpy as np
from collections import Counter
import argparse
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders


def analyze_with_dataloader(config):
    """
    使用 DataLoader 分析数据集

    Args:
        config: Config 对象
    """
    print("="*60)
    print("使用 DataLoader 分析 Action Type 分布")
    print("="*60)
    print(f"数据集路径: {config.DATA_DIR}")
    print(f"序列长度: {config.SEQUENCE_LENGTH}")
    print(f"过滤窗口: {config.FILTER_WINDOW}")
    print(f"批次大小: {config.BATCH_SIZE}")
    print()

    # 加载数据
    print("加载数据集...")
    print(f"使用 {config.NUM_WORKERS} 个并行 workers 进行数据预处理...")
    train_loader, val_loader, num_users = get_dataloaders(config)

    print(f"✓ 数据加载完成")
    print(f"  用户数: {num_users}")
    print(f"  训练集大小: {len(train_loader.dataset):,} 个序列")
    print(f"  验证集大小: {len(val_loader.dataset):,} 个序列")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  验证批次数: {len(val_loader)}")
    print()

    # 分析训练集
    print("分析训练集...")
    train_stats = analyze_loader(train_loader, "训练集")

    # 分析验证集
    print("\n分析验证集...")
    val_stats = analyze_loader(val_loader, "验证集")

    # 合并统计
    print("\n" + "="*60)
    print("总体统计（训练集 + 验证集）")
    print("="*60)

    total_stats = merge_stats(train_stats, val_stats)
    print_statistics(total_stats, "总体")

    # 对比训练集和验证集
    print("\n" + "="*60)
    print("训练集 vs 验证集对比")
    print("="*60)
    compare_stats(train_stats, val_stats)


def analyze_loader(dataloader, name):
    """
    分析单个 DataLoader

    Args:
        dataloader: DataLoader 对象
        name: 数据集名称

    Returns:
        stats: 统计信息字典
    """
    action_counter = Counter()
    user_counter = Counter()
    time_period_counter = Counter()

    # 用于统计特征
    all_features = {
        'delta_t': [],
        'delta_x': [],
        'delta_y': [],
        'speed': [],
        'accel': [],
        'button': [],
        'state': []
    }

    total_sequences = 0

    for sequences, conditions in tqdm(dataloader, desc=f"分析{name}"):
        batch_size = sequences.size(0)
        total_sequences += batch_size

        # 统计 action types
        action_types = conditions['action_type'].cpu().numpy().flatten()
        for action in action_types:
            action_counter[action] += 1

        # 统计用户
        user_ids = conditions['user_id'].cpu().numpy().flatten()
        for user in user_ids:
            user_counter[user] += 1

        # 统计时间段
        time_periods = conditions['time_period'].cpu().numpy().flatten()
        for period in time_periods:
            time_period_counter[period] += 1

        # 收集特征统计（只采样部分数据，避免内存溢出）
        if len(all_features['delta_t']) < 100000:  # 只收集前100k个序列
            sequences_np = sequences.cpu().numpy()
            for i in range(min(batch_size, 100)):  # 每个batch只采样100个
                seq = sequences_np[i]
                all_features['delta_t'].extend(seq[:, 0])
                all_features['delta_x'].extend(seq[:, 1])
                all_features['delta_y'].extend(seq[:, 2])
                all_features['speed'].extend(seq[:, 3])
                all_features['accel'].extend(seq[:, 4])
                all_features['button'].extend(seq[:, 5])
                all_features['state'].extend(seq[:, 6])

    # 转换为 numpy 数组
    for key in all_features:
        all_features[key] = np.array(all_features[key])

    stats = {
        'total_sequences': total_sequences,
        'action_counter': action_counter,
        'user_counter': user_counter,
        'time_period_counter': time_period_counter,
        'features': all_features
    }

    return stats


def merge_stats(stats1, stats2):
    """合并两个统计结果"""
    merged = {
        'total_sequences': stats1['total_sequences'] + stats2['total_sequences'],
        'action_counter': stats1['action_counter'] + stats2['action_counter'],
        'user_counter': stats1['user_counter'] + stats2['user_counter'],
        'time_period_counter': stats1['time_period_counter'] + stats2['time_period_counter'],
        'features': {}
    }

    # 合并特征（只取第一个的特征统计）
    merged['features'] = stats1['features']

    return merged


def print_statistics(stats, name):
    """打印统计信息"""
    print(f"\n{name}:")
    print(f"  总序列数: {stats['total_sequences']:,}")

    # Action Type 分布
    print(f"\n  {'='*56}")
    print(f"  Action Type 分布:")
    print(f"  {'='*56}")

    action_names = {0: 'Move', 1: 'Click', 2: 'Drag'}
    total_actions = sum(stats['action_counter'].values())

    for action_id in sorted(stats['action_counter'].keys()):
        count = stats['action_counter'][action_id]
        percentage = (count / total_actions) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"  {action_id}: {action_names[action_id]:<10} {count:>8,} ({percentage:>5.2f}%) {bar}")

    # 时间段分布
    print(f"\n  {'='*56}")
    print(f"  时间段分布:")
    print(f"  {'='*56}")

    period_names = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night'}
    total_periods = sum(stats['time_period_counter'].values())

    for period_id in sorted(stats['time_period_counter'].keys()):
        count = stats['time_period_counter'][period_id]
        percentage = (count / total_periods) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"  {period_id}: {period_names[period_id]:<12} {count:>8,} ({percentage:>5.2f}%) {bar}")

    # 用户分布
    print(f"\n  {'='*56}")
    print(f"  用户分布:")
    print(f"  {'='*56}")
    print(f"  总用户数: {len(stats['user_counter'])}")
    print(f"  最活跃用户:")

    for user_id, count in stats['user_counter'].most_common(5):
        percentage = (count / stats['total_sequences']) * 100
        print(f"    User {user_id}: {count:>8,} 序列 ({percentage:>5.2f}%)")

    # 特征统计
    print(f"\n  {'='*56}")
    print(f"  特征统计 (基于采样):")
    print(f"  {'='*56}")

    feature_names = ['delta_t', 'delta_x', 'delta_y', 'speed', 'accel', 'button', 'state']
    print(f"  {'Feature':<12} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10} {'Variance':>12}")
    print(f"  {'-'*70}")

    for fname in feature_names:
        data = stats['features'][fname]
        if len(data) > 0:
            print(f"  {fname:<12} {data.min():>10.4f} {data.max():>10.4f} "
                  f"{data.mean():>10.4f} {data.std():>10.4f} {data.var():>12.6f}")


def compare_stats(train_stats, val_stats):
    """对比训练集和验证集"""
    print(f"\nAction Type 分布对比:")
    print(f"  {'Action':<15} {'训练集':>15} {'验证集':>15} {'差异':>10}")
    print(f"  {'-'*58}")

    action_names = {0: 'Move', 1: 'Click', 2: 'Drag'}
    train_total = sum(train_stats['action_counter'].values())
    val_total = sum(val_stats['action_counter'].values())

    for action_id in sorted(train_stats['action_counter'].keys()):
        train_count = train_stats['action_counter'][action_id]
        val_count = val_stats['action_counter'][action_id]

        train_pct = (train_count / train_total) * 100
        val_pct = (val_count / val_total) * 100
        diff = val_pct - train_pct

        print(f"  {action_names[action_id]:<15} {train_pct:>14.2f}% {val_pct:>14.2f}% {diff:>9.2f}%")

    print(f"\n时间段分布对比:")
    print(f"  {'Period':<15} {'训练集':>15} {'验证集':>15} {'差异':>10}")
    print(f"  {'-'*58}")

    period_names = {0: 'Morning', 1: 'Afternoon', 2: 'Evening', 3: 'Night'}
    train_total = sum(train_stats['time_period_counter'].values())
    val_total = sum(val_stats['time_period_counter'].values())

    for period_id in sorted(train_stats['time_period_counter'].keys()):
        train_count = train_stats['time_period_counter'][period_id]
        val_count = val_stats['time_period_counter'][period_id]

        train_pct = (train_count / train_total) * 100
        val_pct = (val_count / val_total) * 100
        diff = val_pct - train_pct

        print(f"  {period_names[period_id]:<15} {train_pct:>14.2f}% {val_pct:>14.2f}% {diff:>9.2f}%")


def main():
    parser = argparse.ArgumentParser(description='使用 DataLoader 分析数据集')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='数据集目录（覆盖 config.py 中的设置）')
    parser.add_argument('--filter-window', type=str, default=None,
                       help='过滤窗口（覆盖 config.py 中的设置）')
    parser.add_argument('--max-sessions', type=int, default=None,
                       help='每个用户最大 session 数（用于快速测试）')

    args = parser.parse_args()

    # 加载配置
    config = Config()

    # 覆盖配置
    if args.data_dir:
        config.DATA_DIR = args.data_dir
    if args.filter_window:
        config.FILTER_WINDOW = args.filter_window
    if args.max_sessions:
        config.MAX_SESSIONS_PER_USER = args.max_sessions

    # 分析
    analyze_with_dataloader(config)

    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断分析")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
