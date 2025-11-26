"""
检测BOUN数据集中每个用户的屏幕分辨率
通过查找每个用户CSV文件中的最大x和y坐标值
并分析用户操作集中在屏幕的哪些区域
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict


def process_file(csv_file):
    """
    处理单个CSV文件，返回最大和最小x和y值，以及所有坐标点

    Args:
        csv_file: CSV文件路径

    Returns:
        (user_id, test_type, min_x, min_y, max_x, max_y, x_coords, y_coords) 或 None（如果处理失败）
    """
    try:
        # 从路径中提取user_id和test_type
        # 路径格式: .../users/user1/internal_tests/session_001.csv
        parts = Path(csv_file).parts
        user_id = None
        test_type = None

        for i, part in enumerate(parts):
            if part == 'users' and i + 1 < len(parts):
                user_id = parts[i + 1]
                if i + 2 < len(parts):
                    test_type = parts[i + 2]
                break

        if user_id is None or test_type is None:
            return None

        # 读取CSV
        df = pd.read_csv(csv_file)

        if df.empty or 'x' not in df.columns or 'y' not in df.columns:
            return None

        min_x = df['x'].min()
        min_y = df['y'].min()
        max_x = df['x'].max()
        max_y = df['y'].max()

        # 获取所有坐标点用于热力图分析
        x_coords = df['x'].values
        y_coords = df['y'].values

        return (user_id, test_type, min_x, min_y, max_x, max_y, x_coords, y_coords)

    except Exception as e:
        return None


def analyze_screen_regions(x_coords, y_coords, max_x, max_y, grid_size=10):
    """
    分析用户操作集中在屏幕的哪些区域

    Args:
        x_coords: 所有x坐标列表
        y_coords: 所有y坐标列表
        max_x: 屏幕最大x坐标
        max_y: 屏幕最大y坐标
        grid_size: 网格大小(将屏幕分为grid_size x grid_size个区域)

    Returns:
        heatmap: 网格热力图矩阵
        region_stats: 区域统计信息
    """
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None, None

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)

    # 创建网格
    heatmap = np.zeros((grid_size, grid_size))

    # 计算每个点所在的网格位置
    x_bins = np.linspace(0, max_x, grid_size + 1)
    y_bins = np.linspace(0, max_y, grid_size + 1)

    x_indices = np.digitize(x_coords, x_bins) - 1
    y_indices = np.digitize(y_coords, y_bins) - 1

    # 确保索引在有效范围内
    x_indices = np.clip(x_indices, 0, grid_size - 1)
    y_indices = np.clip(y_indices, 0, grid_size - 1)

    # 统计每个网格的点数
    for x_idx, y_idx in zip(x_indices, y_indices):
        heatmap[y_idx, x_idx] += 1

    # 计算统计信息
    total_points = len(x_coords)
    heatmap_percent = (heatmap / total_points) * 100

    # 找到热点区域(超过平均值的区域)
    avg_percent = 100.0 / (grid_size * grid_size)
    hot_regions = []

    for i in range(grid_size):
        for j in range(grid_size):
            if heatmap_percent[i, j] > avg_percent * 1.5:  # 超过平均值1.5倍
                x_start = int(x_bins[j])
                x_end = int(x_bins[j + 1])
                y_start = int(y_bins[i])
                y_end = int(y_bins[i + 1])
                hot_regions.append({
                    'grid_pos': (i, j),
                    'x_range': (x_start, x_end),
                    'y_range': (y_start, y_end),
                    'percent': heatmap_percent[i, j],
                    'count': int(heatmap[i, j])
                })

    # 按百分比排序
    hot_regions.sort(key=lambda x: x['percent'], reverse=True)

    region_stats = {
        'total_points': total_points,
        'avg_percent_per_grid': avg_percent,
        'hot_regions': hot_regions,
        'heatmap_percent': heatmap_percent
    }

    return heatmap, region_stats


def detect_resolutions(data_dir, num_workers=16, grid_size=10, analyze_regions=True):
    """
    检测所有用户的屏幕分辨率并分析操作集中区域

    Args:
        data_dir: 数据集根目录 (boun-mouse-dynamics-dataset/users)
        num_workers: 并行worker数量
        grid_size: 网格大小(将屏幕分为grid_size x grid_size个区域)
        analyze_regions: 是否分析操作集中区域
    """
    data_path = Path(data_dir)

    print("="*70)
    print("BOUN Mouse Dynamics Dataset - Screen Resolution Detection")
    print("="*70)
    print(f"Data directory: {data_path}")
    print(f"Workers: {num_workers}\n")

    # 收集所有CSV文件
    print("Scanning for CSV files...")
    all_files = []

    for user_dir in sorted(data_path.iterdir()):
        if not user_dir.is_dir():
            continue

        for test_type in ['training', 'internal_tests', 'external_tests']:
            test_dir = user_dir / test_type
            if not test_dir.exists():
                continue

            csv_files = list(test_dir.glob('*.csv'))
            all_files.extend(csv_files)

    print(f"Found {len(all_files):,} CSV files\n")

    # 使用多进程处理所有文件
    print("Processing files...")
    user_resolutions = defaultdict(lambda: {'max_x': 0, 'max_y': 0, 'file_count': 0})

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_file, all_files),
            total=len(all_files),
            desc="Analyzing files"
        ))

    # 汇总每个用户和测试类型的最大和最小分辨率
    # 结构: user_resolutions[user_id][test_type] = {'min_x': ..., 'min_y': ..., 'max_x': ..., 'max_y': ..., 'file_count': ..., 'x_coords': [], 'y_coords': []}
    user_resolutions = defaultdict(lambda: defaultdict(lambda: {
        'min_x': float('inf'), 'min_y': float('inf'),
        'max_x': 0, 'max_y': 0, 'file_count': 0,
        'x_coords': [], 'y_coords': []
    }))

    for result in results:
        if result is None:
            continue

        user_id, test_type, min_x, min_y, max_x, max_y, x_coords, y_coords = result
        user_resolutions[user_id][test_type]['min_x'] = min(user_resolutions[user_id][test_type]['min_x'], min_x)
        user_resolutions[user_id][test_type]['min_y'] = min(user_resolutions[user_id][test_type]['min_y'], min_y)
        user_resolutions[user_id][test_type]['max_x'] = max(user_resolutions[user_id][test_type]['max_x'], max_x)
        user_resolutions[user_id][test_type]['max_y'] = max(user_resolutions[user_id][test_type]['max_y'], max_y)
        user_resolutions[user_id][test_type]['file_count'] += 1
        # 收集所有坐标点
        user_resolutions[user_id][test_type]['x_coords'].extend(x_coords)
        user_resolutions[user_id][test_type]['y_coords'].extend(y_coords)

    # 打印 internal_tests 结果
    print("\n" + "="*105)
    print("INTERNAL_TESTS Results")
    print("="*105)
    print(f"{'User ID':<12} {'Min X':<10} {'Min Y':<10} {'Max X':<10} {'Max Y':<10} {'Delta X':<10} {'Delta Y':<10} {'Files':<10}")
    print("-"*105)

    internal_min_x = float('inf')
    internal_min_y = float('inf')
    internal_max_x = 0
    internal_max_y = 0

    for user_id in sorted(user_resolutions.keys()):
        if 'internal_tests' in user_resolutions[user_id]:
            data = user_resolutions[user_id]['internal_tests']
            min_x = int(data['min_x'])
            min_y = int(data['min_y'])
            max_x = int(data['max_x'])
            max_y = int(data['max_y'])
            delta_x = max_x - min_x
            delta_y = max_y - min_y
            file_count = data['file_count']

            print(f"{user_id:<12} {min_x:<10} {min_y:<10} {max_x:<10} {max_y:<10} {delta_x:<10} {delta_y:<10} {file_count:<10}")

            internal_min_x = min(internal_min_x, min_x)
            internal_min_y = min(internal_min_y, min_y)
            internal_max_x = max(internal_max_x, max_x)
            internal_max_y = max(internal_max_y, max_y)

    internal_delta_x = internal_max_x - internal_min_x
    internal_delta_y = internal_max_y - internal_min_y
    print("-"*105)
    print(f"{'INTERNAL':<12} {internal_min_x:<10} {internal_min_y:<10} {internal_max_x:<10} {internal_max_y:<10} {internal_delta_x:<10} {internal_delta_y:<10}")
    print("="*105)

    # 打印 external_tests 结果
    print("\n" + "="*105)
    print("EXTERNAL_TESTS Results")
    print("="*105)
    print(f"{'User ID':<12} {'Min X':<10} {'Min Y':<10} {'Max X':<10} {'Max Y':<10} {'Delta X':<10} {'Delta Y':<10} {'Files':<10}")
    print("-"*105)

    external_min_x = float('inf')
    external_min_y = float('inf')
    external_max_x = 0
    external_max_y = 0

    for user_id in sorted(user_resolutions.keys()):
        if 'external_tests' in user_resolutions[user_id]:
            data = user_resolutions[user_id]['external_tests']
            min_x = int(data['min_x'])
            min_y = int(data['min_y'])
            max_x = int(data['max_x'])
            max_y = int(data['max_y'])
            delta_x = max_x - min_x
            delta_y = max_y - min_y
            file_count = data['file_count']

            print(f"{user_id:<12} {min_x:<10} {min_y:<10} {max_x:<10} {max_y:<10} {delta_x:<10} {delta_y:<10} {file_count:<10}")

            external_min_x = min(external_min_x, min_x)
            external_min_y = min(external_min_y, min_y)
            external_max_x = max(external_max_x, max_x)
            external_max_y = max(external_max_y, max_y)

    external_delta_x = external_max_x - external_min_x
    external_delta_y = external_max_y - external_min_y
    print("-"*105)
    print(f"{'EXTERNAL':<12} {external_min_x:<10} {external_min_y:<10} {external_max_x:<10} {external_max_y:<10} {external_delta_x:<10} {external_delta_y:<10}")
    print("="*105)

    # 打印 training 结果（如果有）
    has_training = any('training' in user_resolutions[uid] for uid in user_resolutions.keys())
    if has_training:
        print("\n" + "="*85)
        print("TRAINING Results")
        print("="*85)
        print(f"{'User ID':<12} {'Min X':<10} {'Min Y':<10} {'Max X':<10} {'Max Y':<10} {'Files':<10}")
        print("-"*85)

        training_min_x = float('inf')
        training_min_y = float('inf')
        training_max_x = 0
        training_max_y = 0

        for user_id in sorted(user_resolutions.keys()):
            if 'training' in user_resolutions[user_id]:
                data = user_resolutions[user_id]['training']
                min_x = int(data['min_x'])
                min_y = int(data['min_y'])
                max_x = int(data['max_x'])
                max_y = int(data['max_y'])
                file_count = data['file_count']

                print(f"{user_id:<12} {min_x:<10} {min_y:<10} {max_x:<10} {max_y:<10} {file_count:<10}")

                training_min_x = min(training_min_x, min_x)
                training_min_y = min(training_min_y, min_y)
                training_max_x = max(training_max_x, max_x)
                training_max_y = max(training_max_y, max_y)

        print("-"*85)
        print(f"{'TRAINING':<12} {training_min_x:<10} {training_min_y:<10} {training_max_x:<10} {training_max_y:<10}")
        print("="*85)

    # 打印整体统计
    all_min_x = min(internal_min_x, external_min_x)
    all_min_y = min(internal_min_y, external_min_y)
    all_max_x = max(internal_max_x, external_max_x)
    all_max_y = max(internal_max_y, external_max_y)

    print("\n" + "="*85)
    print("OVERALL Summary")
    print("="*85)
    print(f"Internal Tests: Min ({internal_min_x}, {internal_min_y})  Max ({internal_max_x}, {internal_max_y})")
    print(f"External Tests: Min ({external_min_x}, {external_min_y})  Max ({external_max_x}, {external_max_y})")
    print(f"Overall:        Min ({all_min_x}, {all_min_y})  Max ({all_max_x}, {all_max_y})")
    print("="*85)

    print("\n" + "="*85)
    print("Recommendation for config.py:")
    print("-"*85)
    print(f"# Use the maximum observed resolution across all test types")
    print(f"SCREEN_WIDTH = {all_max_x}")
    print(f"SCREEN_HEIGHT = {all_max_y}")
    print("-"*85)
    print(f"# Minimum observed coordinates (usually close to 0)")
    print(f"# Min X = {all_min_x}, Min Y = {all_min_y}")
    print("="*85)

    # 区域分析
    if analyze_regions:
        print("\n" + "="*85)
        print("SCREEN REGION ANALYSIS")
        print("="*85)
        print(f"Grid Size: {grid_size}x{grid_size}")
        print(f"Screen divided into {grid_size * grid_size} regions\n")

        # 分析internal_tests的整体区域分布
        print("-"*85)
        print("INTERNAL TESTS - Overall Region Distribution")
        print("-"*85)

        all_x = []
        all_y = []
        for user_id in user_resolutions.keys():
            if 'internal_tests' in user_resolutions[user_id]:
                all_x.extend(user_resolutions[user_id]['internal_tests']['x_coords'])
                all_y.extend(user_resolutions[user_id]['internal_tests']['y_coords'])

        if len(all_x) > 0:
            _, stats = analyze_screen_regions(all_x, all_y, internal_max_x, internal_max_y, grid_size)
            if stats:
                print(f"Total data points analyzed: {stats['total_points']:,}")
                print(f"Average points per grid: {stats['total_points'] / (grid_size * grid_size):.0f} ({stats['avg_percent_per_grid']:.2f}%)")
                print(f"\nTop 10 hottest regions (concentrated activity areas):")
                print(f"{'Rank':<6} {'Grid(Y,X)':<12} {'X Range':<20} {'Y Range':<20} {'Points':<12} {'Percent':<10}")
                print("-"*85)

                for idx, region in enumerate(stats['hot_regions'][:10], 1):
                    grid_y, grid_x = region['grid_pos']
                    x_range = f"{region['x_range'][0]}-{region['x_range'][1]}"
                    y_range = f"{region['y_range'][0]}-{region['y_range'][1]}"
                    print(f"{idx:<6} ({grid_y},{grid_x}){'':<7} {x_range:<20} {y_range:<20} {region['count']:<12,} {region['percent']:.2f}%")

                # 打印ASCII热力图
                print("\n" + "-"*85)
                print("Activity Heatmap (ASCII visualization, higher numbers = more activity):")
                print("-"*85)
                heatmap_percent = stats['heatmap_percent']

                # 将百分比映射到字符
                def percent_to_char(percent):
                    if percent == 0:
                        return '·'
                    elif percent < 0.5:
                        return '░'
                    elif percent < 1.0:
                        return '▒'
                    elif percent < 2.0:
                        return '▓'
                    else:
                        return '█'

                print("   ", end="")
                for j in range(grid_size):
                    print(f"{j:>3}", end="")
                print()

                for i in range(grid_size):
                    print(f"{i:>2} ", end="")
                    for j in range(grid_size):
                        char = percent_to_char(heatmap_percent[i, j])
                        print(f"  {char}", end="")
                    print()

                print("\nLegend: · = 0%  ░ = <0.5%  ▒ = 0.5-1%  ▓ = 1-2%  █ = >2%")

        # 分析external_tests的整体区域分布
        print("\n" + "-"*85)
        print("EXTERNAL TESTS - Overall Region Distribution")
        print("-"*85)

        all_x = []
        all_y = []
        for user_id in user_resolutions.keys():
            if 'external_tests' in user_resolutions[user_id]:
                all_x.extend(user_resolutions[user_id]['external_tests']['x_coords'])
                all_y.extend(user_resolutions[user_id]['external_tests']['y_coords'])

        if len(all_x) > 0:
            _, stats = analyze_screen_regions(all_x, all_y, external_max_x, external_max_y, grid_size)
            if stats:
                print(f"Total data points analyzed: {stats['total_points']:,}")
                print(f"Average points per grid: {stats['total_points'] / (grid_size * grid_size):.0f} ({stats['avg_percent_per_grid']:.2f}%)")
                print(f"\nTop 10 hottest regions (concentrated activity areas):")
                print(f"{'Rank':<6} {'Grid(Y,X)':<12} {'X Range':<20} {'Y Range':<20} {'Points':<12} {'Percent':<10}")
                print("-"*85)

                for idx, region in enumerate(stats['hot_regions'][:10], 1):
                    grid_y, grid_x = region['grid_pos']
                    x_range = f"{region['x_range'][0]}-{region['x_range'][1]}"
                    y_range = f"{region['y_range'][0]}-{region['y_range'][1]}"
                    print(f"{idx:<6} ({grid_y},{grid_x}){'':<7} {x_range:<20} {y_range:<20} {region['count']:<12,} {region['percent']:.2f}%")

                # 打印ASCII热力图
                print("\n" + "-"*85)
                print("Activity Heatmap (ASCII visualization, higher numbers = more activity):")
                print("-"*85)
                heatmap_percent = stats['heatmap_percent']

                print("   ", end="")
                for j in range(grid_size):
                    print(f"{j:>3}", end="")
                print()

                for i in range(grid_size):
                    print(f"{i:>2} ", end="")
                    for j in range(grid_size):
                        char = percent_to_char(heatmap_percent[i, j])
                        print(f"  {char}", end="")
                    print()

                print("\nLegend: · = 0%  ░ = <0.5%  ▒ = 0.5-1%  ▓ = 1-2%  █ = >2%")

        print("="*85)

    return user_resolutions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Detect screen resolutions and analyze activity regions in BOUN dataset')
    parser.add_argument('--data-dir', type=str, default='boun-mouse-dynamics-dataset/users',
                       help='Path to dataset users directory')
    parser.add_argument('--workers', type=int, default=16,
                       help='Number of parallel workers')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size for region analysis (default: 10x10)')
    parser.add_argument('--no-region-analysis', action='store_true',
                       help='Disable region analysis')
    args = parser.parse_args()

    detect_resolutions(
        args.data_dir,
        num_workers=args.workers,
        grid_size=args.grid_size,
        analyze_regions=not args.no_region_analysis
    )
