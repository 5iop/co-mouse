"""
数据清洗脚本：按Release切分，过滤异常片段（多线程版本）
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil

# Minimum sequence length (must match Config.SEQUENCE_LENGTH in config.py)
MIN_SEQUENCE_LENGTH = 100

# Screen resolution bounds
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

def clean_session_file(args):
    """
    清洗单个session文件（全局函数用于multiprocessing）

    Args:
        args: (input_file, output_file) 元组

    Returns:
        stats: 统计信息字典
    """
    input_file, output_file = args

    try:
        df = pd.read_csv(input_file)

        if df.empty or len(df) < 2:
            return {'status': 'empty', 'segments_kept': 0, 'segments_dropped': 0, 'points_removed': 0}

        original_event_count = len(df)

        # Filter by window="browsing" (must match Config.FILTER_WINDOW)
        df = df[df['window'] == 'browsing'].reset_index(drop=True)

        # Filter out-of-bounds coordinates (x not in [0, 1920], y not in [0, 1080])
        before_coord_filter = len(df)
        df = df[(df['x'] >= 0) & (df['x'] <= SCREEN_WIDTH) &
                (df['y'] >= 0) & (df['y'] <= SCREEN_HEIGHT)].reset_index(drop=True)
        points_removed = before_coord_filter - len(df)

        # 如果过滤后完全没有数据，直接返回
        if df.empty:
            return {'status': 'empty', 'segments_kept': 0, 'segments_dropped': 0, 'points_removed': points_removed}

        # 计算时间间隔
        df['delta_t'] = df['client_timestamp'].diff().fillna(0)

        # 找到Release事件作为切分点
        release_indices = df[df['state'] == 'Released'].index.tolist()

        # 添加起始和结束位置
        segments_bounds = [0] + [idx + 1 for idx in release_indices]
        if segments_bounds[-1] < len(df):
            segments_bounds.append(len(df))

        cleaned_segments = []
        stats = {'segments_kept': 0, 'segments_dropped': 0}

        # 处理每个片段
        for i in range(len(segments_bounds) - 1):
            start = segments_bounds[i]
            end = segments_bounds[i + 1]
            segment = df.iloc[start:end].copy()

            if len(segment) < 2:
                stats['segments_dropped'] += 1
                continue

            # 检查1: delta_t是否有超过30秒的
            if (segment['delta_t'].max() > 30.0):
                stats['segments_dropped'] += 1
                continue

            # 检查2: 是否有位置移动
            x_range = segment['x'].max() - segment['x'].min()
            y_range = segment['y'].max() - segment['y'].min()

            if x_range < 1 and y_range < 1:  # 移动小于1像素视为没动
                stats['segments_dropped'] += 1
                continue

            # 通过检查，保留该片段
            cleaned_segments.append(segment)
            stats['segments_kept'] += 1

        # 合并所有保留的片段
        if cleaned_segments:
            result_df = pd.concat(cleaned_segments, ignore_index=True)
            # 删除辅助列
            result_df = result_df.drop(columns=['delta_t'])

            # 在写入前检查最终数据量是否足够
            if len(result_df) < MIN_SEQUENCE_LENGTH:
                # 最终数据量不足，不写入文件
                stats['status'] = 'too_short_after_cleaning'
                stats['events_before'] = len(df)
                stats['events_after'] = len(result_df)
                stats['points_removed'] = points_removed
                return stats

            # 数据量足够，保存文件
            output_file.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_file, index=False)
            stats['status'] = 'success'
            stats['events_before'] = len(df)
            stats['events_after'] = len(result_df)
            stats['points_removed'] = points_removed
        else:
            stats['status'] = 'all_dropped'
            stats['points_removed'] = points_removed

        return stats

    except Exception as e:
        return {'status': 'error', 'error': str(e), 'points_removed': 0}


def clean_dataset(input_dir, output_dir, num_workers=16, demo=False):
    """
    清洗整个数据集（多线程）

    Args:
        input_dir: 输入目录 (boun-mouse-dynamics-dataset/users)
        output_dir: 输出目录 (boun-mouse-dynamics-dataset-cleaned/users)
        num_workers: 工作线程数
        demo: 如果为True，只处理user1和user2用于快速测试
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 清空输出目录
    if output_path.exists():
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
        print("✓ Output directory cleaned")

    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有CSV文件
    all_files = []
    user_dirs = sorted(input_path.iterdir())

    # Demo模式：只处理user1和user2
    if demo:
        user_dirs = [d for d in user_dirs if d.name in ['user1', 'user2']]
        print(f"⚡ DEMO MODE: Only processing user1 and user2")

    for user_dir in user_dirs:
        if not user_dir.is_dir():
            continue

        for test_type in ['internal_tests', 'external_tests']:
            test_dir = user_dir / test_type
            if not test_dir.exists():
                continue

            for csv_file in test_dir.glob('*.csv'):
                # 构造输出路径
                relative_path = csv_file.relative_to(input_path)
                output_file = output_path / relative_path
                all_files.append((csv_file, output_file))

    print(f"Found {len(all_files)} session files to process")
    print(f"Input dir: {input_path}")
    print(f"Output dir: {output_path}")
    print(f"Workers: {num_workers}")
    print("="*60)

    # 多线程处理所有文件
    global_stats = {
        'total_files': len(all_files),
        'success': 0,
        'empty': 0,
        'too_short_after_cleaning': 0,
        'all_dropped': 0,
        'error': 0,
        'total_segments_kept': 0,
        'total_segments_dropped': 0,
        'total_events_before': 0,
        'total_events_after': 0,
        'total_points_removed': 0
    }

    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(clean_session_file, all_files),
            total=len(all_files),
            desc="Cleaning files"
        ))

    # 汇总统计
    for stats in results:
        status = stats.get('status', 'unknown')
        if status in global_stats:
            global_stats[status] += 1

        global_stats['total_segments_kept'] += stats.get('segments_kept', 0)
        global_stats['total_segments_dropped'] += stats.get('segments_dropped', 0)
        global_stats['total_events_before'] += stats.get('events_before', 0)
        global_stats['total_events_after'] += stats.get('events_after', 0)
        global_stats['total_points_removed'] += stats.get('points_removed', 0)

    # 打印统计信息
    print("\n" + "="*60)
    print("Cleaning completed!")
    print("="*60)
    print(f"Total files processed: {global_stats['total_files']}")
    print(f"  ✓ Success: {global_stats['success']}")
    print(f"  ⊘ Empty: {global_stats['empty']}")
    print(f"  ⊘ Too short after cleaning (< {MIN_SEQUENCE_LENGTH} events): {global_stats['too_short_after_cleaning']}")
    print(f"  ⊘ All segments dropped: {global_stats['all_dropped']}")
    print(f"  ✗ Errors: {global_stats['error']}")
    print(f"\nSegments:")
    print(f"  ✓ Kept: {global_stats['total_segments_kept']}")
    print(f"  ✗ Dropped: {global_stats['total_segments_dropped']}")
    kept_ratio = global_stats['total_segments_kept'] / (global_stats['total_segments_kept'] + global_stats['total_segments_dropped']) * 100 if (global_stats['total_segments_kept'] + global_stats['total_segments_dropped']) > 0 else 0
    print(f"  Retention rate: {kept_ratio:.1f}%")
    print(f"\nEvents:")
    print(f"  Before: {global_stats['total_events_before']:,}")
    print(f"  After: {global_stats['total_events_after']:,}")
    event_ratio = global_stats['total_events_after'] / global_stats['total_events_before'] * 100 if global_stats['total_events_before'] > 0 else 0
    print(f"  Retention rate: {event_ratio:.1f}%")
    print(f"\nOut-of-bounds coordinates removed:")
    print(f"  Points removed: {global_stats['total_points_removed']:,}")
    print(f"  (x not in [0, {SCREEN_WIDTH}] or y not in [0, {SCREEN_HEIGHT}])")
    print("="*60)


if __name__ == "__main__":
    import argparse

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Clean BOUN Mouse Dynamics Dataset')
    parser.add_argument('--demo', action='store_true', help='Demo mode: only process user1 and user2')
    parser.add_argument('--workers', type=int, default=16, help='Number of parallel workers (default: 16)')
    args = parser.parse_args()

    # 配置路径
    INPUT_DIR = "boun-mouse-dynamics-dataset/users"
    OUTPUT_DIR = "boun-mouse-dynamics-dataset-cleaned/users"

    clean_dataset(INPUT_DIR, OUTPUT_DIR, num_workers=args.workers, demo=args.demo)

    print("\nDone! You can now update config.py:")
    print(f'DATA_DIR = "{OUTPUT_DIR}"')
