"""
Dataset class for mouse trajectory data
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from config import Config
from pathlib import Path
import pickle
import hashlib
from multiprocessing import Pool, cpu_count
from functools import partial


# Global worker function for multiprocessing
def _process_file_worker(args):
    """Worker function to process a single file (must be global for multiprocessing)"""
    session_file, user_id, sequence_length, overlap, filter_window = args

    try:
        df = pd.read_csv(session_file)

        # Skip empty files
        if df.empty or len(df) == 0:
            return None, 'skipped'

        # Filter by window if specified
        if filter_window is not None:
            df = df[df['window'] == filter_window].reset_index(drop=True)

        if len(df) < sequence_length:
            return None, 'skipped'

        # Compute features (vectorized)
        features = _compute_features_static(df)

        # Extract time period
        time_period = _get_time_period_static(df['client_timestamp'].iloc[0])

        # Determine action type
        action_type = _get_action_type_static(df)

        # Create sliding window sequences
        sequences = []
        conditions = []
        stride = sequence_length - overlap

        for i in range(0, len(features) - sequence_length + 1, stride):
            seq = features[i:i + sequence_length]

            # Extract start and end positions
            start_x = df['x'].iloc[i] / Config.SCREEN_WIDTH
            start_y = df['y'].iloc[i] / Config.SCREEN_HEIGHT
            end_x = df['x'].iloc[i + sequence_length - 1] / Config.SCREEN_WIDTH
            end_y = df['y'].iloc[i + sequence_length - 1] / Config.SCREEN_HEIGHT

            sequences.append(seq)
            conditions.append({
                'user_id': user_id,
                'time_period': time_period,
                'action_type': action_type,
                'start_pos': np.array([start_x, start_y], dtype=np.float32),
                'end_pos': np.array([end_x, end_y], dtype=np.float32)
            })

        return (sequences, conditions), 'success'

    except Exception as e:
        return None, 'error'


def _compute_features_static(df):
    """Static version of feature computation for multiprocessing"""
    n = len(df)

    x = df['x'].values.astype(np.float32) / Config.SCREEN_WIDTH
    y = df['y'].values.astype(np.float32) / Config.SCREEN_HEIGHT
    t = df['client_timestamp'].values.astype(np.float32)

    button = df['button'].map(Config.BUTTON_MAP).fillna(0).values.astype(np.float32)
    state = df['state'].map(Config.STATE_MAP).fillna(0).values.astype(np.float32)

    delta_t = np.zeros(n, dtype=np.float32)
    delta_x = np.zeros(n, dtype=np.float32)
    delta_y = np.zeros(n, dtype=np.float32)

    delta_t[1:] = t[1:] - t[:-1]
    delta_x[1:] = x[1:] - x[:-1]
    delta_y[1:] = y[1:] - y[:-1]

    # Clip delta_t to reasonable range (0-5 seconds)
    # This prevents outliers from dominating the loss
    delta_t = np.clip(delta_t, 0, 5.0)

    distance = np.sqrt(delta_x**2 + delta_y**2)
    speed = distance / (delta_t + 1e-6)
    speed = np.clip(speed, 0, 10)  # Clip to reasonable range
    speed_normalized = speed / 10.0  # Normalize to [0, 1]

    acceleration = np.zeros(n, dtype=np.float32)
    acceleration[2:] = (speed[2:] - speed[1:-1]) / (delta_t[2:] + 1e-6)
    acceleration = np.clip(acceleration, -100, 100)  # Clip to reasonable range
    acceleration_normalized = (acceleration + 100.0) / 200.0  # Normalize to [0, 1]

    # Normalize button and state
    button_normalized = button / 3.0  # Max value is 3 -> [0, 1]
    state_normalized = state / 2.0    # Max value is 2 -> [0, 1]

    features = np.stack([delta_t, delta_x, delta_y, speed_normalized, acceleration_normalized, button_normalized, state_normalized], axis=1).astype(np.float32)
    return features


def _get_time_period_static(timestamp):
    """Static version for multiprocessing"""
    dt = datetime.fromtimestamp(timestamp)
    hour = dt.hour
    if 6 <= hour < 12:
        return 0
    elif 12 <= hour < 18:
        return 1
    elif 18 <= hour < 22:
        return 2
    else:
        return 3


def _get_action_type_static(df):
    """Static version for multiprocessing"""
    has_press = (df['state'] == 'Pressed').any()
    has_release = (df['state'] == 'Released').any()

    if has_press and has_release:
        press_idx = df[df['state'] == 'Pressed'].index
        release_idx = df[df['state'] == 'Released'].index

        if len(press_idx) > 0 and len(release_idx) > 0:
            press_pos = df.loc[press_idx[0], ['x', 'y']].values
            release_pos = df.loc[release_idx[0], ['x', 'y']].values
            distance = np.linalg.norm(release_pos - press_pos)

            if distance > 10:
                return 2  # drag
            else:
                return 1  # click

    return 0  # move


class MouseTrajectoryDataset(Dataset):
    def __init__(self, data_dir, sequence_length=100, overlap=50, train=True, train_split=0.8, max_sessions_per_user=None, filter_window=None, use_dataset_split=True, demo=False):
        """
        Args:
            data_dir: Path to users directory
            sequence_length: Number of events per sequence
            overlap: Number of overlapping events between sequences
            train: If True, load training data; else validation data
            train_split: Fraction of data for training (only used if use_dataset_split=False)
            max_sessions_per_user: Max sessions to load per user (None=all, for quick testing use 10)
            filter_window: Filter events by window name (None=all, "browsing"=only browsing)
            use_dataset_split: If True, use dataset's built-in train/val split (training/ vs internal_tests/)
            demo: If True, only process user1 and user2 for quick testing
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.overlap = overlap
        self.train = train
        self.max_sessions_per_user = max_sessions_per_user
        self.filter_window = filter_window
        self.use_dataset_split = use_dataset_split
        self.demo = demo

        # Load all sessions
        self.sequences = []
        self.conditions = []

        print(f"\n{'='*60}")
        print(f"Loading {'training' if train else 'validation'} data...")
        if demo:
            print(f"⚡ DEMO MODE: Only loading user1 and user2")
        if use_dataset_split:
            print(f"Using dataset's built-in split: {'training/' if train else 'internal_tests/'}")
        else:
            print(f"Using random split: {train_split*100:.0f}% train / {(1-train_split)*100:.0f}% val")
        if max_sessions_per_user:
            print(f"Quick mode: loading max {max_sessions_per_user} sessions per user")
        print(f"Note: Data already filtered to 'browsing' window by clean_data.py")
        print(f"{'='*60}\n")

        # Try to load from cache
        cache_loaded = False
        if hasattr(Config, 'USE_CACHE') and Config.USE_CACHE:
            cache_loaded = self._try_load_from_cache(train_split, use_dataset_split)

        if not cache_loaded:
            print("Building dataset from scratch...")
            self._load_data(train_split)

            # Save to cache
            if hasattr(Config, 'USE_CACHE') and Config.USE_CACHE:
                self._save_to_cache(train_split, use_dataset_split)

        print(f"\n{'='*60}")
        print(f"✓ Loaded {len(self.sequences)} sequences")

        # Get unique users for embedding
        self.user_to_idx = {}
        for cond in self.conditions:
            user = cond['user_id']
            if user not in self.user_to_idx:
                self.user_to_idx[user] = len(self.user_to_idx)

        self.num_users = len(self.user_to_idx)
        print(f"✓ Number of unique users: {self.num_users}")
        print(f"{'='*60}\n")

    def _load_data(self, train_split):
        """Load and preprocess all session files with multiprocessing"""
        from tqdm import tqdm

        user_dirs = sorted([d for d in Path(self.data_dir).iterdir() if d.is_dir()])

        # Demo模式：只加载user1和user2
        if self.demo:
            user_dirs = [d for d in user_dirs if d.name in ['user1', 'user2']]

        print(f"Found {len(user_dirs)} users")

        # Collect all file tasks
        file_tasks = []
        for user_dir in user_dirs:
            user_id = user_dir.name
            sessions_for_user = 0

            if self.use_dataset_split:
                # Use dataset's built-in split
                if self.train:
                    # Training: use 'training/' directory
                    test_types = ['training']
                else:
                    # Validation: use 'internal_tests/' directory
                    test_types = ['internal_tests']
            else:
                # Use random split across both directories
                test_types = ['internal_tests', 'external_tests']

            for test_type in test_types:
                test_dir = user_dir / test_type
                if not test_dir.exists():
                    continue

                session_files = sorted(test_dir.glob('*.csv'))

                if self.use_dataset_split:
                    # Use all files from the designated directory
                    files = session_files
                else:
                    # Split into train/val
                    split_idx = int(len(session_files) * train_split)
                    if self.train:
                        files = session_files[:split_idx]
                    else:
                        files = session_files[split_idx:]

                # Limit sessions
                if self.max_sessions_per_user:
                    remaining = self.max_sessions_per_user - sessions_for_user
                    files = files[:remaining]

                for session_file in files:
                    file_tasks.append((session_file, user_id, self.sequence_length, self.overlap, self.filter_window))
                    sessions_for_user += 1

                    if self.max_sessions_per_user and sessions_for_user >= self.max_sessions_per_user:
                        break

        print(f"Total files to process: {len(file_tasks)}")

        # Get number of workers
        num_workers = getattr(Config, 'NUM_WORKERS', 0)
        if num_workers > 0:
            print(f"Using {num_workers} parallel workers")
        else:
            print(f"Using single-threaded mode")

        # Process files
        self.stats = {'processed': 0, 'skipped': 0, 'errors': 0}

        if num_workers > 0:
            # Multiprocessing
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(
                    pool.imap(_process_file_worker, file_tasks),
                    total=len(file_tasks),
                    desc="Processing files"
                ))
        else:
            # Single threaded
            results = []
            for task in tqdm(file_tasks, desc="Processing files"):
                results.append(_process_file_worker(task))

        # Collect results
        for result, status in results:
            if status == 'success' and result is not None:
                sequences, conditions = result
                self.sequences.extend(sequences)
                self.conditions.extend(conditions)
                self.stats['processed'] += 1
            elif status == 'skipped':
                self.stats['skipped'] += 1
            elif status == 'error':
                self.stats['errors'] += 1

        # Print statistics
        print(f"\nDataset statistics:")
        print(f"  ✓ Processed: {self.stats['processed']} files")
        if self.stats['skipped'] > 0:
            print(f"  ⊘ Skipped: {self.stats['skipped']} files (empty/too short)")
        if self.stats['errors'] > 0:
            print(f"  ✗ Errors: {self.stats['errors']} files (corrupted)")

    def _process_session(self, session_file, user_id):
        """Process a single session file into sequences"""
        try:
            df = pd.read_csv(session_file)

            # Skip empty files
            if df.empty or len(df) == 0:
                self.stats['skipped'] += 1
                return False

            # Filter by window if specified
            if self.filter_window is not None:
                df = df[df['window'] == self.filter_window].reset_index(drop=True)

            if len(df) < self.sequence_length:
                self.stats['skipped'] += 1
                return False

            # Compute features
            features = self._compute_features(df)

            # Extract time period from timestamp
            time_period = self._get_time_period(df['client_timestamp'].iloc[0])

            # Determine action type from the sequence
            action_type = self._get_action_type(df)

            # Create sliding window sequences
            stride = self.sequence_length - self.overlap
            for i in range(0, len(features) - self.sequence_length + 1, stride):
                seq = features[i:i + self.sequence_length]

                # Extract start and end positions
                start_x = df['x'].iloc[i] / Config.SCREEN_WIDTH
                start_y = df['y'].iloc[i] / Config.SCREEN_HEIGHT
                end_x = df['x'].iloc[i + self.sequence_length - 1] / Config.SCREEN_WIDTH
                end_y = df['y'].iloc[i + self.sequence_length - 1] / Config.SCREEN_HEIGHT

                self.sequences.append(seq)
                self.conditions.append({
                    'user_id': user_id,
                    'time_period': time_period,
                    'action_type': action_type,
                    'start_pos': np.array([start_x, start_y], dtype=np.float32),
                    'end_pos': np.array([end_x, end_y], dtype=np.float32)
                })

            return True

        except Exception as e:
            # Silently count errors - corrupted files are normal in datasets
            self.stats['errors'] += 1
            return False

    def _compute_features(self, df):
        """Compute feature vector for each event - VECTORIZED for speed"""
        n = len(df)

        # Normalize coordinates - vectorized
        x = df['x'].values.astype(np.float32) / Config.SCREEN_WIDTH
        y = df['y'].values.astype(np.float32) / Config.SCREEN_HEIGHT
        t = df['client_timestamp'].values.astype(np.float32)

        # Encode button and state - vectorized
        button = df['button'].map(Config.BUTTON_MAP).fillna(0).values.astype(np.float32)
        state = df['state'].map(Config.STATE_MAP).fillna(0).values.astype(np.float32)

        # Compute deltas - vectorized
        delta_t = np.zeros(n, dtype=np.float32)
        delta_x = np.zeros(n, dtype=np.float32)
        delta_y = np.zeros(n, dtype=np.float32)

        delta_t[1:] = t[1:] - t[:-1]
        delta_x[1:] = x[1:] - x[:-1]
        delta_y[1:] = y[1:] - y[:-1]

        # Clip delta_t to reasonable range (0-5 seconds)
        # This prevents outliers from dominating the loss
        delta_t = np.clip(delta_t, 0, 5.0)

        # Compute speed - vectorized
        distance = np.sqrt(delta_x**2 + delta_y**2)
        speed = distance / (delta_t + 1e-6)
        speed = np.clip(speed, 0, 10)  # Clip to reasonable range
        speed_normalized = speed / 10.0  # Normalize to [0, 1]

        # Compute acceleration - vectorized
        acceleration = np.zeros(n, dtype=np.float32)
        acceleration[2:] = (speed[2:] - speed[1:-1]) / (delta_t[2:] + 1e-6)
        acceleration = np.clip(acceleration, -100, 100)  # Clip to reasonable range
        acceleration_normalized = (acceleration + 100.0) / 200.0  # Normalize to [0, 1]

        # Normalize button and state
        button_normalized = button / 3.0  # Max value is 3 -> [0, 1]
        state_normalized = state / 2.0    # Max value is 2 -> [0, 1]

        # Stack all features - vectorized
        features = np.stack([
            delta_t,
            delta_x,
            delta_y,
            speed_normalized,
            acceleration_normalized,
            button_normalized,
            state_normalized
        ], axis=1).astype(np.float32)

        return features

    def _get_time_period(self, timestamp):
        """Determine time period (0: morning, 1: afternoon, 2: evening, 3: night)"""
        dt = datetime.fromtimestamp(timestamp)
        hour = dt.hour

        if 6 <= hour < 12:
            return 0  # morning
        elif 12 <= hour < 18:
            return 1  # afternoon
        elif 18 <= hour < 22:
            return 2  # evening
        else:
            return 3  # night

    def _get_action_type(self, df):
        """Determine primary action type (0: move, 1: click, 2: drag)"""
        # Check if there are any button presses
        has_press = (df['state'] == 'Pressed').any()
        has_release = (df['state'] == 'Released').any()

        # If there's a press and release, it's a click or drag
        if has_press and has_release:
            # Check if there's significant movement during pressed state
            press_idx = df[df['state'] == 'Pressed'].index
            release_idx = df[df['state'] == 'Released'].index

            if len(press_idx) > 0 and len(release_idx) > 0:
                press_pos = df.loc[press_idx[0], ['x', 'y']].values
                release_pos = df.loc[release_idx[0], ['x', 'y']].values
                distance = np.linalg.norm(release_pos - press_pos)

                if distance > 10:  # If moved more than 10 pixels
                    return 2  # drag
                else:
                    return 1  # click

        return 0  # move

    def _get_cache_path(self, train_split, use_dataset_split):
        """Generate unique cache filename based on parameters"""
        # Create a hash of all parameters that affect the data
        params_str = f"{self.sequence_length}_{self.overlap}_{train_split}_{self.train}_{self.max_sessions_per_user}_{self.filter_window}_{use_dataset_split}"
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]

        cache_dir = Path(Config.CACHE_DIR if hasattr(Config, 'CACHE_DIR') else 'cache')
        cache_dir.mkdir(exist_ok=True)

        filename = f"dataset_{'train' if self.train else 'val'}_{params_hash}.pkl"
        return cache_dir / filename

    def _try_load_from_cache(self, train_split, use_dataset_split):
        """Try to load preprocessed data from cache"""
        cache_path = self._get_cache_path(train_split, use_dataset_split)

        if cache_path.exists():
            print(f"Loading from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)

                self.sequences = cached_data['sequences']
                self.conditions = cached_data['conditions']

                print(f"✓ Cache loaded successfully!")
                return True
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}")
                print("  Will rebuild dataset...")
                return False

        print(f"No cache found at {cache_path}")
        return False

    def _save_to_cache(self, train_split, use_dataset_split):
        """Save preprocessed data to cache"""
        cache_path = self._get_cache_path(train_split, use_dataset_split)

        print(f"\nSaving to cache: {cache_path}")
        try:
            cache_data = {
                'sequences': self.sequences,
                'conditions': self.conditions
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"✓ Cache saved successfully!")
            file_size = cache_path.stat().st_size / 1024 / 1024  # MB
            print(f"  Cache size: {file_size:.2f} MB")
        except Exception as e:
            print(f"⚠ Failed to save cache: {e}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Returns:
            sequence: Tensor of shape (sequence_length, input_dim)
            condition: Dictionary with condition information
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        condition = self.conditions[idx]

        # Convert condition to tensors
        user_idx = self.user_to_idx[condition['user_id']]
        condition_tensor = {
            'user_id': torch.LongTensor([user_idx]),
            'time_period': torch.LongTensor([condition['time_period']]),
            'action_type': torch.LongTensor([condition['action_type']]),
            'start_pos': torch.FloatTensor(condition['start_pos']),
            'end_pos': torch.FloatTensor(condition['end_pos'])
        }

        return sequence, condition_tensor


def get_dataloaders(config, demo=False):
    """Create train and validation dataloaders"""
    use_dataset_split = getattr(config, 'USE_DATASET_SPLIT', True)

    train_dataset = MouseTrajectoryDataset(
        data_dir=config.DATA_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        overlap=config.OVERLAP,
        train=True,
        train_split=config.TRAIN_SPLIT,
        max_sessions_per_user=config.MAX_SESSIONS_PER_USER,
        filter_window=config.FILTER_WINDOW,
        use_dataset_split=use_dataset_split,
        demo=demo
    )

    val_dataset = MouseTrajectoryDataset(
        data_dir=config.DATA_DIR,
        sequence_length=config.SEQUENCE_LENGTH,
        overlap=config.OVERLAP,
        train=False,
        train_split=config.TRAIN_SPLIT,
        max_sessions_per_user=config.MAX_SESSIONS_PER_USER,
        filter_window=config.FILTER_WINDOW,
        use_dataset_split=use_dataset_split,
        demo=demo
    )

    # DataLoader workers: 0 for Windows, 4-8 for Linux/Mac
    import platform
    dataloader_workers = 0 if platform.system() == 'Windows' else min(8, config.NUM_WORKERS)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=dataloader_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=dataloader_workers,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )

    return train_loader, val_loader, train_dataset.num_users


if __name__ == "__main__":
    # Test dataset loading
    config = Config()
    train_loader, val_loader, num_users = get_dataloaders(config)

    print(f"\nNumber of users: {num_users}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test one batch
    for sequences, conditions in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Sequences: {sequences.shape}")
        print(f"  User IDs: {conditions['user_id'].shape}")
        print(f"  Time periods: {conditions['time_period'].shape}")
        print(f"  Action types: {conditions['action_type'].shape}")
        print(f"  Start positions: {conditions['start_pos'].shape}")
        print(f"  End positions: {conditions['end_pos'].shape}")
        break
