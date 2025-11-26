# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements a **Conditional LSTM-VAE** (Variational Autoencoder with LSTM) for generating realistic mouse movement trajectories. The model learns from the BOUN Mouse Dynamics Dataset and generates trajectories conditioned on:
- User ID (different users have different mouse usage patterns)
- Time period (morning/afternoon/evening/night)
- Action type (move/click/drag)
- Start and end positions

## Essential Commands

### Setup and Testing
```bash
# Install dependencies (Python 3.8+)
pip install -r requirements.txt

# Test dataset loading
python dataset.py

# Test model architecture
python model.py

# Run comprehensive setup tests
python test_setup.py
```

### Training
```bash
# Start training with default settings
python train.py

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_20.pt

# Monitor training with TensorBoard
tensorboard --logdir runs
# Then open http://localhost:6006
```

### Data Cleaning
```bash
# Clean raw dataset (filter browsing window, remove short sequences)
python clean_data.py
# Input: boun-mouse-dynamics-dataset/users
# Output: boun-mouse-dynamics-dataset-cleaned/users
```

### Diagnostics
```bash
# Run diagnostic checks on dataset and model
python diagnose.py
# Outputs: feature distribution, model initialization stats, data sanity checks
```

### Generation
```bash
# Generate trajectories from trained model
python generate.py --checkpoint checkpoints/best_model.pt --num_samples 5 --temperature 1.0
```

## Architecture Overview

### Core Components

**1. Configuration (`config.py`)**
- Centralized configuration using `Config` class
- **IMPORTANT**: Config requires CUDA GPU - will raise RuntimeError if CUDA unavailable
- Key settings:
  - Data: `DATA_DIR="boun-mouse-dynamics-dataset-cleaned/users"`, `SEQUENCE_LENGTH=100`, `OVERLAP=50`, `TRAIN_SPLIT=0.8`
  - Model: `HIDDEN_DIM=256`, `LATENT_DIM=64`, `NUM_LAYERS=2`
  - Training: `BATCH_SIZE=1280`, `LEARNING_RATE=1e-3`, `KL_WEIGHT=0.05`, `KL_ANNEAL_EPOCHS=30`
  - Performance: `USE_CACHE=False` (disabled by default), `NUM_WORKERS=16` for parallel data loading
  - Filtering: `FILTER_WINDOW="browsing"` limits to browsing window events only

**2. Dataset (`dataset.py`)**
- `MouseTrajectoryDataset` class handles loading and preprocessing
- **Caching System**: Optional pickle cache in `cache/` directory (disabled by default with `USE_CACHE=False`)
- **Multiprocessing**: Uses `NUM_WORKERS=16` parallel processes for data preprocessing
- Features computed per event: `[delta_t, delta_x, delta_y, speed, acceleration, button_encoded, state_encoded]`
- Sliding window approach with overlap to create sequences
- Coordinate normalization to [0, 1] based on screen resolution (1920x1080)
- **Key Methods**:
  - `_compute_features()`: Vectorized feature engineering
  - `_get_time_period()`: Maps timestamp to time period (0-3)
  - `_get_action_type()`: Classifies sequence as move/click/drag
  - Cache methods: `_get_cache_path()`, `_try_load_from_cache()`, `_save_to_cache()`

**3. Model (`model.py`)**
- `ConditionalLSTMVAE` class: encoder-decoder architecture
- **Encoder**: LSTM processes input + conditions → latent distribution (μ, log σ²)
- **Reparameterization trick**: `z = μ + σ * ε` for backpropagation through sampling
- **Decoder**: Latent z + conditions → LSTM → reconstructed trajectory
- **Loss with Per-Feature Weighting**: `Total = Weighted MSE Reconstruction + β * KL Divergence`
  - Feature weights: `[delta_t: 0.01, delta_x: 100.0, delta_y: 100.0, others: 1.0]`
  - **Critical**: delta_x/delta_y have 100× weight to ensure model learns trajectory endpoints
  - delta_t has 0.01× weight to prevent outliers from dominating loss
  - Without weighting, delta_t variance (7.93 before clipping) would dominate 85%+ of loss
- Condition encoding via embeddings for categorical variables (user, time, action) + normalized positions
- **Key Methods**:
  - `encode_condition()`: Combines all condition embeddings
  - `encode()`: Sequence → latent parameters
  - `reparameterize()`: Sampling with gradient flow
  - `decode()`: Latent → sequence
  - `generate()`: Sample from prior + decode
  - `vae_loss()`: Computes weighted reconstruction loss + KL divergence

**4. Training (`train.py`)**
- `train_epoch()`: Training loop with KL annealing
- `validate()`: Validation without gradient updates
- **KL Annealing**: Gradually increases KL weight from `KL_WEIGHT` to 1.0 over `KL_ANNEAL_EPOCHS`
- Gradient clipping at `GRADIENT_CLIP=1.0` for stability
- Learning rate scheduling with ReduceLROnPlateau (factor=0.5, patience=5)
- Checkpointing:
  - Saves every `SAVE_EVERY` epochs (default: 5)
  - Always saves `best_model.pt` when validation loss improves
  - Final model saved as `final_model.pt`
- TensorBoard logging: train/val losses, KL weight

**5. Generation (`generate.py`)**
- `load_model()`: Loads checkpoint and creates model
- `create_condition()`: Builds condition tensor from parameters
- `decode_trajectory()`: Converts feature deltas → absolute coordinates
- `plot_trajectory()`: Visualizes trajectories with matplotlib
- Outputs saved to `outputs/` directory with descriptive filenames

**6. Data Cleaning (`clean_data.py`)**
- Preprocesses raw BOUN dataset before training
- **Filtering**: Only keeps `window="browsing"` events (matches `Config.FILTER_WINDOW`)
- **Length Check**: Skips files with < 100 events after filtering (matches `Config.SEQUENCE_LENGTH`)
- **Segmentation**: Splits by `Released` events, filters segments by:
  - Time gaps: Drops segments with > 30s gaps between events
  - Movement: Drops segments with < 1 pixel movement
- **Multiprocessing**: Uses 16 workers for parallel processing
- **Output**: `boun-mouse-dynamics-dataset-cleaned/users/` with same structure as input

**7. Diagnostics (`diagnose.py`)**
- Comprehensive diagnostic tool for debugging training issues
- **Optimized**: Loads dataset only once, then runs all checks
- Three diagnostic checks:
  1. `check_feature_distribution()`: Shows min/max/mean/std/variance for all 7 features
  2. `check_model_output()`: Tests untrained model on sample batch, shows latent distribution and per-feature reconstruction errors
  3. `check_data_sanity()`: Validates dataset size, user count, condition distributions
- **Important**: Converts tensors to CPU with `.cpu().numpy()` and casts to int64 for `np.bincount()`
- Useful for identifying feature scale issues:
  - Before fixes: `delta_t` variance = 7.93, max = 512s (outliers dominating)
  - After fixes: `delta_t` variance = 0.01, max = 5s (clipped)

### Data Flow

1. **Cleaning** (optional): Raw CSV → Filter browsing window → Filter short files → Split by Release → Filter segments → Cleaned CSV
2. **Loading**: Cleaned CSV → Feature engineering → Sliding windows → (Optional cache) → Dataset
3. **Training**: Sequences + conditions → Encoder → Latent z → Decoder → Reconstruction
4. **Generation**: Conditions + random z ~ N(0,1) → Decoder → Trajectory features → Absolute coordinates

## Important Implementation Details

### Dataset Structure
```
boun-mouse-dynamics-dataset/users/
├── user001/
│   ├── internal_tests/
│   │   ├── session_001.csv  # Fields: client_timestamp, x, y, button, state, window
│   │   └── ...
│   └── external_tests/
└── user002/
    └── ...
```

### Feature Engineering
All operations are vectorized using NumPy for performance:
- `delta_t[1:] = t[1:] - t[:-1]` (time differences)
- **`delta_t = np.clip(delta_t, 0, 5.0)`** - Clips outliers to [0, 5] seconds
  - Prevents extreme values (originally up to 512s) from dominating loss
  - Applied in both `_compute_features_static()` and `_compute_features()`
  - Reduces variance from 7.93 to 0.01 without dropping data
- `delta_x[1:] = x[1:] - x[:-1]` (spatial displacements, normalized)
- `delta_y[1:] = y[1:] - y[:-1]`
- `speed = distance / (delta_t + 1e-6)`
- `acceleration[2:] = (speed[2:] - speed[1:-1]) / (delta_t[2:] + 1e-6)`
- Button encoding: `{'None': 0, 'Left': 1, 'Right': 2, 'Middle': 3}`
- State encoding: `{'Move': 0, 'Pressed': 1, 'Released': 2}`

### Conditioning System
The model concatenates embeddings + normalized positions:
- User embedding: `(batch, USER_EMBED_DIM=32)`
- Time embedding: `(batch, TIME_PERIOD_EMBED_DIM=16)`
- Action embedding: `(batch, ACTION_TYPE_EMBED_DIM=16)`
- Positions: `(batch, POSITION_DIM=4)` for [start_x, start_y, end_x, end_y]
- Total condition vector: `(batch, CONDITION_DIM=128)` (though computed dims sum to 68)

### KL Annealing Strategy
```python
kl_weight = min(1.0, config.KL_WEIGHT * (epoch / config.KL_ANNEAL_EPOCHS))
```
Starts at `KL_WEIGHT=0.05` and linearly grows over epochs:
- Epoch 0-30: KL weight increases from 0 to 0.05 (warm-up phase)
- Epoch 30-600: KL weight continues growing from 0.05 to 1.0
- Epoch 600+: Full KL weight of 1.0 is reached

This prevents posterior collapse in early training.

### Cache System
Cache filename includes hash of: `sequence_length, overlap, train_split, train/val, max_sessions_per_user, filter_window`
- Format: `dataset_{train|val}_{hash}.pkl`
- Stored in `cache/` directory
- **Currently disabled** by default (`Config.USE_CACHE = False`)
- Set `Config.USE_CACHE = True` to enable caching for faster subsequent loads

## Configuration Tuning

### For Faster Training (Lower Quality)
```python
HIDDEN_DIM = 128
LATENT_DIM = 32
NUM_LAYERS = 1
BATCH_SIZE = 512
```

### For Better Quality (Slower)
```python
HIDDEN_DIM = 512
LATENT_DIM = 128
NUM_LAYERS = 3
BATCH_SIZE = 64
KL_WEIGHT = 0.0001  # Start lower
KL_ANNEAL_EPOCHS = 40  # Anneal longer
```

### For Memory Issues
```python
BATCH_SIZE = 16
SEQUENCE_LENGTH = 50
NUM_WORKERS = 0  # Disable multiprocessing
```

### For Quick Testing
```python
MAX_SESSIONS_PER_USER = 10  # Limit data
USE_CACHE = True  # Speed up subsequent loads
```

## Data Cleaning and Diagnostics

### Cleaning Workflow
**IMPORTANT**: Run `clean_data.py` before training to ensure dataset quality and consistency.

1. **Why clean?** The raw BOUN dataset contains:
   - Events from multiple windows (not just browsing)
   - Files with < 100 events (too short for SEQUENCE_LENGTH)
   - Segments with large time gaps (> 30s, likely interruptions)
   - Static sequences (< 1 pixel movement, not useful for trajectory learning)

2. **What clean_data.py does**:
   - Filters to keep only `window="browsing"` events
   - Removes files with < 100 events after filtering
   - Splits sessions by `Released` button events
   - Drops segments with > 30s time gaps or < 1 pixel movement
   - Outputs cleaned data to `boun-mouse-dynamics-dataset-cleaned/users/`

3. **Expected results**:
   - Original: ~124,000 files
   - After cleaning: ~82,000 files (34% reduction)
   - This ensures `dataset.py` skips minimal files during training

### Diagnostic Workflow
**Use `diagnose.py` to debug training issues**:

1. **When to run**:
   - Before training: Verify dataset loads correctly
   - After poor training: Identify feature scale issues
   - When loss is stuck: Check model initialization

2. **What to look for**:
   - Feature variance imbalance (e.g., `delta_t` variance >> others)
   - Extreme values in features (outliers not clipped properly)
   - KL divergence too high/low initially
   - Uneven condition distributions (some users/actions too rare)

3. **Fixed issues** (as of latest commits):
   - **delta_t outliers**: Now clipped to [0, 5] seconds in `dataset.py`
     - Before: max=512s, variance=7.93
     - After: max=5s, variance=0.01
   - **Feature weight imbalance**: Per-feature weights in `model.py:vae_loss()`
     - delta_x/delta_y now have 100× weight vs delta_t's 0.01× weight
   - **Tensor type errors**: `diagnose.py` now uses `.cpu().numpy().astype(np.int64)`

## Common Issues

### GPU Required
The config enforces CUDA availability. If you need CPU support, modify `config.py`:
```python
# Change from:
if not torch.cuda.is_available():
    raise RuntimeError(...)
DEVICE = torch.device("cuda")

# To:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Windows DataLoader
DataLoader `num_workers` is set to 0 for Windows compatibility in `dataset.py:526`. Multiprocessing for preprocessing uses `Config.NUM_WORKERS=12`.

### Empty/Corrupted Files
The dataset loader silently skips empty or corrupted files and reports statistics at the end.

### Loss Values After Feature Weighting Fix
**After applying per-feature weights (delta_x/delta_y ×100), loss values will be much higher but this is correct:**
- **Before fix**: Val loss ~0.3 (low because model ignored delta_x/delta_y)
  - delta_t MSE: 6.53 (85% of total loss)
  - delta_x/delta_y MSE: 0.0008 (ignored by optimizer)
  - Generated trajectories missed targets by 500+ pixels
- **After fix**: Val loss ~10-30 (higher but learning the right features)
  - delta_t MSE: 0.01-0.05 (controlled)
  - delta_x/delta_y MSE: 0.1-0.3 (now being optimized!)
  - Generated trajectories should reach targets within <100 pixels

**Do not be alarmed by higher loss values** - the model is now correctly prioritizing trajectory endpoints over time prediction. Evaluate model quality by:
1. Running `diagnose.py` to check per-feature MSE balance
2. Generating trajectories and measuring endpoint error
3. Visual inspection of trajectory smoothness and realism

## Model Checkpoints

Checkpoint structure:
```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': dict,
    'loss': float,
    'config': Config object
}
```

To load for inference only:
```python
from model import ConditionalLSTMVAE
model = ConditionalLSTMVAE(config, num_users).to(device)
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Generation Parameters

- `user_id`: Integer index (0 to num_users-1)
- `time_period`: 0=morning (6-12h), 1=afternoon (12-18h), 2=evening (18-22h), 3=night (22-6h)
- `action_type`: 0=move, 1=click, 2=drag
- `start_pos`, `end_pos`: Tuples (x, y) in pixels
- `temperature`: Sampling variance (0.5=conservative, 1.0=default, 1.5=diverse)

## File Organization

- `config.py`: All hyperparameters and settings
- `dataset.py`: Data loading, preprocessing, caching
- `model.py`: LSTM-VAE architecture and loss function
- `train.py`: Training loop with checkpointing
- `generate.py`: Trajectory generation and visualization
- `test_setup.py`: Comprehensive setup verification
- `checkpoints/`: Saved model checkpoints
- `cache/`: Preprocessed dataset cache (pickled)
- `outputs/`: Generated trajectories (CSV + PNG)
- `runs/`: TensorBoard logs
