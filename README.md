# Conditional LSTM-VAE for Mouse Trajectory Generation

Generate realistic mouse trajectories conditioned on user ID, time period, starting/ending positions, and action type.

## Overview

This project implements a **Conditional LSTM-VAE** (Variational Autoencoder with LSTM) to generate realistic mouse movement trajectories. The model learns from the BOUN Mouse Dynamics Dataset and can generate trajectories based on specific conditions:

- **User ID**: Different users have different mouse usage patterns
- **Time Period**: Morning, afternoon, evening, or night
- **Action Type**: Move, click, or drag operations
- **Start/End Positions**: Where the trajectory begins and ends

## Project Structure

```
co-mouse/
├── boun-mouse-dynamics-dataset/         # Original raw dataset
├── boun-mouse-dynamics-dataset-cleaned/ # Cleaned dataset (after clean_data.py)
├── cache/                               # Dataset cache (optional, disabled by default)
├── checkpoints/                         # Saved model checkpoints
├── outputs/                            # Generated trajectories
├── runs/                               # TensorBoard logs
├── clean_data.py                       # Data cleaning script
├── config.py                           # Configuration settings
├── dataset.py                          # Data loading and preprocessing
├── diagnose.py                         # Diagnostic tool for debugging
├── model.py                            # Conditional LSTM-VAE architecture
├── train.py                            # Training script
├── generate.py                         # Generation script
├── test_setup.py                       # Setup verification script
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

## Installation

1. **Ensure you have Python 3.8+ installed**

2. **Install dependencies** (already in your venv):
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Clean the Dataset (Required First Step)

Before training, clean the raw dataset to filter and preprocess the data:

```bash
python clean_data.py
```

This will:
- Filter to keep only `window="browsing"` events
- Remove files with < 100 events
- Split by button release events
- Filter out segments with large time gaps or minimal movement
- Output to `boun-mouse-dynamics-dataset-cleaned/users/`

**Expected**: ~82,000 processed files from ~124,000 original files

### 2. Run Diagnostics (Optional but Recommended)

Check dataset statistics and model initialization:

```bash
python diagnose.py
```

This will show:
- Feature distributions (min/max/mean/variance)
- Model initialization stats
- Data sanity checks (user count, condition distributions)

### 3. Test Dataset Loading

Verify that the dataset is loaded correctly:

```bash
python dataset.py
```

This will display:
- Number of users
- Number of training/validation batches
- Sample batch shapes

### 4. Test Model Architecture

Verify the model architecture:

```bash
python model.py
```

This will show:
- Model architecture
- Input/output shapes
- Sample loss computation

### 5. Train the Model

Start training with default settings:

```bash
python train.py
```

**Training features:**
- Automatic train/validation split (80/20)
- KL annealing for stable training
- Learning rate scheduling
- Checkpoint saving every 5 epochs
- TensorBoard logging

**Resume from checkpoint:**
```bash
python train.py --resume checkpoints/checkpoint_epoch_20.pt
```

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir runs
```
Then open http://localhost:6006 in your browser.

### 6. Generate Trajectories

After training, generate new trajectories:

```bash
python generate.py --checkpoint checkpoints/best_model.pt --num_samples 5 --temperature 1.0
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint (required)
- `--num_samples`: Number of trajectories to generate per condition (default: 5)
- `--temperature`: Sampling temperature (higher = more diverse, default: 1.0)

**Output:**
- CSV files with trajectory data in `outputs/`
- PNG visualizations of trajectories in `outputs/`

## Configuration

Edit `config.py` to customize:

### Data Settings
```python
SEQUENCE_LENGTH = 100    # Number of mouse events per sequence
OVERLAP = 50            # Overlap between sequences
TRAIN_SPLIT = 0.8       # Train/validation split
```

### Model Architecture
```python
HIDDEN_DIM = 256        # LSTM hidden dimension
LATENT_DIM = 64         # Latent space dimension
NUM_LAYERS = 2          # Number of LSTM layers
DROPOUT = 0.2           # Dropout rate
```

### Training Parameters
```python
BATCH_SIZE = 1280       # Large batch size for GPU efficiency
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
KL_WEIGHT = 0.05        # Initial KL weight (grows to 1.0 over ~600 epochs)
KL_ANNEAL_EPOCHS = 30   # Warm-up period for KL annealing
```

### Data Settings
```python
DATA_DIR = "boun-mouse-dynamics-dataset-cleaned/users"
FILTER_WINDOW = "browsing"  # Only use browsing window events
USE_CACHE = False           # Caching disabled by default
NUM_WORKERS = 16            # Parallel workers for data preprocessing
```

## How It Works

### 0. Data Cleaning (Preprocessing)

The cleaning script (`clean_data.py`) preprocesses the raw dataset:

1. **Window Filtering**: Keeps only `window="browsing"` events
2. **Length Filtering**: Skips files with < 100 events after filtering
3. **Segmentation**: Splits sessions by `Released` button events
4. **Quality Filtering**:
   - Drops segments with > 30 second time gaps (interruptions)
   - Drops segments with < 1 pixel movement (static mouse)
5. **Output**: Cleaned dataset in `boun-mouse-dynamics-dataset-cleaned/`

**Result**: ~82,000 clean files from ~124,000 original files

### 1. Data Loading

The dataset loader (`dataset.py`) processes cleaned CSV files:

1. **Feature Engineering**:
   - `delta_t`: Time difference between events
   - `delta_x`, `delta_y`: Spatial displacement
   - `speed`: Movement speed
   - `acceleration`: Change in speed
   - `button`: Encoded button state
   - `state`: Encoded action state

2. **Sequence Creation**:
   - Sliding window with overlap
   - Extract start/end positions
   - Determine time period and action type

3. **Normalization**:
   - Coordinates normalized to [0, 1]
   - Screen resolution: 1920x1080 (configurable)

### 2. Model Architecture

The **Conditional LSTM-VAE** consists of:

**Encoder**:
- Embeds condition information (user, time, action, positions)
- LSTM processes input sequence + conditions
- Outputs mean (μ) and log variance (log σ²) of latent distribution

**Latent Space**:
- Reparameterization trick: `z = μ + σ * ε` (ε ~ N(0,1))
- Allows backpropagation through sampling

**Decoder**:
- Combines latent variable `z` with conditions
- LSTM generates output sequence
- Reconstructs trajectory features

**Loss Function**:
```
Total Loss = Reconstruction Loss + β * KL Divergence
```
- Reconstruction: MSE between input and output
- KL Divergence: Regularizes latent space to N(0,1)
- β annealing: Gradually increases from small value to 1.0

### 3. Generation Process

To generate a trajectory:

1. **Specify conditions**: User, time period, action type, start/end positions
2. **Sample latent variable**: `z ~ N(0, 1)`
3. **Decode**: Pass `z` + conditions through decoder
4. **Post-process**: Convert features to absolute coordinates

## Example Usage

### Custom Generation Script

```python
from config import Config
from model import ConditionalLSTMVAE, load_model
import torch

# Load trained model
config = Config()
model = load_model('checkpoints/best_model.pt', num_users=50, config=config)

# Define custom condition
condition = {
    'user_id': 5,
    'time_period': 1,        # 0=morning, 1=afternoon, 2=evening, 3=night
    'action_type': 0,        # 0=move, 1=click, 2=drag
    'start_pos': (100, 200),
    'end_pos': (800, 600)
}

# Generate trajectory
trajectory = model.generate(condition, seq_len=100, temperature=1.0)

# Post-process and save...
```

## Understanding the Output

### Generated CSV Format

```csv
timestamp,x,y,button,state
0.0,100,200,None,Move
0.012,102,203,None,Move
0.024,105,207,None,Move
...
```

### Visualization

The generated plots show:
- **Blue line**: Mouse trajectory path
- **Green circle**: Start position
- **Red X**: End position
- **Orange triangles**: Click events (if any)

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size or sequence length in `config.py`:
```python
BATCH_SIZE = 64  # Reduce from 1280
SEQUENCE_LENGTH = 50
```

### Issue: Poor Quality Trajectories

**Solutions**:
1. Train for more epochs (aim for loss < 0.5)
2. Run diagnostics to check feature distributions:
   ```bash
   python diagnose.py
   ```
3. Adjust KL weight if needed:
   ```python
   KL_WEIGHT = 0.01  # Lower if KL dominates
   ```
4. Increase model capacity:
   ```python
   HIDDEN_DIM = 512
   LATENT_DIM = 128
   ```

### Issue: Training Instability

**Solutions**:
1. Use gradient clipping (already enabled)
2. Lower learning rate:
   ```python
   LEARNING_RATE = 5e-4
   ```
3. Extend KL annealing:
   ```python
   KL_ANNEAL_EPOCHS = 40
   ```

## Advanced Features

### Diagnostics Tool

Use `diagnose.py` to debug training issues:

```bash
python diagnose.py
```

**What it checks**:
1. **Feature Distribution**: Shows min/max/mean/std/variance for all 7 features
   - Identifies scale imbalances (e.g., `delta_t` >> others)
2. **Model Initialization**: Tests untrained model on sample batch
   - Shows latent distribution statistics
   - Per-feature reconstruction errors
3. **Data Sanity**: Validates dataset properties
   - Dataset sizes, user counts
   - Condition distributions

**Common Issues Detected**:
- Feature scale imbalance → Normalize large-variance features
- High initial KL divergence → Lower `KL_WEIGHT`
- Uneven reconstruction errors → Consider weighted loss

### Data Cleaning Pipeline

The `clean_data.py` script ensures data quality:

**Configuration** (in script):
```python
MIN_SEQUENCE_LENGTH = 100  # Must match Config.SEQUENCE_LENGTH
MAX_TIME_GAP = 30.0        # Seconds
MIN_MOVEMENT = 1.0         # Pixels
NUM_WORKERS = 16           # Parallel processing
```

**Customization**:
- Adjust `MAX_TIME_GAP` to be more/less strict on interruptions
- Change `MIN_MOVEMENT` to filter static sequences differently
- Modify window filter to include other windows beyond "browsing"

### Data Augmentation

Enable in `config.py`:
```python
USE_AUGMENTATION = True
NOISE_LEVEL = 0.01
```

### Custom Screen Resolution

```python
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
```

### Model Variants

**Larger Model** (better quality, slower):
```python
HIDDEN_DIM = 512
LATENT_DIM = 128
NUM_LAYERS = 3
```

**Faster Model** (faster training, lower quality):
```python
HIDDEN_DIM = 128
LATENT_DIM = 32
NUM_LAYERS = 1
```

## Dataset Information

**BOUN Mouse Dynamics Dataset**:
- Multiple users with unique mouse usage patterns
- Internal and external test sessions
- Fields: timestamp, x, y, button, state, window

## Citation

If you use this code, please cite the BOUN Mouse Dynamics Dataset:
```
@inproceedings{ahmed2019boun,
  title={BOUN Mouse Dynamics Dataset},
  author={Ahmed, Ahmed Awad E and Traore, Issa},
  booktitle={...},
  year={2019}
}
```

## License

This project is provided as-is for research and educational purposes.

## Next Steps

1. **Experiment with hyperparameters**: Try different architectures and training settings
2. **Add more conditions**: Window name, day of week, etc.
3. **Improve post-processing**: Smooth trajectories, better coordinate reconstruction
4. **Evaluation metrics**: Implement quantitative metrics for trajectory realism
5. **Application-specific fine-tuning**: Train on specific application contexts

## Support

For questions or issues:
1. Check the Troubleshooting section
2. Review the configuration in `config.py`
3. Examine TensorBoard logs for training insights
4. Test with the example scripts first

Happy trajectory generation!
