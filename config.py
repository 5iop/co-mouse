"""
Configuration file for Conditional LSTM-VAE Mouse Trajectory Generation
"""

import torch

class Config:
    # Data settings
    DATA_DIR = "boun-mouse-dynamics-dataset-cleaned/users"
    SEQUENCE_LENGTH = 100  # Number of mouse events per sequence
    OVERLAP = 50  # Overlap between consecutive sequences
    TRAIN_SPLIT = 0.8  # Only used if USE_DATASET_SPLIT=False
    USE_DATASET_SPLIT = False  # Use random split (training/ dir is too small, only 10 files per user)
    MAX_SESSIONS_PER_USER = None  # Limit sessions for quick testing (None=all, 10=fast, 50=medium)
    FILTER_WINDOW = None  # No filtering needed - clean_data.py already filtered to 'browsing' only
    USE_CACHE = False  # Cache disabled - will rebuild dataset from scratch each time
    CACHE_DIR = "cache"  # Directory for cached data
    NUM_WORKERS = 16  # Number of parallel workers for data loading (0=single thread, 12=fast)

    # Screen resolution (for normalization)
    SCREEN_WIDTH = 1920
    SCREEN_HEIGHT = 1080

    # Feature dimensions
    INPUT_DIM = 7  # [delta_t, delta_x, delta_y, speed, acceleration, button_encoded, state_encoded]
    CONDITION_DIM = 128  # Combined condition embedding size

    # Conditional embeddings
    USER_EMBED_DIM = 32
    TIME_PERIOD_EMBED_DIM = 16  # Morning/Afternoon/Evening/Night
    ACTION_TYPE_EMBED_DIM = 16  # Move/Click/Drag
    POSITION_DIM = 4  # [start_x, start_y, end_x, end_y] normalized

    # Model architecture
    HIDDEN_DIM = 256
    LATENT_DIM = 64
    NUM_LAYERS = 2
    DROPOUT = 0.2

    # Training parameters
    BATCH_SIZE = 1280  # Increased for better GPU utilization
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 100
    KL_WEIGHT = 1.0  # Start with full KL weight to prevent posterior collapse
    KL_ANNEAL_EPOCHS = 0  # No annealing - keep constant KL weight for stability
    GRADIENT_CLIP = 1.0

    # Generation parameters
    NUM_SAMPLES = 10  # Number of trajectories to generate per condition
    TEMPERATURE = 1.0  # Sampling temperature

    # Checkpointing
    CHECKPOINT_DIR = "checkpoints"
    SAVE_EVERY = 5  # Save checkpoint every N epochs

    # Early Stopping
    EARLY_STOPPING = True  # Enable early stopping
    EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for N epochs
    EARLY_STOPPING_MIN_DELTA = 1e-4  # Minimum change to qualify as improvement

    # Device - Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available! Please install PyTorch with CUDA support.\n"
            "Visit: https://pytorch.org/get-started/locally/\n"
            "Or run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )
    DEVICE = torch.device("cuda")

    # Random seed
    SEED = 42

    # Data augmentation
    USE_AUGMENTATION = True
    NOISE_LEVEL = 0.01

    # Time periods (in hours)
    TIME_PERIODS = {
        'morning': (6, 12),
        'afternoon': (12, 18),
        'evening': (18, 22),
        'night': (22, 6)
    }

    # Action types
    ACTION_TYPES = ['move', 'click', 'drag']

    # Button encoding
    BUTTON_MAP = {'None': 0, 'Left': 1, 'Right': 2, 'Middle': 3}

    # State encoding
    STATE_MAP = {'Move': 0, 'Pressed': 1, 'Released': 2}
