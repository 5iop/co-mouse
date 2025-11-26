"""
Test data format and GPU usage
"""

import torch
import numpy as np
from config import Config
from dataset import get_dataloaders
from model import ConditionalLSTMVAE, vae_loss


def print_section(title):
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def test_gpu():
    """Test GPU availability"""
    print_section("1. GPU Configuration")

    try:
        print(f"✓ CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA Version: {torch.version.cuda}")
            print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"✓ Config.DEVICE: {Config.DEVICE}")
            return True
    except Exception as e:
        print(f"✗ GPU Error: {e}")
        return False


def test_dataloader():
    """Test dataloader and data format"""
    print_section("2. Data Loading Test")

    # Use quick mode for testing
    config = Config()
    original_max = config.MAX_SESSIONS_PER_USER
    config.MAX_SESSIONS_PER_USER = 5  # Quick test

    print("Loading small dataset for testing...")
    train_loader, val_loader, num_users = get_dataloaders(config)

    print(f"\n✓ Data loaded successfully!")
    print(f"  - Number of users: {num_users}")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Validation batches: {len(val_loader)}")

    # Restore original setting
    config.MAX_SESSIONS_PER_USER = original_max

    return train_loader, num_users


def test_batch_format(train_loader):
    """Test batch data format"""
    print_section("3. Batch Format Test")

    batch_sequences, batch_conditions = next(iter(train_loader))

    print(f"\n✓ Batch retrieved successfully!")
    print(f"\nSequences:")
    print(f"  - Type: {type(batch_sequences)}")
    print(f"  - Shape: {batch_sequences.shape}")
    print(f"  - Dtype: {batch_sequences.dtype}")
    print(f"  - Device: {batch_sequences.device}")
    print(f"  - Min value: {batch_sequences.min().item():.4f}")
    print(f"  - Max value: {batch_sequences.max().item():.4f}")
    print(f"  - Mean: {batch_sequences.mean().item():.4f}")

    print(f"\nConditions:")
    for key, value in batch_conditions.items():
        print(f"  - {key}:")
        print(f"      Type: {type(value)}")
        print(f"      Shape: {value.shape}")
        print(f"      Dtype: {value.dtype}")
        print(f"      Device: {value.device}")
        if value.dtype in [torch.long, torch.int]:
            print(f"      Values: {value.squeeze().tolist()[:5]}...")  # First 5

    # Validate expected format
    config = Config()
    batch_size = batch_sequences.shape[0]
    seq_len = batch_sequences.shape[1]
    input_dim = batch_sequences.shape[2]

    assert batch_sequences.shape == (batch_size, config.SEQUENCE_LENGTH, config.INPUT_DIM), \
        f"Sequence shape mismatch: expected ({batch_size}, {config.SEQUENCE_LENGTH}, {config.INPUT_DIM})"

    assert batch_conditions['user_id'].shape == (batch_size, 1), \
        f"User ID shape mismatch"

    assert batch_conditions['time_period'].shape == (batch_size, 1), \
        f"Time period shape mismatch"

    assert batch_conditions['action_type'].shape == (batch_size, 1), \
        f"Action type shape mismatch"

    assert batch_conditions['start_pos'].shape == (batch_size, 2), \
        f"Start position shape mismatch"

    assert batch_conditions['end_pos'].shape == (batch_size, 2), \
        f"End position shape mismatch"

    print(f"\n✓ All format checks PASSED!")

    return batch_sequences, batch_conditions


def test_gpu_transfer(batch_sequences, batch_conditions):
    """Test GPU memory transfer"""
    print_section("4. GPU Transfer Test")

    config = Config()

    # Check initial GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        print(f"Initial GPU memory: {initial_memory:.2f} MB")

    # Move to GPU
    print("\nMoving data to GPU...")
    sequences_gpu = batch_sequences.to(config.DEVICE)
    conditions_gpu = {k: v.to(config.DEVICE) for k, v in batch_conditions.items()}

    print(f"✓ Sequences on: {sequences_gpu.device}")
    print(f"✓ Conditions on: {conditions_gpu['user_id'].device}")

    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"\nCurrent GPU memory: {current_memory:.2f} MB")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")
        print(f"Memory increase: {current_memory - initial_memory:.2f} MB")

    return sequences_gpu, conditions_gpu


def test_model_forward(sequences_gpu, conditions_gpu, num_users):
    """Test model forward pass on GPU"""
    print_section("5. Model Forward Pass Test")

    config = Config()

    print("Creating model...")
    model = ConditionalLSTMVAE(config, num_users).to(config.DEVICE)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {num_params:,} parameters")
    print(f"✓ Model on device: {next(model.parameters()).device}")

    if torch.cuda.is_available():
        model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        print(f"✓ Model size: {model_memory:.2f} MB")

        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"\nGPU memory before forward: {initial_memory:.2f} MB")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        x_recon, mu, logvar = model(sequences_gpu, conditions_gpu)

    print(f"✓ Forward pass successful!")
    print(f"  - Reconstruction shape: {x_recon.shape}")
    print(f"  - Mu shape: {mu.shape}")
    print(f"  - Logvar shape: {logvar.shape}")
    print(f"  - Reconstruction on: {x_recon.device}")

    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**2
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nGPU memory after forward: {current_memory:.2f} MB")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")

    return model, x_recon, mu, logvar


def test_backward_pass(model, sequences_gpu, conditions_gpu):
    """Test backward pass and optimizer"""
    print_section("6. Backward Pass Test")

    config = Config()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory before backward: {initial_memory:.2f} MB")

    # Forward
    print("Running forward + backward pass...")
    x_recon, mu, logvar = model(sequences_gpu, conditions_gpu)

    # Loss
    loss, recon_loss, kl_loss = vae_loss(x_recon, sequences_gpu, mu, logvar, 0.001)

    print(f"✓ Loss computed:")
    print(f"  - Total loss: {loss.item():.4f}")
    print(f"  - Recon loss: {recon_loss.item():.4f}")
    print(f"  - KL loss: {kl_loss.item():.4f}")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Backward pass successful!")

    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**2
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nGPU memory after backward: {current_memory:.2f} MB")
        print(f"Peak GPU memory: {peak_memory:.2f} MB")

    # Verify gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"✓ Gradients computed: {has_grads}/{total_params} parameters")


def test_generation(model, conditions_gpu):
    """Test trajectory generation"""
    print_section("7. Generation Test")

    config = Config()

    print("Generating trajectories...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(conditions_gpu, config.SEQUENCE_LENGTH, temperature=1.0)

    print(f"✓ Generation successful!")
    print(f"  - Generated shape: {generated.shape}")
    print(f"  - Generated device: {generated.device}")
    print(f"  - Generated dtype: {generated.dtype}")
    print(f"  - Min value: {generated.min().item():.4f}")
    print(f"  - Max value: {generated.max().item():.4f}")


def main():
    print("="*60)
    print("COMPREHENSIVE DATA FORMAT & GPU TEST")
    print("="*60)

    try:
        # Test 1: GPU
        gpu_ok = test_gpu()
        if not gpu_ok:
            print("\n⚠ GPU test failed, but continuing...")

        # Test 2: Data loading
        train_loader, num_users = test_dataloader()

        # Test 3: Batch format
        batch_sequences, batch_conditions = test_batch_format(train_loader)

        # Test 4: GPU transfer
        sequences_gpu, conditions_gpu = test_gpu_transfer(batch_sequences, batch_conditions)

        # Test 5: Model forward
        model, x_recon, mu, logvar = test_model_forward(sequences_gpu, conditions_gpu, num_users)

        # Test 6: Backward pass
        test_backward_pass(model, sequences_gpu, conditions_gpu)

        # Test 7: Generation
        test_generation(model, conditions_gpu)

        # Summary
        print_section("✓ ALL TESTS PASSED!")
        print("\nSummary:")
        print("  ✓ GPU configuration correct")
        print("  ✓ Data format correct")
        print("  ✓ GPU memory transfer working")
        print("  ✓ Model forward pass working")
        print("  ✓ Backward pass working")
        print("  ✓ Generation working")
        print("\nYou're ready to train!")
        print(f"{'='*60}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ TEST FAILED")
        print(f"{'='*60}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
