"""
Training script for Conditional LSTM-VAE
"""

import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime

from config import Config
from dataset import get_dataloaders
from model import ConditionalLSTMVAE, vae_loss


def setup_logger(timestamp):
    """
    设置日志记录器，同时输出到控制台和文件

    Args:
        timestamp: 训练开始时间戳

    Returns:
        logger: 配置好的日志记录器
    """
    # 创建logs目录
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # 创建logger
    logger = logging.getLogger('TrainingLogger')
    logger.setLevel(logging.INFO)

    # 清除已有的handlers（避免重复）
    logger.handlers.clear()

    # 文件handler - 保存所有日志
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def train_epoch(model, dataloader, optimizer, config, epoch, writer, logger=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    # KL annealing (if KL_ANNEAL_EPOCHS=0, use constant weight)
    if config.KL_ANNEAL_EPOCHS > 0:
        kl_weight = min(1.0, config.KL_WEIGHT * (epoch / config.KL_ANNEAL_EPOCHS))
    else:
        kl_weight = config.KL_WEIGHT

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (sequences, conditions) in enumerate(pbar):
        # Move to device
        sequences = sequences.to(config.DEVICE)
        for key in conditions:
            conditions[key] = conditions[key].to(config.DEVICE)

        # Forward pass
        x_recon, mu, logvar = model(sequences, conditions)

        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(x_recon, sequences, mu, logvar, kl_weight)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}',
            'kl_w': f'{kl_weight:.4f}'
        })

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)

    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Train/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Recon_Loss', avg_recon_loss, epoch)
        writer.add_scalar('Train/KL_Loss', avg_kl_loss, epoch)
        writer.add_scalar('Train/KL_Weight', kl_weight, epoch)

    return avg_loss, avg_recon_loss, avg_kl_loss


def validate(model, dataloader, config, epoch, writer, logger=None):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0

    kl_weight = min(1.0, config.KL_WEIGHT * (epoch / config.KL_ANNEAL_EPOCHS))

    with torch.no_grad():
        for sequences, conditions in tqdm(dataloader, desc="Validation"):
            # Move to device
            sequences = sequences.to(config.DEVICE)
            for key in conditions:
                conditions[key] = conditions[key].to(config.DEVICE)

            # Forward pass
            x_recon, mu, logvar = model(sequences, conditions)

            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(x_recon, sequences, mu, logvar, kl_weight)

            # Accumulate losses
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

    # Average losses
    avg_loss = total_loss / len(dataloader)
    avg_recon_loss = total_recon_loss / len(dataloader)
    avg_kl_loss = total_kl_loss / len(dataloader)

    # Log to tensorboard
    if writer is not None:
        writer.add_scalar('Val/Total_Loss', avg_loss, epoch)
        writer.add_scalar('Val/Recon_Loss', avg_recon_loss, epoch)
        writer.add_scalar('Val/KL_Loss', avg_kl_loss, epoch)

    return avg_loss, avg_recon_loss, avg_kl_loss


def save_checkpoint(model, optimizer, epoch, loss, config, filename, num_users=None):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'num_users': num_users  # Save num_users for generation
    }
    filepath = os.path.join(config.CHECKPOINT_DIR, filename)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath, config):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def main(args):
    # Set random seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Create directories
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%y%m%d%H%M")

    # Setup logger
    logger = setup_logger(timestamp)
    logger.info("="*70)
    logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Timestamp: {timestamp}")
    logger.info("="*70)

    # Initialize tensorboard
    writer = SummaryWriter('runs/clstm_vae')

    # Load data
    logger.info("Loading data...")
    train_loader, val_loader, num_users = get_dataloaders(Config, demo=args.demo)

    logger.info(f"Number of users: {num_users}")
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # Create model
    logger.info("\nInitializing model...")
    model = ConditionalLSTMVAE(Config, num_users).to(Config.DEVICE)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params:,}")

    # Log model configuration
    logger.info(f"\nModel Configuration:")
    logger.info(f"  HIDDEN_DIM: {Config.HIDDEN_DIM}")
    logger.info(f"  LATENT_DIM: {Config.LATENT_DIM}")
    logger.info(f"  NUM_LAYERS: {Config.NUM_LAYERS}")
    logger.info(f"  DROPOUT: {Config.DROPOUT}")
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  BATCH_SIZE: {Config.BATCH_SIZE}")
    logger.info(f"  LEARNING_RATE: {Config.LEARNING_RATE}")
    logger.info(f"  NUM_EPOCHS: {Config.NUM_EPOCHS}")
    logger.info(f"  KL_WEIGHT: {Config.KL_WEIGHT}")
    logger.info(f"  KL_ANNEAL_EPOCHS: {Config.KL_ANNEAL_EPOCHS}")
    logger.info(f"  EARLY_STOPPING: {Config.EARLY_STOPPING}")
    logger.info(f"  EARLY_STOPPING_PATIENCE: {Config.EARLY_STOPPING_PATIENCE}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, Config)
        start_epoch += 1

    # Training loop
    logger.info("\n" + "="*70)
    logger.info("Starting training...")
    logger.info("="*70)
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch}/{Config.NUM_EPOCHS-1}")
        logger.info(f"{'='*70}")

        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, Config, epoch, writer, logger
        )

        logger.info(f"Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f})")

        # Validate
        val_loss, val_recon, val_kl = validate(
            model, val_loader, Config, epoch, writer, logger
        )

        logger.info(f"Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})")

        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            logger.info(f"Learning rate changed: {current_lr:.6f} -> {new_lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, Config,
                f'checkpoint_epoch_{epoch}_{timestamp}.pt',
                num_users=num_users
            )

        # Save best model and check for improvement
        if val_loss < best_val_loss - Config.EARLY_STOPPING_MIN_DELTA:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model, optimizer, epoch, val_loss, Config,
                f'best_model_{timestamp}.pt',
                num_users=num_users
            )
            logger.info(f"✓ New best model saved! Val loss: {val_loss:.4f} (improved by {improvement:.4f})")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epoch(s) (best: {best_val_loss:.4f})")

        # Early stopping
        if Config.EARLY_STOPPING and epochs_without_improvement >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(f"\n{'='*70}")
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            logger.info(f"Best validation loss: {best_val_loss:.4f}")
            logger.info(f"{'='*70}")
            break

    # Save final model
    save_checkpoint(
        model, optimizer, Config.NUM_EPOCHS - 1, val_loss, Config,
        f'final_model_{timestamp}.pt',
        num_users=num_users
    )

    writer.close()
    logger.info("\n" + "="*70)
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Training ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Conditional LSTM-VAE')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--demo', action='store_true',
                       help='Demo mode: only train on user1 and user2 for quick testing')
    args = parser.parse_args()

    main(args)
