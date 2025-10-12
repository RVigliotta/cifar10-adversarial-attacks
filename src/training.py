import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import random
import sys
from pathlib import Path
from models.model_definitions import TestCNNv2
from attacks import fgsm_attack, pgd_attack
from data_loading import load_or_process_data

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.4):
    """Apply MixUp data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, filepath, is_best=False):
    """Save a complete model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'is_best': is_best
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load a complete checkpoint"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))


def train_clean_model(num_epochs=20, batch_size=64, resume_from=None):
    """
    Train a clean model (without adversarial training)

    Args:
        num_epochs: Number of training epochs
        batch_size: Batch size
        resume_from: Path to checkpoint to resume training from
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Clean Model on {device}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}")

    # Model and component initialization
    model = TestCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    train_loader, test_loader = load_or_process_data(batch_size=batch_size)

    # Setup directories and training variables
    MODEL_DIR = Path('../models/saved_models')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume training if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch, best_val_loss = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch += 1  # Start from next epoch

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Apply MixUp
            images, y_a, y_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        val_loss /= len(test_loader)
        train_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        epoch_time = time.time() - epoch_start_time

        # Update scheduler
        scheduler.step()

        # Logging
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}%")

        # Save checkpoint
        current_lr = scheduler.get_last_lr()[0]
        checkpoint_path = MODEL_DIR / f'test_cnn_clean_epoch_{epoch + 1}.ckpt'
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = MODEL_DIR / 'test_cnn_clean.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best clean model (Val Loss: {val_loss:.4f}, LR: {current_lr:.6f})")

        print("-" * 60)

    print(f"Clean training completed! Best validation loss: {best_val_loss:.4f}")


def train_adversarial_model(use_pgd=True, num_epochs=None, batch_size=128,
                            epsilon=0.03, num_iter=7, resume_from=None):
    """
    Train a model with adversarial training

    Args:
        use_pgd: If True use PGD, otherwise FGSM
        num_epochs: Number of epochs (default: 40 for FGSM, 60 for PGD)
        batch_size: Batch size
        epsilon: Perturbation magnitude
        num_iter: Number of iterations for PGD
        resume_from: Path to checkpoint to resume training from
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default parameters
    if num_epochs is None:
        num_epochs = 60 if use_pgd else 40

    attack_name = "PGD" if use_pgd else "FGSM"
    print(f"Adversarial Training ({attack_name}) on {device}")
    print(f"Epochs: {num_epochs}, Batch Size: {batch_size}, Epsilon: {epsilon}")

    # Setup attack parameters
    pgd_alpha = (2 * epsilon) / num_iter if use_pgd else None
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    # Model and component initialization
    model = TestCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # Data loading
    train_loader, test_loader = load_or_process_data(batch_size=batch_size)

    # Setup directories and variables
    MODEL_DIR = Path('../models/saved_models')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_suffix = 'pgd' if use_pgd else 'fgsm'
    model_name = f'test_cnn_v2_{model_suffix}'

    start_epoch = 0
    best_val_loss = float('inf')

    # Resume training if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        start_epoch, best_val_loss = load_checkpoint(resume_from, model, optimizer, scheduler)
        start_epoch += 1  # Start from next epoch

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial examples with 50% probability
            if torch.rand(1).item() < 0.5:
                images_denorm = images * std + mean
                model.eval()
                with torch.enable_grad():
                    if use_pgd:
                        adv_images_denorm = pgd_attack(
                            model, images_denorm, labels,
                            epsilon=epsilon, alpha=pgd_alpha, num_iter=num_iter,
                            mean=mean, std=std, device=device
                        )
                    else:
                        adv_images_denorm = fgsm_attack(
                            model, images_denorm, labels,
                            epsilon=epsilon, mean=mean, std=std, device=device
                        )
                inputs = (adv_images_denorm - mean) / std
            else:
                inputs = images

            # Forward pass and optimization
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        val_loss /= len(test_loader)
        train_loss = running_loss / len(train_loader)
        acc = 100 * correct / total
        epoch_time = time.time() - epoch_start_time

        # Update scheduler
        scheduler.step()

        # Logging
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}%")

        # Save checkpoint
        current_lr = scheduler.get_last_lr()[0]
        checkpoint_path = MODEL_DIR / f'{model_name}_epoch_{epoch + 1}.ckpt'
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = MODEL_DIR / f'{model_name}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best {attack_name} model (Val Loss: {val_loss:.4f}, LR: {current_lr:.6f})")

        print("-" * 60)

    total_time = time.time() - start_time
    print(f"{attack_name} training completed in {total_time:.2f}s!")
    print(f"Best validation loss: {best_val_loss:.4f}")
