import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.model_definitions import TestCNNv2
from src.attacks import fgsm_attack, pgd_attack
from src.data_loading import load_or_process_data


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation: combines two batches with a random factor λ."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def load_processed_data(batch_size=64):
    """
    Loads preprocessed datasets (train.pt, test.pt) if available,
    otherwise regenerates CIFAR-10 datasets and saves them for future use.
    Automatically handles compatibility with PyTorch 2.6+.
    """
    data_dir = '../data/processed'
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, 'train.pt')
    test_path = os.path.join(data_dir, 'test.pt')

    # Definition of standard transformations for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # If .pt files exist, try loading them
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading data from local cache...")

        try:
            #Try loading with PyTorch 2.6+ safe mode
            with torch.serialization.safe_globals([datasets.CIFAR10]):
                train_set = torch.load(train_path, map_location='cpu', weights_only=False)
                test_set = torch.load(test_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"Safe load failed ({e}), regenerating datasets...")
            train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
            test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
            torch.save(train_set, train_path)
            torch.save(test_set, test_path)

    else:
        print("Downloading and preparing CIFAR-10 datasets...")
        train_set = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        torch.save(train_set, train_path)
        torch.save(test_set, test_path)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"  -> train batches: {len(train_loader)}  test batches: {len(test_loader)}")
    return train_loader, test_loader


def train_model():
    """Standard training (no attacks)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 1
    batch_size = 64

    model = TestCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_or_process_data(batch_size=batch_size)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            images, labels_a, labels_b, lam = mixup_data(images, labels)

            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        acc = 100 * correct / total

        print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {running_loss / len(train_loader):.4f} '
              f'| Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}%')

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('../models/saved_models', exist_ok=True)
            torch.save(model.state_dict(), '../models/saved_models/test_cnn_final.pth')
            print(f'--> Saved best model (Val Loss: {val_loss:.4f})')


def train_model_adversarial(use_pgd=True):
    """
    Adversarial training with FGSM or PGD.
    - use_pgd=True  -> PGD attack
    - use_pgd=False -> FGSM attack
    """
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")

    num_epochs = 1
    batch_size = 128
    epsilon = 0.03
    num_iter = 7
    pgd_alpha = (2 * epsilon) / num_iter

    # Mean and std of the dataset (e.g., standard CIFAR10)
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    model = TestCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_processed_data(batch_size)
    best_val_loss = float('inf')

    model_name = 'test_cnn_v2_pgd_final.pth' if use_pgd else 'test_cnn_v2_fgsm_final.pth'
    model_path = f'../models/saved_models/{model_name}'

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # 50% chance to use adversarial images
            if torch.rand(1).item() < 0.5:
                # Denormalize images to generate perturbations
                images_denorm = images * std + mean

                model.eval()  # model in eval mode to generate attack
                with torch.enable_grad():
                    if use_pgd:
                        adv_images_denorm = pgd_attack(
                            model, images_denorm, labels,
                            epsilon=epsilon, alpha=pgd_alpha, num_iter=num_iter, mean=mean, std=std, device=device
                        )
                    else:
                        adv_images_denorm = fgsm_attack(
                            model, images_denorm, labels,
                            epsilon=epsilon, mean=mean, std=std, device=device
                        )
                # Return to normalized space
                inputs = (adv_images_denorm - mean) / std
            else:
                inputs = images

            # Training step
            model.train()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(test_loader)
        acc = 100 * correct / total
        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {running_loss / len(train_loader):.4f} '
              f'| Val Loss: {val_loss:.4f} | Val Acc: {acc:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('../models/saved_models', exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f'--> Saved best {model_name} (Val Loss: {val_loss:.4f})')

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds.")


if __name__ == '__main__':
    train_model_adversarial(use_pgd=True)
