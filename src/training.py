import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader
from models.model_definitions import TestCNNv2
from src.attacks import fgsm_attack


def mixup_data(x, y, alpha=0.4):
    """Mixup augmentation."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def load_processed_data(batch_size=64):
    try:
        # Load preprocessed datasets
        train_set = torch.load('../data/processed/train.pt', weights_only=False)
        test_set = torch.load('../data/processed/test.pt', weights_only=False)

        # Create DataLoaders
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    except FileNotFoundError:
        raise RuntimeError("Preprocessed data not found. Run data_loading.py first")


def train_model():
    # Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 14
    batch_size = 64

    # Initialize Model, Optimizer and Loss function
    model = TestCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()

    # Load data
    train_loader, test_loader = load_processed_data(batch_size)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Mixup augmentation
            images, labels_a, labels_b, lam = mixup_data(images, labels)

            # Forward pass
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Epoch evaluation on validation set
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

        # Print metrics
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('../models/saved_models', exist_ok=True)
            torch.save(model.state_dict(), '../models/saved_models/test_cnn_v2.pth')
            print(f'--> Best model saved with Val Loss: {val_loss:.4f}')


def train_model_adversarial():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 14
    batch_size = 64
    epsilon = 0.03

    model = TestCNNv2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader = load_processed_data(batch_size)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Generation of adversarial examples with 50% probability
            if torch.rand(1) < 0.5:
                model.eval()

                # Cacolate the adversarial examples keeping the gradients
                with torch.enable_grad():
                    adv_images = fgsm_attack(model, images, labels, epsilon)

                model.train()
                inputs = adv_images
            else:
                inputs = images

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

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
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Accuracy: {100 * correct / total:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), '../models/saved_models/test_cnn_v2_adv.pth')
            print(f'--> Best model saved with Val Loss: {val_loss:.4f}')


if __name__ == '__main__':
    train_model_adversarial()
