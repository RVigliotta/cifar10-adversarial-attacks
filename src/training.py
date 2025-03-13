import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model_definitions import TestCNN
import os


def load_processed_data(batch_size=32):
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
    num_epochs = 10
    batch_size = 32

    # Initialize Model, Optimizer and Loss function
    model = TestCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Load data
    train_loader, test_loader = load_processed_data(batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # Epoch evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%')

    # Save model
    os.makedirs('../models/saved_models', exist_ok=True)
    torch.save(model.state_dict(), '../models/saved_models/test_cnn.pth')
    print('Training completed and model saved!')


if __name__ == '__main__':
    train_model()
