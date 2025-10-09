import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

# --- Caching directory ---
PROCESSED_DIR = "../data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_or_process_data(batch_size=128, num_workers=2):
    """
    Downloads CIFAR-10, normalizes it, and creates DataLoaders.
    If processed files exist, loads them directly.
    Also saves "raw" versions as tensors for robust caching.
    """

    train_raw_path = os.path.join(PROCESSED_DIR, "train_raw.pt")
    test_raw_path = os.path.join(PROCESSED_DIR, "test_raw.pt")

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # --- If already saved, load from file ---
    if os.path.exists(train_raw_path) and os.path.exists(test_raw_path):
        print("🔁 Loading data from local cache...")
        train_dict = torch.load(train_raw_path, map_location="cpu")
        test_dict = torch.load(test_raw_path, map_location="cpu")

        train_data = train_dict["data"]
        train_targets = train_dict["targets"]
        test_data = test_dict["data"]
        test_targets = test_dict["targets"]

    else:
        print("⬇️  Downloading CIFAR-10 from torchvision...")
        train_dataset = datasets.CIFAR10(root="../data", train=True, download=True)
        test_dataset = datasets.CIFAR10(root="../data", train=False, download=True)

        # Convert to tensors
        train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2).float() / 255.0
        test_data = torch.tensor(test_dataset.data).permute(0, 3, 1, 2).float() / 255.0
        train_targets = torch.tensor(train_dataset.targets, dtype=torch.long)
        test_targets = torch.tensor(test_dataset.targets, dtype=torch.long)

        # Robust saving as dictionary
        torch.save({"data": train_data, "targets": train_targets}, train_raw_path)
        torch.save({"data": test_data, "targets": test_targets}, test_raw_path)
        print("💾 Dataset saved in TensorDataset-compatible format.")

    # --- Apply transformation (normalization) ---
    train_tensor_dataset = TensorDataset(
        transforms.Normalize(mean, std)(train_data),
        train_targets
    )
    test_tensor_dataset = TensorDataset(
        transforms.Normalize(mean, std)(test_data),
        test_targets
    )

    # --- DataLoader ---
    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_tensor_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

