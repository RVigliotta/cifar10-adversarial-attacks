import torchvision
import torchvision.transforms as transforms
import torch
import os


def download_cifar10():
    """Download CIFAR-10 dataset in raw directory"""
    try:
        print("Downloading CIFAR-10...")
        transform = transforms.Compose([transforms.ToTensor()])
        torchvision.datasets.CIFAR10(
            root='../data/raw',
            train=True,
            download=True,
            transform=transform
        )
        torchvision.datasets.CIFAR10(
            root='../data/raw',
            train=False,
            download=True,
            transform=transform
        )
        print("Download completed!")
    except Exception as e:
        print(f"Error during download: {e}")


def preprocess_data():
    """Preprocess e salva i dati nella directory processed"""
    # Crea le directory se non esistono
    os.makedirs('../data/processed', exist_ok=True)

    # Trasformazioni
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Carica i dataset
    train_set = torchvision.datasets.CIFAR10(
        root='../data/raw',
        train=True,
        transform=train_transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root='../data/raw',
        train=False,
        transform=test_transform
    )

    # Salva solo i dati e le etichette
    torch.save(train_set, '../data/processed/train.pt')
    torch.save(test_set, '../data/processed/test.pt')


if __name__ == '__main__':
    download_cifar10()
    preprocess_data()
