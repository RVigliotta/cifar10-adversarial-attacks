import torchvision
import torchvision.transforms as transforms
import torch


def download_cifar10():
    try:
        print("Starting dataset download...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='../data/raw', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='../data/raw', train=False, download=True, transform=transform)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error during the download: {e}")


def preprocess_data():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.CIFAR10(root='../data/raw', train=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='../data/raw', train=False, transform=transform)
    torch.save(train_set, '../data/processed/train.pt')
    torch.save(test_set, '../data/processed/test.pt')


download_cifar10()
preprocess_data()
