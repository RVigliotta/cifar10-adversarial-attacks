import torchvision
import torchvision.transforms as transforms


def download_cifar10():
    try:
        print("Starting dataset download...")
        transform = transforms.Compose([transforms.ToTensor()])
        train_set = torchvision.datasets.CIFAR10(root='../data/raw', train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='../data/raw', train=False, download=True, transform=transform)
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error during the download: {e}")


download_cifar10()
