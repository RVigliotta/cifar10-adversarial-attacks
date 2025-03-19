import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from models.model_definitions import TestCNNv2
from evaluation import CleanEvaluator, AttackEvaluator
from attacks import fgsm_attack


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def plot_results(results):
    models = list(results.keys())
    accuracies = [res['accuracy'] for res in results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(models, accuracies, color=['blue', 'red', 'green'])
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.savefig('../docs/results.png')
    plt.show()


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_set = torch.load('../data/processed/train.pt', weights_only=False)
    test_set = torch.load('../data/processed/test.pt', weights_only=False)

    train_dataset = CustomDataset(train_set)
    test_dataset = CustomDataset(test_set)

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = TestCNNv2().to(device)
    model.load_state_dict(torch.load('../models/saved_models/test_cnn_v2.pth', map_location=device))

    clean_evaluator = CleanEvaluator(device=device)
    clean_metrics = clean_evaluator.evaluate(model, test_loader)
    print(f"Clean Accuracy: {clean_metrics['accuracy']:.2%}")

    attack_evaluator = AttackEvaluator(
        attack_fn=fgsm_attack,
        attack_params={'epsilon': 0.03},
        device=device
    )
    attack_metrics = attack_evaluator.evaluate(model, test_loader)
    print(f"FGSM Attack Accuracy: {attack_metrics['accuracy']:.2%}")

    results = {
        'Clean Model': clean_metrics,
        'FGSM Attack': attack_metrics
    }
    plot_results(results)


if __name__ == '__main__':
    main()
