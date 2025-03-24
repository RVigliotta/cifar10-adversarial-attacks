import matplotlib.pyplot as plt
import numpy as np
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
    models = ['Clean Model', 'Adversarial Model']
    categories = ['Clean Data', 'Attacked Data']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    clean_clean = results['Clean Model (Clean)']['accuracy']
    clean_attacked = results['Clean Model (Attacked)']['accuracy']
    adv_clean = results['Adversarial Model (Clean)']['accuracy']
    adv_attacked = results['Adversarial Model (Attacked)']['accuracy']

    values = np.array([[clean_clean, clean_attacked],
                       [adv_clean, adv_attacked]])

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width / 2, values[:, 0], width, label='Clean Data', color=colors[0], edgecolor='black')
    rects2 = ax.bar(x + width / 2, values[:, 1], width, label='Attacked Data', color=colors[1], edgecolor='black')

    # Aggiungi testo, etichette e formattazione
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('Model Robustness Comparison\n(FGSM Attack, ε=0.03)', fontsize=16, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(fontsize=12, framealpha=0.9)

    # Aggiungi valori sulle barre
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=11)

    autolabel(rects1)
    autolabel(rects2)

    # Aggiungi una legenda per i colori
    plt.tight_layout()
    plt.savefig('../docs/robustness_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Caricamento dati
    test_set = torch.load('../data/processed/test.pt', weights_only=False)
    test_dataset = CustomDataset(test_set)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Caricamento modelli
    model_clean = TestCNNv2().to(device)
    model_adv = TestCNNv2().to(device)

    try:
        model_clean.load_state_dict(torch.load('../models/saved_models/test_cnn_v2.pth', map_location=device))
        model_adv.load_state_dict(torch.load('../models/saved_models/test_cnn_v2_adv.pth', map_location=device))
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        return

    # Inizializzazione evaluator
    clean_evaluator = CleanEvaluator(device=device)
    attack_evaluator = AttackEvaluator(
        attack_fn=fgsm_attack,
        attack_params={'epsilon': 0.03},
        device=device
    )

    # Valutazioni complete
    results = {}

    print("\nEvaluating Clean Model:")
    results['Clean Model (Clean)'] = clean_evaluator.evaluate(model_clean, test_loader)
    results['Clean Model (Attacked)'] = attack_evaluator.evaluate(model_clean, test_loader)

    print("\nEvaluating Adversarial Model:")
    results['Adversarial Model (Clean)'] = clean_evaluator.evaluate(model_adv, test_loader)
    results['Adversarial Model (Attacked)'] = attack_evaluator.evaluate(model_adv, test_loader)

    # Stampa risultati
    print("\nFinal Results:")
    for key, value in results.items():
        print(f"{key}: {value['accuracy']:.2%}")

    plot_results(results)


if __name__ == '__main__':
    main()
