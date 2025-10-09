import os
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from evaluation import CleanEvaluator, AttackEvaluator, set_seed
from models.model_definitions import TestCNNv2
from src.attacks import fgsm_attack, pgd_attack


def load_test_dataset(processed_path='../data/processed/test.pt', raw_path='../data/processed/test_raw.pt'):
    """
    Robustly loads the test set.
    - If ../data/processed/test.pt is a Dataset object (e.g., torchvision CIFAR10), it uses it directly.
    - If not available, tries to load test_raw.pt (dict with 'data' and 'targets') and builds a TensorDataset.
    """
    if os.path.exists(processed_path):
        try:
            ds = torch.load(processed_path)
            # if it's a dataset object (has __len__ and __getitem__), use it directly
            if hasattr(ds, '__len__') and hasattr(ds, '__getitem__'):
                return ds
            # otherwise if it's a dict, build a TensorDataset
            if isinstance(ds, dict) and 'data' in ds and 'targets' in ds:
                data = torch.tensor(ds['data']).permute(0, 3, 1, 2).float().div(255.0)  # HWC->CHW and scale to [0,1]
                targets = torch.tensor(ds['targets']).long()
                return TensorDataset(data, targets)
        except Exception as e:
            print(f"Warning: cannot load {processed_path}: {e}")

    # fallback: try raw dict
    if os.path.exists(raw_path):
        try:
            ds = torch.load(raw_path)
            if isinstance(ds, dict) and 'data' in ds and 'targets' in ds:
                data = torch.tensor(ds['data']).permute(0, 3, 1, 2).float().div(255.0)
                targets = torch.tensor(ds['targets']).long()
                return TensorDataset(data, targets)
        except Exception as e:
            print(f"Warning: cannot load {raw_path}: {e}")

    raise FileNotFoundError(
        "No suitable test dataset found. Run data_loading.py to create ../data/processed/test.pt or test_raw.pt")


def plot_results(results, outpath='../docs/robustness_comparison.png'):
    models = ['Clean Model', 'Adversarial Model']
    colors = ['#1f77b4', '#ff7f0e']

    clean_clean = results['Clean Model (Clean)']['accuracy']
    clean_attacked = results['Clean Model (Attacked)']['accuracy']
    adv_clean = results['Adversarial Model (Clean)']['accuracy']
    adv_attacked = results['Adversarial Model (Attacked)']['accuracy']

    values = np.array([[clean_clean, clean_attacked],
                       [adv_clean, adv_attacked]])

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, values[:, 0], width, label='Clean Data', color=colors[0], edgecolor='black')
    rects2 = ax.bar(x + width / 2, values[:, 1], width, label='Attacked Data', color=colors[1], edgecolor='black')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Model Robustness Comparison (FGSM Attack, ε=0.03)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.show()


def main():
    # seed & device
    seed = 42
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}, seed: {seed}")

    # load test dataset
    try:
        test_ds = load_test_dataset()
    except FileNotFoundError as e:
        print(e)
        return

    # If dataset is a TensorDataset with unnormalized [0,1], we must normalize for model
    # Decide expected normalization: use mean/std=(0.5,0.5,0.5)
    # If the dataset items are PIL/images with transforms already applied, they will be normalized by the dataset.
    # We'll create a DataLoader that returns normalized images (model expects normalized inputs).
    def collate_fn(batch):
        # batch elements might be (tensor_image, label) or (PIL converted via dataset, label)
        imgs, labs = zip(*batch)
        imgs = torch.stack([img if img.max() > 1.1 else img for img in imgs], dim=0)  # heuristic
        # If imgs are in [0,1], normalize to [-1,1] using mean=0.5,std=0.5
        mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std
        labs = torch.tensor(labs).long()
        return imgs, labs

    # build DataLoader (small batch size for quick evaluation)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # load models (robustly try multiple possible adv filenames)
    model_clean = TestCNNv2().to(device)
    model_adv = TestCNNv2().to(device)
    clean_path = '../models/saved_models/test_cnn_v2.pth'
    adv_candidates = [
        '../models/saved_models/test_cnn_v2_pgd.pth',
        '../models/saved_models/test_cnn_v2_fgsm.pth',
        '../models/saved_models/test_cnn_v2_adv.pth'
    ]

    try:
        model_clean.load_state_dict(torch.load(clean_path, map_location=device))
        print(f"Loaded clean model from {clean_path}")
    except Exception as e:
        print(f"Error loading clean model ({clean_path}): {e}")
        return

    adv_loaded = False
    for p in adv_candidates:
        if os.path.exists(p):
            try:
                model_adv.load_state_dict(torch.load(p, map_location=device))
                print(f"Loaded adversarial model from {p}")
                adv_loaded = True
                break
            except Exception as e:
                print(f"Found {p} but failed to load: {e}")

    if not adv_loaded:
        print("No adversarial model found among candidates; using clean model for adv evaluation (not ideal).")
        model_adv.load_state_dict(torch.load(clean_path, map_location=device))

    # evaluators (attack epsilon consistent with training: 0.03)
    clean_eval = CleanEvaluator(device=device, use_amp=False, seed=seed)
    attack_eval = AttackEvaluator(attack_fn=fgsm_attack, attack_params={'epsilon': 0.03}, device=device, seed=seed)

    results = {}

    print("\nEvaluating Clean Model (clean test set)...")
    results['Clean Model (Clean)'] = clean_eval.evaluate(model_clean, test_loader)

    print("\nEvaluating Clean Model (FGSM attacked)...")
    results['Clean Model (Attacked)'] = attack_eval.evaluate(model_clean, test_loader)

    print("\nEvaluating Adversarial Model (clean test set)...")
    results['Adversarial Model (Clean)'] = clean_eval.evaluate(model_adv, test_loader)

    print("\nEvaluating Adversarial Model (FGSM attacked)...")
    results['Adversarial Model (Attacked)'] = attack_eval.evaluate(model_adv, test_loader)

    print("\nFinal Results:")
    for k, v in results.items():
        print(f"{k}: {v['accuracy']:.2%}")

    plot_results(results)


if __name__ == '__main__':
    main()
