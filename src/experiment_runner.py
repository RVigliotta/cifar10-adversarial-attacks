"""
experiment_runner.py

Runner tutto-in-uno, pensato per essere eseguito direttamente (no repro_check esterno).
Contiene una funzione set_seed interna identica a quella che usi negli altri file.
"""

import os
import time
import csv
import random
from pathlib import Path

import numpy as np
import torch

# importa i moduli della repo (assicurati che i file esistano)
from src.data_loading import load_or_process_data
from src.training import train_model, train_model_adversarial
from models.model_definitions import TestCNNv2
from src.evaluation import CleanEvaluator, AttackEvaluator
from src.attacks import fgsm_attack, pgd_attack


# ---------------------------------------------------------
# LOCAL set_seed (usata qui, non dipende da repro_check)
# ---------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # rendi cuDNN deterministico (potrebbe rallentare)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------
# CONFIG DEFAULT (modifica qui per sperimentare)
# ---------------------------------------------------------
DEFAULT_SEED = 42
DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

QUICK = {
    'train_epochs': 1,
    'adv_epochs': 1,
    'batch_size': 64,
    'pgd_steps': 7,
    'pgd_alpha': 2 / 255,
    'epsilon': 0.03
}

FULL = {
    'train_epochs': 16,
    'adv_epochs': 70,
    'batch_size': 128,
    'pgd_steps': 7,
    'pgd_alpha': (2 * 0.03) / 7,
    'epsilon': 0.03
}

MODEL_DIR = Path('../models/saved_models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_MODEL_PATH = MODEL_DIR / 'test_cnn_final.pth'
FGSM_ADV_MODEL_PATH = MODEL_DIR / 'test_cnn_v2_fgsm_final.pth'
PGD_ADV_MODEL_PATH = MODEL_DIR / 'test_cnn_v2_pgd_final.pth'


# ---------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------
def device_info():
    print("Device:", DEFAULT_DEVICE)


def save_results_csv(results, outpath='experiment_results.csv'):
    if not results:
        print("No results to save.")
        return
    keys = list(results[0].keys())
    with open(outpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {outpath}")


# ---------------------------------------------------------
# STEPS
# ---------------------------------------------------------
def step_prepare_data(batch_size, num_workers=2, seed=DEFAULT_SEED):
    print("\n== STEP: prepare data ==")
    set_seed(seed)
    train_loader, test_loader = load_or_process_data(batch_size=batch_size, num_workers=num_workers)
    print("  -> train batches:", len(train_loader), " test batches:", len(test_loader))
    return train_loader, test_loader


def step_sanity_attacks(test_loader, device=DEFAULT_DEVICE, n_batches=1, eps=8 / 255):
    print("\n== STEP: attack sanity check ==")
    set_seed(DEFAULT_SEED)
    model = TestCNNv2().to(device)

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    images_denorm = images * std + mean

    print("  images_denorm range:", float(images_denorm.min()), float(images_denorm.max()))

    adv_fgsm = fgsm_attack(model, images_denorm, labels, epsilon=eps, mean=mean, std=std, device=device)
    adv_pgd = pgd_attack(model, images_denorm, labels, epsilon=eps, alpha=2 / 255, num_iter=7,
                         mean=mean, std=std, device=device)

    print("  FGSM max perturb L_inf:", float((adv_fgsm - images_denorm).abs().max()))
    print("  PGD  max perturb L_inf:", float((adv_pgd - images_denorm).abs().max()))

    model.eval()
    import torch.nn.functional as F
    with torch.no_grad():
        out_clean = model((images_denorm - mean) / std)
        out_fgsm = model((adv_fgsm - mean) / std)
        out_pgd = model((adv_pgd - mean) / std)
    print("  Loss clean:", float(F.cross_entropy(out_clean, labels).item()))
    print("  Loss fgsm:", float(F.cross_entropy(out_fgsm, labels).item()))
    print("  Loss pgd:", float(F.cross_entropy(out_pgd, labels).item()))

    return True


def step_train_clean(train_loader=None, test_loader=None):
    print("\n== STEP: train clean model ==")
    set_seed(DEFAULT_SEED)
    # usa la funzione train_model già presente in src/training.py
    train_model()
    print("  -> clean model should be saved to:", CLEAN_MODEL_PATH)
    return CLEAN_MODEL_PATH


def step_eval_model(model_path, test_loader, device=DEFAULT_DEVICE, eps=0.03):
    print(f"\n== STEP: evaluate {model_path.name} ==")
    set_seed(DEFAULT_SEED)
    device = DEFAULT_DEVICE
    model = TestCNNv2().to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    print("  Loaded model:", model_path)

    clean_eval = CleanEvaluator(device=device, use_amp=False, seed=DEFAULT_SEED)
    fgsm_eval = AttackEvaluator(attack_fn=fgsm_attack, attack_params={'epsilon': eps}, device=device, seed=DEFAULT_SEED)
    pgd_eval = AttackEvaluator(attack_fn=pgd_attack,
                               attack_params={'epsilon': eps, 'alpha': 2 / 255, 'num_iter': 7},
                               device=device, seed=DEFAULT_SEED)

    res_clean = clean_eval.evaluate(model, test_loader)
    res_fgsm = fgsm_eval.evaluate(model, test_loader)
    res_pgd = pgd_eval.evaluate(model, test_loader)

    print("  Results -> Clean:", f"{res_clean['accuracy']:.4f}",
          " FGSM:", f"{res_fgsm['accuracy']:.4f}",
          " PGD:", f"{res_pgd['accuracy']:.4f}")

    return {
        'model': model_path.name,
        'clean_acc': res_clean['accuracy'],
        'fgsm_acc': res_fgsm['accuracy'],
        'pgd_acc': res_pgd['accuracy']
    }


def step_train_adversarial(use_pgd=True):
    print("\n== STEP: adversarial training (use_pgd=%s) ==" % use_pgd)
    set_seed(DEFAULT_SEED)
    train_model_adversarial(use_pgd=use_pgd)
    return PGD_ADV_MODEL_PATH if use_pgd else FGSM_ADV_MODEL_PATH


# ---------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------
def run_full_workflow(mode='quick'):
    print("RUN MODE:", mode)
    params = QUICK if mode == 'quick' else FULL
    batch_size = params['batch_size']
    eps = params['epsilon']

    # 1) Data
    train_loader, test_loader = step_prepare_data(batch_size=batch_size)

    # 2) Sanity check attacks
    step_sanity_attacks(test_loader, device=DEFAULT_DEVICE, eps=params['epsilon'])

    results = []

    # 3) Train clean (baseline)
    step_train_clean(train_loader, test_loader)

    # 4) Eval clean model
    if CLEAN_MODEL_PATH.exists():
        r_clean = step_eval_model(CLEAN_MODEL_PATH, test_loader, device=DEFAULT_DEVICE, eps=eps)
        results.append(r_clean)
    else:
        print("Warning: clean model file not found:", CLEAN_MODEL_PATH)

    # 5) Adversarial training FGSM
    fgsm_model_path = step_train_adversarial(use_pgd=False)
    if fgsm_model_path.exists():
        r_adv_fgsm = step_eval_model(fgsm_model_path, test_loader, device=DEFAULT_DEVICE, eps=eps)
        results.append(r_adv_fgsm)
    else:
        print("Warning: FGSM adv model file not found:", fgsm_model_path)

    # 6) Adversarial training PGD
    pgd_model_path = step_train_adversarial(use_pgd=True)
    if pgd_model_path.exists():
        r_adv_pgd = step_eval_model(pgd_model_path, test_loader, device=DEFAULT_DEVICE, eps=eps)
        results.append(r_adv_pgd)
    else:
        print("Warning: PGD adv model file not found:", pgd_model_path)

    # Save results
    save_results_csv(results, outpath='experiment_results.csv')
    return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    device_info()
    # scegli 'quick' per debug, 'full' per training reale
    results = run_full_workflow(mode='quick')
    print("\nFinal Results Summary:")
    for r in results:
        print(r)


if __name__ == '__main__':
    main()
