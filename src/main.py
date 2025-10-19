"""
main.py
Runs training or evaluation of models (Clean, FGSM, PGD) on CIFAR-10.
Now includes interpretability/explainability analysis (Grad-CAM, Saliency, and perturbation visualization).
"""

import random
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from data_loading import load_or_process_data
from training import train_clean_model, train_adversarial_model
from evaluation import (
    CleanEvaluator,
    AttackEvaluator,
    plot_model_comparison,
    plot_robustness_analysis,
    visualize_interpretability,
)
from attacks import fgsm_attack, pgd_attack
from models.model_definitions import TestCNNv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models"
DOCS_DIR = PROJECT_ROOT / "docs" / "evaluation_results"

clean_model_path = MODELS_DIR / "test_cnn_clean.pth"
fgsm_model_path = MODELS_DIR / "test_cnn_fgsm.pth"
pgd_model_path = MODELS_DIR / "test_cnn_pgd.pth"


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device_info():
    """Display device information and return device object"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    return device


def prepare_data(batch_size=128):
    """Prepare and load CIFAR-10 dataset"""
    set_seed(42)
    train_loader, test_loader = load_or_process_data(batch_size=batch_size)
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    return train_loader, test_loader


def evaluate_models(test_loader, device=None, epsilon=0.03):
    """Evaluate three models (Clean, FGSM, PGD) with detailed metrics"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    clean_evaluator = CleanEvaluator(device=device)
    fgsm_evaluator = AttackEvaluator(attack_fn=fgsm_attack, attack_params={"epsilon": epsilon}, device=device)
    pgd_evaluator = AttackEvaluator(
        attack_fn=pgd_attack, attack_params={"epsilon": epsilon, "alpha": 2 / 255, "num_iter": 7}, device=device
    )

    model_configs = [
        ("Clean", clean_model_path),
        ("FGSM", fgsm_model_path),
        ("PGD", pgd_model_path),
    ]

    for model_name, model_path in model_configs:
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        print(f"\nEvaluating {model_name} model...")
        model = TestCNNv2().to(device)

        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        # --- Evaluate ---
        clean_metrics = clean_evaluator.evaluate(model, test_loader)
        fgsm_metrics = fgsm_evaluator.evaluate(model, test_loader, clean_metrics=clean_metrics)
        pgd_metrics = pgd_evaluator.evaluate(model, test_loader, clean_metrics=clean_metrics)

        results.append({
            "model": model_name,
            "clean_acc": clean_metrics.get("accuracy"),
            "fgsm_acc": fgsm_metrics.get("adv_accuracy"),
            "pgd_acc": pgd_metrics.get("adv_accuracy"),
            "ASR": fgsm_metrics.get("ASR", np.nan),
            "Fooling Rate": fgsm_metrics.get("Fooling Rate", np.nan),
            "Robustness": fgsm_metrics.get("Robustness", np.nan),
            "Distortion (L2)": fgsm_metrics.get("Distortion (L2)", np.nan),
            "Precision": clean_metrics.get("precision"),
            "Recall": clean_metrics.get("recall"),
            "F1": clean_metrics.get("f1"),
            "Evaluation Time (s)": clean_metrics.get("total_time"),
        })

        print(f"Metrics collected for: {model_name}")

    # Save + Plot
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = DOCS_DIR / "evaluation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCombined metrics saved to: {csv_path}")

    plot_model_comparison(results, save_dir=DOCS_DIR)
    plot_robustness_analysis(results, save_dir=DOCS_DIR)

    return results


def run_complete_training_pipeline():
    """Train clean, FGSM-trained, and PGD-trained models"""
    print("Starting Complete Training Pipeline")
    print("=" * 50)

    print("\nTraining Clean Model...")
    train_clean_model(num_epochs=20, batch_size=64)

    print("\nTraining FGSM Adversarial Model...")
    train_adversarial_model(use_pgd=False, num_epochs=40, batch_size=128)

    print("\nTraining PGD Adversarial Model...")
    train_adversarial_model(use_pgd=True, num_epochs=60, batch_size=128)

    print("\nTraining pipeline completed!")


def main():
    """Main execution pipeline"""
    device = device_info()
    set_seed(42)

    # --- Step 1: Prepare Data ---
    train_loader, test_loader = prepare_data(batch_size=128)

    # --- Step 2: Training (optional if models already exist) ---
    run_complete_training_pipeline()

    # --- Step 3: Evaluate Models ---
    print("\nEvaluating models...")
    results = evaluate_models(test_loader, device=device)

    # --- Step 4: Interpretability / Explainability Analysis ---
    print("\nRunning interpretability analysis (Grad-CAM + Saliency)...")

    # Load clean model (base for comparison)
    model = TestCNNv2().to(device)
    if clean_model_path.exists():
        checkpoint = torch.load(clean_model_path, map_location=device)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        model.eval()
    else:
        print("Clean model not found, skipping interpretability analysis.")
        return

    # Define attacks
    attack_fns = {"FGSM": fgsm_attack, "PGD": pgd_attack}
    attack_params = {
        "FGSM": {"epsilon": 0.03},
        "PGD": {"epsilon": 0.03, "alpha": 2 / 255, "num_iter": 7},
    }

    # Run interpretability visualization
    visualize_interpretability(
      model=model,
      dataloader=test_loader,
      attack_fns=attack_fns,
      attack_params=attack_params,
      device=device,
      num_images=1,
      save_dir=DOCS_DIR / "interpretability"
    )

    # --- Step 5: Summary ---
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for result in results:
        print(f"\n{result['model']} Model:")
        print(f"   Clean Accuracy: {result['clean_acc']:.4f}")
        print(f"   FGSM Accuracy:  {result['fgsm_acc']:.4f}")
        print(f"   PGD Accuracy:   {result['pgd_acc']:.4f}")
        print(f"   Robustness:     {result['Robustness']:.4f}")
        print(f"   F1 Score:       {result['F1']:.4f}")

    print("\nInterpretability results saved in '../docs/evaluation_results/interpretability/'")


if __name__ == "__main__":
    main()
