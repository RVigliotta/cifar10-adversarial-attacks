"""
main.py
Runs training or evaluation of models (Clean, FGSM, PGD) on CIFAR-10.
Compatible with GPU (Colab) and saves results in ../docs/evaluation_results.
"""

import random
import sys
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from data_loading import load_or_process_data
from training import train_clean_model, train_adversarial_model
from evaluation import CleanEvaluator, AttackEvaluator, plot_model_comparison, plot_robustness_analysis
from attacks import fgsm_attack, pgd_attack
from models.model_definitions import TestCNNv2

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

MODEL_DIR = Path("../models/saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DOCS_DIR = Path("../docs/evaluation_results")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

CLEAN_MODEL_PATH = MODEL_DIR / "test_cnn_clean.pth"
FGSM_MODEL_PATH = MODEL_DIR / "test_cnn_fgsm.pth"
PGD_MODEL_PATH = MODEL_DIR / "test_cnn_pgd.pth"


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
    """
    Evaluate three models (Clean, FGSM, PGD)
    Generate detailed metrics and advanced comparative plots.

    Args:
        test_loader: DataLoader for test data
        device: Torch device (auto-detected if None)
        epsilon: Perturbation magnitude for attacks

    Returns:
        List of evaluation results for each model
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []

    # Initialize evaluators
    clean_evaluator = CleanEvaluator(device=device)
    fgsm_evaluator = AttackEvaluator(
        attack_fn=fgsm_attack,
        attack_params={"epsilon": epsilon},
        device=device
    )
    pgd_evaluator = AttackEvaluator(
        attack_fn=pgd_attack,
        attack_params={"epsilon": epsilon, "alpha": 2 / 255, "num_iter": 7},
        device=device
    )

    model_configs = [
        ("Clean", CLEAN_MODEL_PATH),
        ("FGSM", FGSM_MODEL_PATH),
        ("PGD", PGD_MODEL_PATH)
    ]

    for model_name, model_path in model_configs:
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            continue

        print(f"\nEvaluating {model_name} model...")
        model = TestCNNv2().to(device)

        # Load model state with flexible checkpoint handling
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                # Assume it's a state dict with different structure
                model.load_state_dict(checkpoint, strict=False)
        else:
            # Direct state dict
            model.load_state_dict(checkpoint, strict=False)

        model.eval()

        # --- Comprehensive Evaluation ---
        clean_metrics = clean_evaluator.evaluate(model, test_loader)
        fgsm_metrics = fgsm_evaluator.evaluate(model, test_loader, clean_metrics=clean_metrics)
        pgd_metrics = pgd_evaluator.evaluate(model, test_loader, clean_metrics=clean_metrics)

        # Extract key metrics
        clean_accuracy = clean_metrics.get("accuracy")
        fgsm_accuracy = fgsm_metrics.get("adv_accuracy")
        pgd_accuracy = pgd_metrics.get("adv_accuracy")

        results.append({
            "model": model_name,
            "clean_acc": clean_accuracy,
            "fgsm_acc": fgsm_accuracy,
            "pgd_acc": pgd_accuracy,
            "ASR": fgsm_metrics.get("ASR", np.nan),
            "Fooling Rate": fgsm_metrics.get("Fooling Rate", np.nan),
            "Robustness": fgsm_metrics.get("Robustness", np.nan),
            "Distortion (L2)": fgsm_metrics.get("Distortion (L2)", np.nan),
            "Precision": clean_metrics.get("precision"),
            "Recall": clean_metrics.get("recall"),
            "F1": clean_metrics.get("f1"),
            "Evaluation Time (s)": clean_metrics.get("total_time")
        })

        print(f"Metrics collected for: {model_name}")

    # --- Save Combined Results ---
    df = pd.DataFrame(results)
    results_csv_path = DOCS_DIR / "evaluation_metrics.csv"
    df.to_csv(results_csv_path, index=False)
    print(f"\nCombined metrics saved to: {results_csv_path}")

    # --- Generate Advanced Visualizations ---
    plot_model_comparison(results)
    plot_robustness_analysis(results)

    return results


def run_complete_training_pipeline():
    """Run complete training pipeline for all model types"""
    print("Starting Complete Training Pipeline")
    print("=" * 50)

    # Train Clean Model
    print("\nTraining Clean Model...")
    train_clean_model(num_epochs=20, batch_size=64)

    # Train FGSM Adversarial Model
    print("\nTraining FGSM Adversarial Model...")
    train_adversarial_model(use_pgd=False, num_epochs=40, batch_size=128)

    # Train PGD Adversarial Model
    print("\nTraining PGD Adversarial Model...")
    train_adversarial_model(use_pgd=True, num_epochs=60, batch_size=128)

    print("\nTraining pipeline completed!")


def main():
    """Main execution function"""
    device = device_info()
    set_seed(42)

    # Option 1: Run complete training pipeline
    run_complete_training_pipeline()

    # Option 2: Evaluate existing models
    print("\nEvaluating existing models...")
    _, test_loader = prepare_data(batch_size=128)
    results = evaluate_models(test_loader, device=device)

    # Display summary
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


if __name__ == "__main__":
    main()
