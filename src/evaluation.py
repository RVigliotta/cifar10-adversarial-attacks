import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.metrics import precision_score, recall_score, f1_score

# Add project root to path for cross-platform compatibility
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize(images, mean, std):
    """Denormalize images using mean and std"""
    return images * std + mean


def normalize(images, mean, std):
    """Normalize images using mean and std"""
    return (images - mean) / std


class BaseEvaluator(ABC):
    """Abstract base class for model evaluators"""

    def __init__(self, device='auto', use_amp=False, seed=42):
        self.device = self._auto_select_device(device)
        self.use_amp = use_amp
        set_seed(seed)

    def _auto_select_device(self, device):
        """Automatically select device (CUDA if available, else CPU)"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    @abstractmethod
    def preprocess_batch(self, model, images, labels):
        """Preprocess batch data - to be implemented by subclasses"""
        return images, labels

    def compute_basic_metrics(self, preds, labels):
        """Compute basic classification metrics"""
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        acc = np.mean(preds_np == labels_np)
        precision = precision_score(labels_np, preds_np, average='macro', zero_division=0)
        recall = recall_score(labels_np, preds_np, average='macro', zero_division=0)
        f1 = f1_score(labels_np, preds_np, average='macro', zero_division=0)
        return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}


class CleanEvaluator(BaseEvaluator):
    """Evaluator for clean (non-adversarial) model performance"""

    def preprocess_batch(self, model, images, labels):
        """No preprocessing for clean evaluation"""
        return images, labels

    def evaluate(self, model, dataloader):
        """Evaluate model on clean data"""
        model.eval()
        model.to(self.device)
        all_preds, all_labels = [], []

        start_time = time.time()

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.compute_basic_metrics(all_preds, all_labels)
        metrics['total_time'] = time.time() - start_time
        return metrics


class AttackEvaluator(BaseEvaluator):
    """Evaluator for adversarial attack scenarios"""

    def __init__(self, attack_fn, attack_params=None, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_fn = attack_fn
        self.attack_params = dict(attack_params or {})

        # Handle normalization parameters
        mean_val = mean if mean is not None else self.attack_params.get('mean', (0.5, 0.5, 0.5))
        std_val = std if std is not None else self.attack_params.get('std', (0.5, 0.5, 0.5))

        self.mean = torch.tensor(mean_val, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std_val, device=self.device).view(1, 3, 1, 1)

        # Remove from attack params to avoid duplication
        self.attack_params.pop('mean', None)
        self.attack_params.pop('std', None)

    def preprocess_batch(self, model, images, labels):
        """Generate adversarial examples for the batch"""
        images_denorm = denormalize(images, self.mean, self.std)
        params = dict(self.attack_params)
        params.setdefault('mean', self.mean)
        params.setdefault('std', self.std)

        adv_images_denorm = self.attack_fn(model, images_denorm, labels, **params)
        adv_images_norm = normalize(adv_images_denorm, self.mean, self.std)
        return adv_images_norm, labels

    def evaluate(self, model, dataloader, clean_metrics=None):
        """Evaluate model under adversarial attack"""
        model.eval()
        model.to(self.device)

        all_preds, all_labels = [], []
        distortions = []
        start_time = time.time()
        attack_times = []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)

            images_denorm = denormalize(images, self.mean, self.std)
            params = dict(self.attack_params)
            params.setdefault('mean', self.mean)
            params.setdefault('std', self.std)

            # ⏱️ Attack generation timing
            attack_start_time = time.time()
            adv_images_denorm = self.attack_fn(model, images_denorm, labels, **params)
            attack_times.append(time.time() - attack_start_time)

            adv_images = normalize(adv_images_denorm, self.mean, self.std)

            # L2 distortion calculation
            batch_distortion = torch.norm(
                (adv_images_denorm - images_denorm).view(images.size(0), -1),
                dim=1
            ).mean().item()
            distortions.append(batch_distortion)

            with torch.no_grad():
                outputs = model(adv_images)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        adv_metrics = self.compute_basic_metrics(all_preds, all_labels)

        # Calculate advanced adversarial metrics
        ASR = 1 - adv_metrics['accuracy']  # Attack Success Rate
        fooling_rate = np.mean(all_preds.numpy() != all_labels.numpy())
        robustness = 1 - fooling_rate
        distortion = np.mean(distortions)
        avg_attack_time = np.mean(attack_times)

        metrics = {
            'clean_accuracy': clean_metrics['accuracy'] if clean_metrics else np.nan,
            'adv_accuracy': adv_metrics['accuracy'],
            'ASR': ASR,
            'Fooling Rate': fooling_rate,
            'Robustness': robustness,
            'Distortion (L2)': distortion,
            'Attack Time (s/batch)': avg_attack_time,
            'Precision': adv_metrics['precision'],
            'Recall': adv_metrics['recall'],
            'F1': adv_metrics['f1'],
            'Total Evaluation Time': time.time() - start_time
        }

        return metrics


def plot_model_comparison(results, save_dir="../docs/evaluation_results"):
    """Create informative and aesthetic comparison plots"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    models = [r['model'] for r in results]

    # Metrics for radar chart
    metrics_to_plot = ['clean_acc', 'fgsm_acc', 'pgd_acc', 'Robustness', 'Precision', 'Recall']
    metrics_labels = ['Clean Acc', 'FGSM Acc', 'PGD Acc', 'Robustness', 'Precision', 'Recall']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Radar Chart for multiple metrics
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    for i, model_result in enumerate(results):
        values = [model_result[metric] for metric in metrics_to_plot]
        values += values[:1]  # Close the circle
        ax1.plot(angles, values, 'o-', linewidth=2, label=model_result['model'])
        ax1.fill(angles, values, alpha=0.1)

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_labels, fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', fontsize=14, pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax1.grid(True)

    # 2. Grouped bar plot for accuracy comparison
    clean_accs = [r['clean_acc'] for r in results]
    fgsm_accs = [r['fgsm_acc'] for r in results]
    pgd_accs = [r['pgd_acc'] for r in results]

    x = np.arange(len(models))
    width = 0.25

    bars1 = ax2.bar(x - width, clean_accs, width, label='Clean', alpha=0.8, color='#2ecc71')
    bars2 = ax2.bar(x, fgsm_accs, width, label='FGSM', alpha=0.8, color='#e74c3c')
    bars3 = ax2.bar(x + width, pgd_accs, width, label='PGD', alpha=0.8, color='#3498db')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy under Different Attack Scenarios', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path / "model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"📊 Model comparison plots saved to {save_path}/model_comparison.png")


def plot_robustness_analysis(results, save_dir="../docs/evaluation_results"):
    """Specific robustness analysis visualization"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    models = [r['model'] for r in results]
    robustness_scores = [r['Robustness'] for r in results]
    asr_scores = [r['ASR'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Robustness plot
    bars1 = ax1.bar(models, robustness_scores, alpha=0.8, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax1.set_ylabel('Robustness Score', fontsize=12)
    ax1.set_title('Model Robustness Comparison', fontsize=14)
    ax1.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=11)

    # Attack Success Rate plot
    bars2 = ax2.bar(models, asr_scores, alpha=0.8, color=['#e74c3c', '#3498db', '#2ecc71'])
    ax2.set_ylabel('Attack Success Rate', fontsize=12)
    ax2.set_title('Attack Effectiveness', fontsize=14)
    ax2.set_ylim(0, 1)

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=11)

    ax1.grid(axis='y', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "robustness_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"🛡️ Robustness analysis saved to {save_path}/robustness_analysis.png")


def save_results_to_csv(results, filename="model_evaluation_results.csv", save_dir="../docs/evaluation_results"):
    """Save evaluation results to CSV file"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    filepath = save_path / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df
