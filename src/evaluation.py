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
from captum.attr import Saliency
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

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

            # Attack generation timing
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


def plot_model_comparison(results, save_dir=None):
    """Create informative and aesthetic comparison plots"""
    if save_dir is None:
        save_dir = "../docs/evaluation_results"
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

    print(f"Model comparison plots saved to {save_path}/model_comparison.png")


def plot_robustness_analysis(results, save_dir=None):
    """Specific robustness analysis visualization"""
    if save_dir is None:
        save_dir = "../docs/evaluation_results"
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

    print(f"Robustness analysis saved to {save_path}/robustness_analysis.png")


def save_results_to_csv(results, filename="model_evaluation_results.csv", save_dir="../docs/evaluation_results"):
    """Save evaluation results to CSV file"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    filepath = save_path / filename
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


def visualize_interpretability(model, dataloader, device, attack_fns=None, attack_params=None,
                               save_dir="../docs/evaluation_results", num_images=1):
    """
    Clean interpretability visualization with balanced layout and improved font sizes.
    """
    epsilon = attack_params.get("epsilon", 0.03) if attack_params else 0.03
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    # --- Prepare sample data ---
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    images, labels = images[:num_images].to(device), labels[:num_images].to(device)

    # --- Find last convolutional layer for Grad-CAM ---
    target_layer = None
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
            break

    if target_layer is None:
        raise ValueError("❌ No convolutional layer found for Grad-CAM.")

    # Initialize interpretability methods
    gradcam = GradCAM(model=model, target_layers=[target_layer])
    saliency = Saliency(model)

    # Normalization parameters
    mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)

    def denorm(x):
        return (x * std + mean).clamp(0, 1)

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Create clean visualization for single image
    img = images[0].unsqueeze(0)
    label = labels[0].item()
    true_class = class_names[label]

    # Get original image
    rgb_img = denorm(img)[0].permute(1, 2, 0).cpu().numpy()
    rgb_img = np.clip(rgb_img, 0, 1)

    # Get model prediction for clean image
    with torch.no_grad():
        clean_output = model(img)
        clean_pred = torch.argmax(clean_output, dim=1).item()
        clean_confidence = torch.softmax(clean_output, dim=1)[0, clean_pred].item()
        clean_class = class_names[clean_pred]

    # Generate saliency map and Grad-CAM for clean image
    saliency_attrs = saliency.attribute(img, target=clean_pred)
    saliency_map = saliency_attrs.abs().max(dim=1)[0].squeeze().cpu().detach().numpy()
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    grayscale_cam = gradcam(input_tensor=img, targets=[ClassifierOutputTarget(clean_pred)])[0]

    # Prepare conditions for plotting
    conditions = []

    # Clean condition
    conditions.append({
        'name': 'Clean',
        'image': rgb_img,
        'saliency': saliency_map,
        'gradcam': grayscale_cam,
        'scores': torch.softmax(clean_output, dim=1)[0].cpu().detach().numpy(),
        'true_class': true_class,
        'pred_class': clean_class,
        'confidence': clean_confidence,
        'is_correct': clean_pred == label
    })

    # Adversarial conditions
    if attack_fns:
        for attack_name, attack_fn in attack_fns.items():
            # Generate adversarial example
            img_denorm = denorm(img)
            adv_img = attack_fn(model, img_denorm, torch.tensor([label], device=device),
                                epsilon=epsilon, mean=mean, std=std)
            adv_img_norm = (adv_img - mean) / std

            # Get adversarial image
            rgb_adv = denorm(adv_img_norm)[0].permute(1, 2, 0).cpu().detach().numpy()
            rgb_adv = np.clip(rgb_adv, 0, 1)

            # Get model prediction on adversarial example
            with torch.no_grad():
                adv_output = model(adv_img_norm)
                adv_pred = torch.argmax(adv_output, dim=1).item()
                adv_confidence = torch.softmax(adv_output, dim=1)[0, adv_pred].item()
                adv_class = class_names[adv_pred]

            # Generate explanations for adversarial image
            saliency_adv = saliency.attribute(adv_img_norm, target=adv_pred)
            saliency_map_adv = saliency_adv.abs().max(dim=1)[0].squeeze().cpu().detach().numpy()
            saliency_map_adv = (saliency_map_adv - saliency_map_adv.min()) / (
                        saliency_map_adv.max() - saliency_map_adv.min() + 1e-8)

            grayscale_cam_adv = gradcam(input_tensor=adv_img_norm, targets=[ClassifierOutputTarget(adv_pred)])[0]

            conditions.append({
                'name': f'{attack_name} Attack',
                'image': rgb_adv,
                'saliency': saliency_map_adv,
                'gradcam': grayscale_cam_adv,
                'scores': torch.softmax(adv_output, dim=1)[0].cpu().detach().numpy(),
                'true_class': true_class,
                'pred_class': adv_class,
                'confidence': adv_confidence,
                'is_correct': adv_pred == label
            })

    # Create figure with adjusted height
    num_rows = len(conditions)
    fig, axes = plt.subplots(num_rows, 4, figsize=(18, 4.2 * num_rows))  # Reduced row height

    if num_rows == 1:
        axes = axes.reshape(1, -1)

    # Plot each condition
    for i, condition in enumerate(conditions):
        # Column 0: Original/Adversarial Image
        axes[i, 0].imshow(condition['image'])
        color = 'green' if condition['is_correct'] else 'red'
        status = "✓ Correct" if condition['is_correct'] else "✗ Wrong"
        axes[i, 0].set_title(
            f"{condition['name']}\nTrue: {condition['true_class']} | Pred: {condition['pred_class']}\n{status} | Conf: {condition['confidence']:.3f}",
            fontsize=15, color=color, fontweight='bold', pad=12)
        axes[i, 0].axis('off')

        # Column 1: Saliency Map
        axes[i, 1].imshow(condition['image'], alpha=0.8)
        im_sal = axes[i, 1].imshow(condition['saliency'], cmap='hot', alpha=0.6)
        axes[i, 1].set_title("Saliency Map", fontsize=15, fontweight='bold', pad=12)
        axes[i, 1].axis('off')

        # Column 2: Grad-CAM
        axes[i, 2].imshow(condition['image'], alpha=0.8)
        im_cam = axes[i, 2].imshow(condition['gradcam'], cmap='jet', alpha=0.6)
        axes[i, 2].set_title("Grad-CAM", fontsize=15, fontweight='bold', pad=12)
        axes[i, 2].axis('off')

        # Column 3: Prediction Distribution (more compact)
        scores = condition['scores']
        top_k = 4
        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]
        top_classes = [class_names[idx] for idx in top_indices]

        # Create more compact bar plot
        bars = axes[i, 3].barh(range(len(top_scores)), top_scores, color='#1e3a5f', alpha=0.8, height=0.6)
        axes[i, 3].set_yticks(range(len(top_scores)))
        axes[i, 3].set_yticklabels(top_classes, fontsize=11)
        axes[i, 3].set_xlim(0, 1)
        axes[i, 3].set_title('Top-4 Predictions', fontsize=15, fontweight='bold', pad=12)
        axes[i, 3].grid(axis='x', alpha=0.3, linestyle='--')

        # Adjust subplot position to make it more compact
        pos = axes[i, 3].get_position()
        axes[i, 3].set_position([pos.x0, pos.y0 + 0.05, pos.width, pos.height - 0.1])

        # Add value labels with increased font size
        for j, (bar, score) in enumerate(zip(bars, top_scores)):
            width = bar.get_width()
            axes[i, 3].text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                            f'{score:.3f}', ha='left', va='center', fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))

            # Highlight the predicted class
            if top_classes[j] == condition['pred_class']:
                bar.set_color('#e74c3c')
                bar.set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(save_path / "interpretability_analysis.png", dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()

    # Clean up
    del gradcam, saliency
    torch.cuda.empty_cache()

    print(f"Interpretability analysis saved to: {save_path}/interpretability_analysis.png")
