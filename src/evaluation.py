import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BaseEvaluator(ABC):
    def __init__(self, device='auto', use_amp=False, seed=42):
        self.device = self._auto_select_device(device)
        self.use_amp = use_amp
        self.seed = seed
        set_seed(seed)
        if use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _auto_select_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    @abstractmethod
    def preprocess_batch(self, model, images, labels):
        """Hook for attacks or transformations before inference"""
        return images, labels

    def compute_metrics(self, outputs, labels):
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        return {'accuracy': correct / labels.size(0), 'total_samples': labels.size(0)}

    def evaluate(self, model, dataloader):
        model.eval()
        model.to(self.device)
        total_metrics = {'accuracy': 0.0, 'total_samples': 0}

        # mean/std for de-normalization if needed
        mean = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5], device=self.device).view(1, 3, 1, 1)

        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            processed_images, processed_labels = self.preprocess_batch(model, images, labels)

            with torch.no_grad():
                if self.use_amp and self.device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        outputs = model(processed_images)
                else:
                    outputs = model(processed_images)

            batch_metrics = self.compute_metrics(outputs, processed_labels)
            total_metrics['accuracy'] += batch_metrics['accuracy'] * batch_metrics['total_samples']
            total_metrics['total_samples'] += batch_metrics['total_samples']

        total_metrics['accuracy'] /= total_metrics['total_samples']
        return total_metrics


class CleanEvaluator(BaseEvaluator):
    """Evaluation on unperturbed data"""

    def preprocess_batch(self, model, images, labels):
        return images, labels


class AttackEvaluator(BaseEvaluator):
    """Evaluation on data perturbed by an attack (FGSM/PGD)

    attack_params may optionally contain 'mean' and 'std' (as lists/tuples or tensors).
    If not provided, defaults to mean=[0.5,0.5,0.5] and std=[0.5,0.5,0.5].
    """

    def __init__(self, attack_fn, attack_params=None, mean=None, std=None, **kwargs):
        super().__init__(**kwargs)
        self.attack_fn = attack_fn
        self.attack_params = dict(attack_params or {})

        # Determine mean/std to use
        mean_val = mean if mean is not None else self.attack_params.get('mean', (0.5, 0.5, 0.5))
        std_val = std if std is not None else self.attack_params.get('std', (0.5, 0.5, 0.5))

        # Convert to tensors on the evaluator device once
        self.mean = torch.tensor(mean_val, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std_val, device=self.device).view(1, 3, 1, 1)

        # Remove raw mean/std from attack_params to avoid double-embedding
        self.attack_params.pop('mean', None)
        self.attack_params.pop('std', None)

    def preprocess_batch(self, model, images, labels):
        """
        Applies the attack in [0,1] space and re-normalizes for the model.
        Ensures attack_fn always receives mean/std and other attack params.
        """
        # Denormalize to [0,1] space before attacking (model expects normalized input)
        images_denorm = images * self.std + self.mean

        # Prepare attack params copy and ensure mean/std are provided to the attack
        params = dict(self.attack_params)  # shallow copy
        # pass mean/std as tensors on correct device
        params.setdefault('mean', self.mean)
        params.setdefault('std', self.std)

        # Call the attack function. attack_fn is expected to accept (model, images, labels, **params)
        adv_images_denorm = self.attack_fn(model, images_denorm, labels, **params)

        # Re-normalize adv images and return
        adv_images_norm = (adv_images_denorm - self.mean) / self.std
        return adv_images_norm, labels
