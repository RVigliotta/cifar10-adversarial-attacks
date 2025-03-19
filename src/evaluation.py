import torch
from abc import ABC, abstractmethod
from tqdm import tqdm


class BaseEvaluator(ABC):
    def __init__(self, device='auto', use_amp=False):
        self.device = self._auto_select_device(device)
        self.use_amp = use_amp
        if use_amp and self.device.type == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def _auto_select_device(self, device):
        """Automatically selects CUDA if available"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

    @abstractmethod
    def preprocess_batch(self, model, images, labels):
        """Hook for transformations/attacks before inference"""
        return images, labels

    def compute_metrics(self, outputs, labels):
        """Computes the main metrics"""
        _, preds = torch.max(outputs, 1)
        correct = (preds == labels).sum().item()
        return {
            'accuracy': correct / labels.size(0),
            'total_samples': labels.size(0)
        }

    def evaluate(self, model, dataloader):
        """Evaluates the model on a DataLoader"""
        model.eval()
        model.to(self.device)

        total_metrics = {'accuracy': 0.0, 'total_samples': 0}

        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(isinstance(self, AttackEvaluator)):
                processed_images, processed_labels = self.preprocess_batch(model, images, labels)

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
    """Evaluation on clean data (without attacks)"""
    def preprocess_batch(self, model, images, labels):
        return images, labels


class AttackEvaluator(BaseEvaluator):
    """Evaluation on data perturbed with an attack"""
    def __init__(self, attack_fn, attack_params, **kwargs):
        super().__init__(**kwargs)
        self.attack_fn = attack_fn
        self.attack_params = attack_params

    def preprocess_batch(self, model, images, labels):
        """Applies the attack to the input data"""
        return self.attack_fn(model, images, labels, **self.attack_params), labels
