import torch


def fgsm_attack(model, images, labels, epsilon=0.03, mean=None, std=None, device='cpu'):
    """
    FGSM attack (Goodfellow et al. 2014)
    Perturbs the images in the direction of the loss gradient.
    Operates in the [0,1] space but normalizes before passing to the model.
    """
    model.eval()

    if mean is None or std is None:
        # Default CIFAR-10 normalization values
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=images.device).view(1, 3, 1, 1)

    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)

    # Normalize input before forward
    inputs = (images - mean) / std
    inputs.requires_grad = True

    outputs = model(inputs)
    loss = torch.nn.functional.cross_entropy(outputs, labels)

    model.zero_grad()
    loss.backward()

    # Apply perturbation
    perturbed = images + epsilon * inputs.grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)

    return perturbed.detach()


def pgd_attack(model, images, labels, epsilon=0.03, alpha=2/255, num_iter=7, mean=None, std=None, device='cpu'):
    """
    PGD (Madry et al. 2018)
    Iterative FGSM with projection into the epsilon-ball around the original images.
    Operates in the [0,1] space, but the model receives normalized inputs.
    """
    model.eval()

    if mean is None or std is None:
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=images.device).view(1, 3, 1, 1)

    images = images.clone().detach().to(images.device)
    labels = labels.to(images.device)
    perturbed = images.clone().detach()

    for _ in range(num_iter):
        perturbed.requires_grad = True
        inputs = (perturbed - mean) / std
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()
        adv_images = perturbed + alpha * perturbed.grad.sign()
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(images + eta, 0, 1).detach()

    return perturbed
