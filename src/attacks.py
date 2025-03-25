import torch


def fgsm_attack(model, images, labels, epsilon=0.03):
    images.requires_grad = True
    outputs = model(images)

    loss = torch.nn.functional.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    perturbed_images = images + epsilon * images.grad.sign()
    perturbed_images = torch.clamp(perturbed_images, 0, 1).detach()

    return perturbed_images


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.01, steps=40, random_start=True):
    """
    Generate adversarial PGD examples
    :param alpha: attack learning rate (step size)
    :param steps: number of iterations
    :param random_start: if True, initialize with random noise
    """
    images = images.clone().detach()

    if random_start:
        # Random initialization in epsilon-ball
        images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        images = torch.clamp(images, 0, 1).detach()

    for _ in range(steps):
        images.requires_grad = True
        outputs = model(images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)

        grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

        # PGD step update
        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        images = torch.clamp(images + eta, 0, 1).detach()

    return images
