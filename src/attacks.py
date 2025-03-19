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
