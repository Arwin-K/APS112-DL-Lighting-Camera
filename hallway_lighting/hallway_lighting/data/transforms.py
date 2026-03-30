"""Image transform builders used by the notebook and training code."""

from __future__ import annotations

from typing import Iterable

import torch


def _build_resize(image_size: tuple[int, int]):
    """Builds a torchvision resize transform lazily."""

    from torchvision import transforms

    return transforms.Resize(image_size, antialias=True)


def build_image_transform(
    image_size: tuple[int, int] = (256, 256),
    normalize: bool = True,
) -> "torch.nn.Module":
    """Builds a standard RGB transform for training or inference."""

    from torchvision import transforms

    steps: list[object] = [_build_resize(image_size), transforms.ToTensor()]
    if normalize:
        steps.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )
    return transforms.Compose(steps)


def denormalize_image(
    image_tensor: torch.Tensor,
    mean: Iterable[float] = (0.485, 0.456, 0.406),
    std: Iterable[float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Undo ImageNet-style normalization for visualization."""

    mean_tensor = torch.tensor(list(mean), device=image_tensor.device).view(-1, 1, 1)
    std_tensor = torch.tensor(list(std), device=image_tensor.device).view(-1, 1, 1)
    return image_tensor * std_tensor + mean_tensor
