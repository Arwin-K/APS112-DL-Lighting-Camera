"""Visualization helpers used by the notebook."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from hallway_lighting.data.point_sampling import PointTarget


def _to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converts CHW tensors or HWC arrays to a displayable NumPy image."""

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in {1, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = image[..., 0]
    return image


def show_image(image: torch.Tensor | np.ndarray, title: str = "Image") -> None:
    """Displays a single image."""

    plt.figure(figsize=(6, 4))
    plt.imshow(_to_numpy_image(image), cmap="viridis" if _to_numpy_image(image).ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_multitask_example(
    image: torch.Tensor | np.ndarray,
    lux_map: torch.Tensor | np.ndarray | None = None,
    floor_mask: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Displays the RGB image alongside optional prediction targets."""

    panels = [("RGB", _to_numpy_image(image))]
    if lux_map is not None:
        panels.append(("Lux Map", _to_numpy_image(lux_map)))
    if floor_mask is not None:
        panels.append(("Floor Mask", _to_numpy_image(floor_mask)))

    plt.figure(figsize=(5 * len(panels), 4))
    for index, (title, panel) in enumerate(panels, start=1):
        plt.subplot(1, len(panels), index)
        plt.imshow(panel, cmap="viridis" if panel.ndim == 2 else None)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_pointwise_lux(point_values: dict[str, float]) -> None:
    """Displays point-wise illuminance values as a bar chart."""

    names = list(point_values.keys())
    values = list(point_values.values())
    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylabel("Lux")
    plt.xticks(rotation=45, ha="right")
    plt.title("Point-wise Hallway Illuminance")
    plt.tight_layout()
    plt.show()


def overlay_points(image: torch.Tensor | np.ndarray, points: Sequence[PointTarget]) -> None:
    """Displays an RGB image with normalized point targets overlaid."""

    array = _to_numpy_image(image)
    height, width = array.shape[:2]
    plt.figure(figsize=(6, 4))
    plt.imshow(array)
    for point in points:
        plt.scatter(point.x * (width - 1), point.y * (height - 1), s=50)
        plt.text(point.x * (width - 1), point.y * (height - 1), point.name, fontsize=8)
    plt.title("Point Targets")
    plt.axis("off")
    plt.show()
