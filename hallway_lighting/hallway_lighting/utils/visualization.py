"""Visualization helpers for notebook training, evaluation, and inference."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import torch

from hallway_lighting.data.point_sampling import PointTarget
from hallway_lighting.data.transforms import denormalize_image


def _to_numpy_image(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Converts tensors or arrays to a displayable NumPy image."""

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in {1, 2, 3}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return image


def prepare_display_image(image: torch.Tensor | np.ndarray, denormalize: bool = True) -> np.ndarray:
    """Converts a normalized CHW tensor or HWC image to display space."""

    if isinstance(image, torch.Tensor) and denormalize and image.ndim == 3 and image.shape[0] == 3:
        image = denormalize_image(image).clamp(0.0, 1.0)
    image_np = _to_numpy_image(image)
    if image_np.ndim == 2:
        return image_np
    return np.clip(image_np, 0.0, 1.0)


def show_image(image: torch.Tensor | np.ndarray, title: str = "Image") -> None:
    """Displays a single image."""

    image_np = prepare_display_image(image)
    plt.figure(figsize=(6, 4))
    plt.imshow(image_np, cmap="gray" if image_np.ndim == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_pointwise_lux(point_values: Mapping[str, float], title: str = "Point-wise Hallway Illuminance") -> None:
    """Displays point-wise illuminance values as a bar chart."""

    names = list(point_values.keys())
    values = list(point_values.values())
    plt.figure(figsize=(8, 4))
    plt.bar(names, values)
    plt.ylabel("Lux")
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def overlay_points(
    image: torch.Tensor | np.ndarray,
    points: Sequence[PointTarget],
    title: str = "Point Targets",
    point_values: Mapping[str, float] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Draws normalized point targets over an image or heatmap."""

    image_np = prepare_display_image(image, denormalize=False)
    target_ax = ax or plt.figure(figsize=(6, 4)).add_subplot(111)
    target_ax.imshow(image_np, cmap="viridis" if image_np.ndim == 2 else None)

    height, width = image_np.shape[:2]
    for point in points:
        x_pixel = point.x * (width - 1)
        y_pixel = point.y * (height - 1)
        target_ax.scatter(x_pixel, y_pixel, s=45, c="white", edgecolors="black")
        if point_values is not None and point.name in point_values:
            label = f"{point.name}\n{point_values[point.name]:.1f}"
        else:
            label = point.name
        target_ax.text(x_pixel + 4, y_pixel - 4, label, fontsize=8, color="white")

    target_ax.set_title(title)
    target_ax.axis("off")
    return target_ax


def create_prediction_figure(
    image: torch.Tensor | np.ndarray,
    lux_map: torch.Tensor | np.ndarray,
    floor_mask_pred: torch.Tensor | np.ndarray | None = None,
    albedo_pred: torch.Tensor | np.ndarray | None = None,
    gloss_pred: torch.Tensor | np.ndarray | None = None,
    points: Sequence[PointTarget] | None = None,
    point_values: Mapping[str, float] | None = None,
    title: str = "Prediction Overview",
) -> plt.Figure:
    """Creates a multi-panel prediction figure for notebook display or saving."""

    panels: list[tuple[str, np.ndarray]] = [
        ("Input Image", prepare_display_image(image)),
        ("Predicted Lux", prepare_display_image(lux_map, denormalize=False)),
    ]
    if floor_mask_pred is not None:
        panels.append(("Predicted Floor Mask", prepare_display_image(floor_mask_pred, denormalize=False)))
    if albedo_pred is not None:
        panels.append(("Albedo Proxy", prepare_display_image(albedo_pred)))
    if gloss_pred is not None:
        panels.append(("Gloss Proxy", prepare_display_image(gloss_pred, denormalize=False)))

    columns = len(panels) + (1 if points is not None else 0)
    figure, axes = plt.subplots(1, columns, figsize=(4.5 * columns, 4))
    if columns == 1:
        axes = [axes]

    for axis, (panel_title, panel_image) in zip(axes, panels):
        axis.imshow(panel_image, cmap="viridis" if panel_image.ndim == 2 else None)
        axis.set_title(panel_title)
        axis.axis("off")

    if points is not None:
        overlay_points(
            image=prepare_display_image(lux_map, denormalize=False),
            points=points,
            title="Point Overlay",
            point_values=point_values,
            ax=axes[-1],
        )

    figure.suptitle(title)
    figure.tight_layout()
    return figure


def save_figure(figure: plt.Figure, path: str | Path) -> Path:
    """Saves a matplotlib figure to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def save_prediction_figure(
    path: str | Path,
    image: torch.Tensor | np.ndarray,
    lux_map: torch.Tensor | np.ndarray,
    floor_mask_pred: torch.Tensor | np.ndarray | None = None,
    albedo_pred: torch.Tensor | np.ndarray | None = None,
    gloss_pred: torch.Tensor | np.ndarray | None = None,
    points: Sequence[PointTarget] | None = None,
    point_values: Mapping[str, float] | None = None,
    title: str = "Prediction Overview",
) -> Path:
    """Builds and saves a prediction summary figure."""

    figure = create_prediction_figure(
        image=image,
        lux_map=lux_map,
        floor_mask_pred=floor_mask_pred,
        albedo_pred=albedo_pred,
        gloss_pred=gloss_pred,
        points=points,
        point_values=point_values,
        title=title,
    )
    return save_figure(figure, path)


def save_heatmap_image(
    path: str | Path,
    heatmap: torch.Tensor | np.ndarray,
    title: str = "Predicted Lux Heatmap",
) -> Path:
    """Saves a single heatmap panel to disk."""

    heatmap_np = prepare_display_image(heatmap, denormalize=False)
    figure, axis = plt.subplots(1, 1, figsize=(5, 4))
    image_handle = axis.imshow(heatmap_np, cmap="viridis")
    axis.set_title(title)
    axis.axis("off")
    figure.colorbar(image_handle, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return save_figure(figure, path)


def save_overlay_visualization(
    path: str | Path,
    image: torch.Tensor | np.ndarray,
    heatmap: torch.Tensor | np.ndarray,
    title: str = "Lux Overlay",
    alpha: float = 0.45,
) -> Path:
    """Saves a heatmap overlay on top of the input image."""

    image_np = prepare_display_image(image)
    heatmap_np = prepare_display_image(heatmap, denormalize=False)
    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    axis.imshow(image_np)
    axis.imshow(heatmap_np, cmap="inferno", alpha=alpha)
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    return save_figure(figure, path)


def save_point_annotation_visualization(
    path: str | Path,
    image: torch.Tensor | np.ndarray,
    points: Sequence[PointTarget],
    point_values: Mapping[str, float] | None = None,
    title: str = "Point-wise Hallway Illuminance",
) -> Path:
    """Saves point annotations over an image or heatmap."""

    figure, axis = plt.subplots(1, 1, figsize=(6, 4))
    overlay_points(
        image=image,
        points=points,
        title=title,
        point_values=point_values,
        ax=axis,
    )
    figure.tight_layout()
    return save_figure(figure, path)


def save_fixture_layout_visualization(
    path: str | Path,
    image: torch.Tensor | np.ndarray,
    fixtures: Sequence[Mapping[str, float | str | tuple[float, float, float, float]]],
    between_regions: Sequence[Mapping[str, object]] | None = None,
    title: str = "Detected Fixture Layout",
) -> Path:
    """Saves detected fixture locations and between-fixture regions over the input image."""

    image_np = prepare_display_image(image)
    figure, axis = plt.subplots(1, 1, figsize=(7, 4.5))
    axis.imshow(image_np)

    height, width = image_np.shape[:2]
    for region in between_regions or []:
        polygon_points = region.get("polygon")
        if not polygon_points:
            continue
        polygon_pixels = [
            (float(x_value) * (width - 1), float(y_value) * (height - 1))
            for x_value, y_value in polygon_points  # type: ignore[misc]
        ]
        axis.add_patch(
            Polygon(
                polygon_pixels,
                closed=True,
                facecolor="#4db6ac",
                edgecolor="#00695c",
                linewidth=1.4,
                alpha=0.22,
            )
        )
        label_x = float(np.mean([point[0] for point in polygon_pixels]))
        label_y = float(np.mean([point[1] for point in polygon_pixels[:2]]))
        region_label = str(region.get("name", "between_region"))
        axis.text(label_x, label_y, region_label, color="white", fontsize=8, ha="center", va="bottom")

    for fixture in fixtures:
        x_pixel = float(fixture["x"]) * (width - 1)
        y_pixel = float(fixture["y"]) * (height - 1)
        confidence = float(fixture.get("confidence", 0.0))
        label = f"{fixture.get('name', 'fixture')} ({confidence:.2f})"
        axis.scatter(x_pixel, y_pixel, s=90, c="#ffd54f", edgecolors="black", linewidths=0.8, zorder=3)
        axis.text(x_pixel + 6, y_pixel - 6, label, color="white", fontsize=8, zorder=4)

    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    return save_figure(figure, path)


def show_multitask_example(
    image: torch.Tensor | np.ndarray,
    lux_map: torch.Tensor | np.ndarray | None = None,
    floor_mask: torch.Tensor | np.ndarray | None = None,
    albedo: torch.Tensor | np.ndarray | None = None,
    gloss: torch.Tensor | np.ndarray | None = None,
) -> None:
    """Displays a simple multitask example."""

    figure = create_prediction_figure(
        image=image,
        lux_map=lux_map if lux_map is not None else image,
        floor_mask_pred=floor_mask,
        albedo_pred=albedo,
        gloss_pred=gloss,
        points=None,
        title="Multitask Example",
    )
    plt.show()
    plt.close(figure)
