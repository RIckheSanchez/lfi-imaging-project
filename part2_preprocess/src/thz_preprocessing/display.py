"""Helpers for showing LFI images the way the scanner saw them.

Pixels are not square, and the Leaf arrays are stored long-axis first.
Showing an array with square pixels (or stretching it to fill a panel)
distorts the sample, which makes real images hard to recognise.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def oriented(image: np.ndarray, target_type: str) -> np.ndarray:
    """Return a 2D image with its long physical axis horizontal."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape {image.shape}")
    return image.T if str(target_type).lower() == "leaf" else image


def pixel_aspect(image_shape: tuple[int, int], metadata: dict[str, Any] | None) -> float:
    """Height/width of one displayed pixel, for ``imshow(aspect=...)``.

    ``image_shape`` is the shape of the array as displayed (after
    :func:`oriented`). ``metadata["display_aspect_ratio"]`` is the physical
    width/height of the full frame; without it pixels are assumed square.
    """
    ratio = (metadata or {}).get("display_aspect_ratio")
    if not ratio:
        return 1.0
    rows, columns = image_shape
    return float((columns / rows) / float(ratio))


__all__ = ["oriented", "pixel_aspect"]
