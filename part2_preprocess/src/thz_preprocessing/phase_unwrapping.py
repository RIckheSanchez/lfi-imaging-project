"""Phase unwrapping stage.

Only relevant when the downstream image-generation module requires
continuous phase estimates (e.g. 3D or swept-frequency modes).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply(signal: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply the configured phase-unwrapping method.

    Parameters
    ----------
    signal : np.ndarray
    params : dict
        Expected keys:
            enabled : bool
            method  : {"itoh_1d", "quality_guided_2d"}

    Returns
    -------
    np.ndarray
    """
    if not params.get("enabled", False):
        return signal  # pass-through

    method = params.get("method", "itoh_1d")

    # TODO: implement
    if method == "itoh_1d":
        raise NotImplementedError("1D Itoh-style unwrapping not yet implemented")
    elif method == "quality_guided_2d":
        raise NotImplementedError("quality-guided 2D unwrapping not yet implemented")
    else:
        raise ValueError(f"Unknown phase-unwrapping method: {method}")
