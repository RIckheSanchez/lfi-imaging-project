"""Signal normalisation stage.

Places measurements from different scans on a common baseline,
enabling cross-dataset comparison.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def apply(signal: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Apply the configured normalisation method.

    Parameters
    ----------
    signal : np.ndarray
    params : dict
        Expected keys:
            method : {"baseline", "zscore", "reference"}

    Returns
    -------
    np.ndarray
    """
    method = params.get("method", "baseline")

    # TODO: implement
    if method == "baseline":
        raise NotImplementedError("baseline subtraction not yet implemented")
    elif method == "zscore":
        raise NotImplementedError("robust z-score not yet implemented")
    elif method == "reference":
        raise NotImplementedError("reference-channel division not yet implemented")
    else:
        raise ValueError(f"Unknown normalisation method: {method}")
