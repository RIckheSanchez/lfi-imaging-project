"""Signal normalisation stage.

Operates on :attr:`LFIRawData.amplitude` so that scans acquired in
different sessions / different drive currents can be compared on a
common scale (proposal §3.2.2). Phase is intentionally **not**
normalised — wrapped phase is dimensionless modulo 2*pi already.

Dispatch table:

* ``"baseline"`` — subtract a per-z-slice low-percentile baseline
  (robust dark-level estimate).
* ``"zscore"``   — per-z-slice zero-mean / unit-variance. When
  ``robust = True``, median and MAD are used instead of mean / std,
  which is the recommended default for THz LFI amplitude maps because
  bright reflective pixels heavily skew the parametric estimators.
* ``"none"``     — pass-through.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from thz_preprocessing.io import LFIRawData


_MAD_TO_STD = 1.4826  # consistency factor under Gaussian assumption


def _baseline_per_zslice(arr: np.ndarray, percentile: float) -> np.ndarray:
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"baseline_percentile must be in [0, 100], got {percentile}")
    baselines = np.percentile(arr.reshape(arr.shape[0], -1), percentile, axis=1)
    return arr - baselines[:, None, None]


def _zscore_per_zslice(arr: np.ndarray, robust: bool) -> np.ndarray:
    n_z = arr.shape[0]
    flat = arr.reshape(n_z, -1)
    if robust:
        centre = np.median(flat, axis=1)
        spread = _MAD_TO_STD * np.median(np.abs(flat - centre[:, None]), axis=1)
    else:
        centre = np.mean(flat, axis=1)
        spread = np.std(flat, axis=1)
    # Guard against degenerate slices (e.g. uniform region).
    spread = np.where(spread > 0, spread, 1.0)
    return (arr - centre[:, None, None]) / spread[:, None, None]


def apply(data: LFIRawData, params: dict[str, Any]) -> LFIRawData:
    """Apply the configured normalisation method to ``data.amplitude``.

    Parameters
    ----------
    data
        Input scan. Not mutated.
    params
        Sub-dict from the resolved config under ``"normalisation"``.

    Returns
    -------
    LFIRawData
        New :class:`LFIRawData` with normalised amplitude.
    """
    method = str(params.get("method", "baseline")).lower()

    if method == "none":
        out = data.amplitude.copy()
    elif method == "baseline":
        out = _baseline_per_zslice(
            data.amplitude, float(params.get("baseline_percentile", 5.0))
        )
    elif method == "zscore":
        out = _zscore_per_zslice(data.amplitude, bool(params.get("robust", True)))
    else:
        raise ValueError(f"Unknown normalisation method: {method!r}")

    new_meta = deepcopy(data.metadata)
    new_meta.setdefault("pipeline", []).append({"normalisation": dict(params)})

    return LFIRawData(
        amplitude=out,
        phase=data.phase.copy(),
        z_indices=list(data.z_indices),
        target_type=data.target_type,
        metadata=new_meta,
    )
