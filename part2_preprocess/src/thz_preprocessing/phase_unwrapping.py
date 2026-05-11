"""Phase unwrapping stage.

Operates on :attr:`LFIRawData.phase`. Two methods are provided to span
both data regimes the project encounters:

* ``"skimage_2d"`` — quality-guided 2D unwrap (:func:`skimage.restoration.unwrap_phase`)
  applied independently to each z-slice. Appropriate for image-level
  scans (n_z == 1, e.g. Skin pilot) where the unwrap is genuinely 2D.
* ``"itoh_1d"``    — Itoh-style 1D unwrap along the z (sweep) axis,
  implemented via :func:`numpy.unwrap`. Appropriate for sweep-axis
  volumetric scans (n_z >> 1, e.g. Leaf), where temporal phase
  continuity along the sweep dominates the unwrap problem (proposal
  §3.2.2). NumPy's ``unwrap`` is mathematically Itoh's algorithm
  applied along the chosen axis.
* ``"none"``       — pass-through (wrapped phase is preserved).

Disabled by default (``enabled = false``) so the pipeline is byte-stable
for the Progress Seminar repeatability test when no decision has been
taken about phase unwrapping (proposal §3.2.5 mid-year pivot).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from thz_preprocessing.io import LFIRawData


def _itoh_1d_along_z(phase: np.ndarray) -> np.ndarray:
    if phase.shape[0] < 2:
        # Itoh-1D is meaningful only when n_z > 1; degrade gracefully.
        return phase.copy()
    return np.unwrap(phase, axis=0)


def _skimage_2d_per_zslice(phase: np.ndarray) -> np.ndarray:
    # Local import: skimage is optional from the perspective of the
    # core pipeline and adds appreciable import latency.
    from skimage.restoration import unwrap_phase

    out = np.empty_like(phase)
    for z in range(phase.shape[0]):
        out[z] = unwrap_phase(phase[z])
    return out


def apply(data: LFIRawData, params: dict[str, Any]) -> LFIRawData:
    """Conditionally apply phase unwrapping to ``data.phase``.

    Parameters
    ----------
    data
        Input scan. Not mutated.
    params
        Sub-dict from the resolved config under ``"phase_unwrapping"``.
        ``enabled = false`` causes the stage to pass through.

    Returns
    -------
    LFIRawData
        New :class:`LFIRawData` with (optionally) unwrapped phase.
    """
    enabled = bool(params.get("enabled", False))
    method = str(params.get("method", "skimage_2d")).lower()

    if not enabled:
        out = data.phase.copy()
    elif method == "none":
        out = data.phase.copy()
    elif method == "skimage_2d":
        out = _skimage_2d_per_zslice(data.phase)
    elif method == "itoh_1d":
        out = _itoh_1d_along_z(data.phase)
    else:
        raise ValueError(f"Unknown phase-unwrapping method: {method!r}")

    new_meta = deepcopy(data.metadata)
    new_meta.setdefault("pipeline", []).append({"phase_unwrapping": dict(params)})

    return LFIRawData(
        amplitude=data.amplitude.copy(),
        phase=out,
        z_indices=list(data.z_indices),
        target_type=data.target_type,
        metadata=new_meta,
    )
