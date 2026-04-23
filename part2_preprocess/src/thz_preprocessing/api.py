"""Public API for the THz LFI pre-processing pipeline.

The pipeline chains four operations in order:
    raw signal -> noise reduction -> normalisation -> phase unwrapping -> output

All operations are driven by a configuration dict so that processing is
repeatable and comparable across datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from thz_preprocessing import (
    config,
    noise_reduction,
    normalisation,
    phase_unwrapping,
)


@dataclass
class PreprocessingResult:
    """Container for a processed signal plus its processing metadata."""

    signal: np.ndarray
    metadata: dict[str, Any]


def run_pipeline(
    raw_signal: np.ndarray,
    config_path: str | None = None,
) -> PreprocessingResult:
    """Run the full pre-processing pipeline on a single scan.

    Parameters
    ----------
    raw_signal : np.ndarray
        Raw self-mixing waveform array, shape (n_pixels, n_samples) or
        (height, width, n_samples) for 2D rasters.
    config_path : str, optional
        Path to a TOML config file. If None, defaults are used.

    Returns
    -------
    PreprocessingResult
        Cleaned signal plus metadata describing the parameters used.
    """
    cfg = config.load(config_path)

    # TODO: implement each stage
    step1 = noise_reduction.apply(raw_signal, cfg["noise_reduction"])
    step2 = normalisation.apply(step1, cfg["normalisation"])
    step3 = phase_unwrapping.apply(step2, cfg["phase_unwrapping"])

    return PreprocessingResult(
        signal=step3,
        metadata={"config": cfg, "version": "0.1.0"},
    )
