"""Public API for the THz LFI pre-processing pipeline.

Pipeline order (proposal §3.2.2):

    LFIRawData -> noise reduction -> normalisation -> phase unwrapping
    -> derived spatial background/stripe correction -> LFIRawData

The pipeline is intentionally type-stable: input and output are both
:class:`thz_preprocessing.io.LFIRawData` objects so downstream
consumers (image-generation, visualisation) see one consistent
interface regardless of which scan type is processed.

All operations are driven by a configuration dict — see
:mod:`thz_preprocessing.config` — so that processing is repeatable
and comparable across datasets (proposal §3.2.4 repeatability
criterion).
"""

from __future__ import annotations

import time
from typing import Any

from thz_preprocessing import (
    config as _config,
    noise_reduction,
    normalisation,
    phase_unwrapping,
    spatial_correction,
)
from thz_preprocessing.io import LFIRawData

__all__ = ["run_pipeline"]


def run_pipeline(
    data: LFIRawData,
    config_path: str | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> LFIRawData:
    """Run the full pre-processing pipeline on a single scan.

    Parameters
    ----------
    data
        Input scan as produced by the data-import module (or by one of
        the temporary loaders in :mod:`thz_preprocessing.io`).
    config_path
        Optional path to a TOML config file. If ``None``, the
        repository's ``config/default.toml`` is used. Per-target
        overrides keyed on ``data.target_type`` are merged on top.
    config_overrides
        Optional dict merged on top of the loaded config. Useful in
        unit tests to nudge a single parameter without writing a TOML.

    Returns
    -------
    LFIRawData
        Cleaned, normalised, and (where enabled) phase-unwrapped scan.
        ``metadata["pipeline"]`` carries an ordered list of the stages
        that were applied, with the parameters used for each — this is
        the self-describing output required by proposal §3.2.2.
    """
    cfg = _config.load(path=config_path, target_type=data.target_type)
    if config_overrides:
        from thz_preprocessing.config import _deep_merge  # type: ignore[attr-defined]

        cfg = _deep_merge(cfg, config_overrides)

    t0 = time.perf_counter()
    step1 = noise_reduction.apply(data, cfg["noise_reduction"])
    t1 = time.perf_counter()
    step2 = normalisation.apply(step1, cfg["normalisation"])
    t2 = time.perf_counter()
    step3 = phase_unwrapping.apply(step2, cfg["phase_unwrapping"])
    t3 = time.perf_counter()
    step4 = spatial_correction.apply(step3, cfg["spatial_correction"])
    if cfg["spatial_correction"]["enabled"]:
        step4.metadata["spatial_products"]["raw_amplitude"] = data.amplitude.copy()
        step4.metadata["spatial_products"]["raw_phase"] = data.phase.copy()
    t4 = time.perf_counter()

    step4.metadata["timings_seconds"] = {
        "noise_reduction": t1 - t0,
        "normalisation": t2 - t1,
        "phase_unwrapping": t3 - t2,
        "spatial_correction": t4 - t3,
        "total": t4 - t0,
    }
    step4.metadata["config_resolved"] = cfg
    step4.metadata["version"] = "0.5.0"
    return step4
