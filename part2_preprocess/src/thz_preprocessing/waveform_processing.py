"""Conservative processing for fitted LFI traces.

The 2026 RDM ``RX_LINE_OF_BEST_FIT`` arrays are already an
acquisition-system fit, not raw ADC samples. This module therefore keeps the
default processing deliberately light: optional per-trace DC removal followed
by Savitzky-Golay smoothing along the 184-sample trace axis.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
from scipy import signal

from thz_preprocessing.io import LFIWaveformData


def evaluate_preservation(
    reference: np.ndarray,
    processed: np.ndarray,
) -> dict[str, float | int]:
    """Quantify how strongly processing changed the non-empty traces."""
    if reference.shape != processed.shape:
        raise ValueError(
            f"Waveform shape mismatch: {reference.shape} vs {processed.shape}"
        )

    ref = np.asarray(reference, dtype=np.float64)
    out = np.asarray(processed, dtype=np.float64)
    valid = np.ptp(ref, axis=-1) > 0.0
    valid_count = int(np.count_nonzero(valid))
    total_count = int(valid.size)
    if valid_count == 0:
        return {
            "valid_trace_count": 0,
            "total_trace_count": total_count,
            "median_correlation": float("nan"),
            "p05_correlation": float("nan"),
            "median_peak_to_peak_retention": float("nan"),
            "roughness_ratio": float("nan"),
            "residual_rms": 0.0,
        }

    ref_valid = ref[valid]
    out_valid = out[valid]
    ref_centered = ref_valid - ref_valid.mean(axis=-1, keepdims=True)
    out_centered = out_valid - out_valid.mean(axis=-1, keepdims=True)

    numerator = np.sum(ref_centered * out_centered, axis=-1)
    denominator = np.sqrt(
        np.sum(ref_centered**2, axis=-1)
        * np.sum(out_centered**2, axis=-1)
    )
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )

    ref_p2p = np.ptp(ref_valid, axis=-1)
    out_p2p = np.ptp(out_valid, axis=-1)
    p2p_retention = np.divide(
        out_p2p,
        ref_p2p,
        out=np.zeros_like(out_p2p),
        where=ref_p2p > 0,
    )

    ref_roughness = np.sqrt(np.mean(np.diff(ref_centered, n=2, axis=-1) ** 2))
    out_roughness = np.sqrt(np.mean(np.diff(out_centered, n=2, axis=-1) ** 2))
    residual_rms = np.sqrt(np.mean((ref_centered - out_centered) ** 2))

    return {
        "valid_trace_count": valid_count,
        "total_trace_count": total_count,
        "median_correlation": float(np.median(correlations)),
        "p05_correlation": float(np.percentile(correlations, 5)),
        "median_peak_to_peak_retention": float(np.median(p2p_retention)),
        "roughness_ratio": float(out_roughness / ref_roughness)
        if ref_roughness > 0
        else float("nan"),
        "residual_rms": float(residual_rms),
    }


def clean_fitted_waveform(
    data: LFIWaveformData,
    *,
    remove_dc: bool = True,
    savgol_window: int = 5,
    savgol_polyorder: int = 3,
) -> LFIWaveformData:
    """Clean fitted traces without changing their sample-axis length."""
    if savgol_window < 3 or savgol_window % 2 == 0:
        raise ValueError("savgol_window must be an odd integer >= 3")
    if savgol_polyorder >= savgol_window:
        raise ValueError("savgol_polyorder must be smaller than savgol_window")
    if data.n_samples < savgol_window:
        raise ValueError(
            f"Trace has {data.n_samples} samples, fewer than window {savgol_window}"
        )

    reference = np.asarray(data.waveform, dtype=np.float64)
    if remove_dc:
        working = reference - np.median(reference, axis=-1, keepdims=True)
    else:
        working = reference.copy()

    cleaned = signal.savgol_filter(
        working,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
        axis=-1,
        mode="interp",
    )

    metadata = deepcopy(data.metadata)
    params = {
        "method": "savgol_on_fitted_trace",
        "remove_dc": remove_dc,
        "savgol_window": savgol_window,
        "savgol_polyorder": savgol_polyorder,
        "sample_axis": -1,
    }
    metadata.setdefault("pipeline", []).append({"waveform_cleaning": params})
    metadata["preservation_metrics"] = evaluate_preservation(working, cleaned)
    metadata.setdefault("version", "0.4.0")

    return LFIWaveformData(
        waveform=cleaned,
        frame_id=data.frame_id,
        side=data.side,
        target_type=data.target_type,
        metadata=metadata,
    )


__all__ = ["clean_fitted_waveform", "evaluate_preservation"]
