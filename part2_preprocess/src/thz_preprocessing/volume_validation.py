"""Quantitative preservation checks for multi-frame LFI volumes."""

from __future__ import annotations

from typing import Any

import numpy as np

from thz_preprocessing.io import LFIRawData


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if not np.any(valid):
        return 0.0
    x = x[valid]
    y = y[valid]
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.sqrt(np.sum(x**2) * np.sum(y**2))
    if denominator == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.sum(x * y) / denominator)


def _minmax(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float64)
    low = float(np.nanmin(frame))
    span = float(np.nanmax(frame) - low)
    if span == 0:
        return np.zeros_like(frame)
    return (frame - low) / span


def evaluate_volume_preservation(
    raw: LFIRawData,
    processed: LFIRawData,
) -> dict[str, Any]:
    """Compare raw and processed volumes without requiring clinical labels.

    Phase is rewrapped and compared modulo ``2*pi``. This is an integrity
    check only: it catches accidental phase changes, but cannot establish
    that the selected unwrap branches are physically correct.
    """
    if raw.shape != processed.shape:
        raise ValueError(f"Volume shape mismatch: {raw.shape} vs {processed.shape}")
    if raw.z_indices != processed.z_indices:
        raise ValueError("Raw and processed z_indices differ")

    amplitude_correlations = []
    amplitude_normalized_rmse = []
    phase_modulo_consistency = []
    phase_rewrap_rmse_radians = []
    unwrapped_phase_p01_p99_span_radians = []
    for index in range(raw.n_z):
        raw_amp = raw.amplitude[index]
        out_amp = processed.amplitude[index]
        amplitude_correlations.append(_pearson(raw_amp, out_amp))
        amplitude_normalized_rmse.append(
            float(np.sqrt(np.mean((_minmax(raw_amp) - _minmax(out_amp)) ** 2)))
        )

        phase_delta = raw.phase[index] - processed.phase[index]
        valid_phase = np.isfinite(phase_delta)
        if np.any(valid_phase):
            wrapped_error = np.angle(np.exp(1j * phase_delta[valid_phase]))
            phase_modulo_consistency.append(
                float(np.abs(np.mean(np.exp(1j * wrapped_error))))
            )
            phase_rewrap_rmse_radians.append(
                float(np.sqrt(np.mean(wrapped_error**2)))
            )
            low, high = np.percentile(
                processed.phase[index][np.isfinite(processed.phase[index])],
                [1, 99],
            )
            unwrapped_phase_p01_p99_span_radians.append(float(high - low))
        else:
            phase_modulo_consistency.append(0.0)
            phase_rewrap_rmse_radians.append(float("nan"))
            unwrapped_phase_p01_p99_span_radians.append(float("nan"))

    adjacent_amplitude_correlations = [
        _pearson(processed.amplitude[index], processed.amplitude[index + 1])
        for index in range(raw.n_z - 1)
    ]

    return {
        "frame_ids": list(raw.z_indices),
        "frame_count": raw.n_z,
        "amplitude_raw_vs_processed_pearson": amplitude_correlations,
        "amplitude_normalized_rmse": amplitude_normalized_rmse,
        "phase_modulo_consistency": phase_modulo_consistency,
        "phase_rewrap_rmse_radians": phase_rewrap_rmse_radians,
        "phase_p01_p99_span_radians": unwrapped_phase_p01_p99_span_radians,
        "phase_unwrapping_status": processed.metadata.get(
            "phase_unwrapping_status"
        ),
        "adjacent_processed_amplitude_pearson": adjacent_amplitude_correlations,
        "summary": {
            "median_amplitude_pearson": float(np.median(amplitude_correlations)),
            "max_amplitude_normalized_rmse": float(
                np.max(amplitude_normalized_rmse)
            ),
            "min_phase_modulo_consistency": float(
                np.min(phase_modulo_consistency)
            ),
            "max_phase_rewrap_rmse_radians": float(
                np.nanmax(phase_rewrap_rmse_radians)
            ),
            "median_adjacent_amplitude_pearson": float(
                np.median(adjacent_amplitude_correlations)
            )
            if adjacent_amplitude_correlations
            else None,
            "pipeline_total_seconds": processed.metadata.get(
                "timings_seconds", {}
            ).get("total"),
        },
    }


__all__ = ["evaluate_volume_preservation"]
