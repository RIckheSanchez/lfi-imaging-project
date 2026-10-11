"""Phase unwrapping stage.

Operates on :attr:`LFIRawData.phase`. Two methods are provided to span
both data regimes the project encounters:

* ``"skimage_2d"`` — quality-guided 2D unwrap (:func:`skimage.restoration.unwrap_phase`)
  applied independently to each z-slice. With ``mask = "amplitude_otsu"``
  the residue gate is evaluated, and the unwrap performed, only inside the
  signal region; the released phase is NaN elsewhere. A slice is released
  only when the residue fraction in that region is at most
  ``max_residue_fraction``, i.e. when unwrapping is close to path
  independent. On the Leaf data an independent unwrap agreed with the
  group's saved result on fewer than half of the pixels, so a loose gate
  is not defensible.
* ``"itoh_1d"``    — Itoh-style 1D unwrap along the stack axis via
  :func:`numpy.unwrap`. Use only with a confirmed coordinate and phase
  continuity assumption. Leaf no longer selects this method by default.
* ``"none"``       — pass-through (wrapped phase is preserved).
* ``"validated_reference"`` — reuse a provenance-tagged saved unwrapped
  reference only after checking every pixel against the wrapped input.
  This is reference-assisted processing, not independent unwrap validation.

Disabled by default (``enabled = false``) so the pipeline is byte-stable
for the Progress Seminar repeatability test when no decision has been
taken about phase unwrapping (proposal §3.2.5 mid-year pivot).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from thz_preprocessing.io import LFIRawData
from thz_preprocessing.masking import signal_mask


def _itoh_1d_along_z(phase: np.ndarray) -> np.ndarray:
    if phase.shape[0] < 2:
        # Itoh-1D is meaningful only when n_z > 1; degrade gracefully.
        return phase.copy()
    return np.unwrap(phase, axis=0)


def _wrapped_delta(delta: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * delta))


def _mean_resultant_length(delta: np.ndarray) -> float:
    finite = np.isfinite(delta)
    if not np.any(finite):
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * delta[finite]))))


def _wrapped_phase_quality(phase: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Return scale-free diagnostics used before spatial unwrapping.

    With ``mask``, only 2x2 cells (and neighbour pairs) that lie entirely
    inside the mask are counted.

    A non-zero residue is a 2x2 cell whose wrapped phase differences do
    not close to zero. Residues do not prove that unwrapping is impossible,
    but a dense field of them makes branch placement strongly method
    dependent and is therefore unsafe to release without validation.
    """
    if phase.ndim != 2:
        raise ValueError(f"Expected a 2D phase slice, got shape {phase.shape}")

    dx = _wrapped_delta(np.diff(phase, axis=1))
    dy = _wrapped_delta(np.diff(phase, axis=0))
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        dx = np.where(mask[:, 1:] & mask[:, :-1], dx, np.nan)
        dy = np.where(mask[1:, :] & mask[:-1, :], dy, np.nan)
    if phase.shape[0] < 2 or phase.shape[1] < 2:
        residue_fraction = 0.0
    else:
        right = _wrapped_delta(phase[:-1, 1:] - phase[:-1, :-1])
        down = _wrapped_delta(phase[1:, 1:] - phase[:-1, 1:])
        left = _wrapped_delta(phase[1:, :-1] - phase[1:, 1:])
        up = _wrapped_delta(phase[:-1, :-1] - phase[1:, :-1])
        circulation = right + down + left + up
        valid = np.isfinite(circulation)
        if mask is not None:
            valid &= mask[:-1, :-1] & mask[:-1, 1:] & mask[1:, :-1] & mask[1:, 1:]
        if np.any(valid):
            charges = np.rint(circulation[valid] / (2.0 * np.pi))
            residue_fraction = float(np.mean(charges != 0))
        else:
            residue_fraction = 1.0

    return {
        "residue_fraction": residue_fraction,
        "horizontal_neighbor_coherence": _mean_resultant_length(dx),
        "vertical_neighbor_coherence": _mean_resultant_length(dy),
    }


def _skimage_2d_per_zslice(
    phase: np.ndarray,
    params: dict[str, Any],
    amplitude: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    # Local import: skimage is optional from the perspective of the
    # core pipeline and adds appreciable import latency.
    from skimage.restoration import unwrap_phase

    out = np.empty_like(phase)
    diagnostics: list[dict[str, Any]] = []
    rng_seed = int(params.get("rng_seed", 0))
    max_residue_fraction = float(params.get("max_residue_fraction", 1.0))
    low_quality_action = str(
        params.get("low_quality_action", "keep_wrapped")
    ).lower()
    if not 0.0 <= max_residue_fraction <= 1.0:
        raise ValueError("max_residue_fraction must be between 0 and 1")
    if low_quality_action not in {"keep_wrapped", "unwrap", "raise"}:
        raise ValueError(
            "low_quality_action must be 'keep_wrapped', 'unwrap', or 'raise'"
        )

    mask_method = str(params.get("mask", "none")).lower()
    if mask_method != "none" and amplitude is None:
        raise ValueError("A signal mask needs the amplitude channel")

    for z in range(phase.shape[0]):
        mask = None
        if mask_method != "none":
            mask = signal_mask(amplitude[z], mask_method)
            if mask.all():
                mask = None
        whole = _wrapped_phase_quality(phase[z])
        quality = whole if mask is None else _wrapped_phase_quality(phase[z], mask)
        quality["residue_fraction_whole_image"] = whole["residue_fraction"]
        quality["signal_fraction"] = 1.0 if mask is None else float(mask.mean())
        accepted = quality["residue_fraction"] <= max_residue_fraction
        if accepted or low_quality_action == "unwrap":
            if mask is None:
                out[z] = unwrap_phase(phase[z], rng=rng_seed)
            else:
                unwrapped = unwrap_phase(np.ma.array(phase[z], mask=~mask), rng=rng_seed)
                out[z] = np.ma.filled(unwrapped, np.nan)
            finite = np.isfinite(out[z]) & np.isfinite(phase[z])
            if np.any(finite):
                cycle_adjustments = np.rint(
                    (out[z][finite] - phase[z][finite]) / (2.0 * np.pi)
                )
                adjustment_fraction = float(np.mean(cycle_adjustments != 0))
                max_adjustment = int(np.max(np.abs(cycle_adjustments)))
            else:
                adjustment_fraction = 0.0
                max_adjustment = 0
            if not accepted:
                status = "exploratory_unwrap"
            elif adjustment_fraction == 0.0:
                status = "already_continuous"
            else:
                status = "unwrapped"
        elif low_quality_action == "keep_wrapped":
            out[z] = phase[z]
            status = "kept_wrapped_low_quality"
            adjustment_fraction = None
            max_adjustment = None
        else:
            raise ValueError(
                f"Phase slice {z} residue fraction "
                f"{quality['residue_fraction']:.6f} exceeds configured limit "
                f"{max_residue_fraction:.6f}"
            )

        diagnostics.append(
            {
                "slice_index": z,
                **quality,
                "max_residue_fraction": max_residue_fraction,
                "mask": mask_method,
                "quality_gate_passed": accepted,
                "status": status,
                "cycle_adjustment_fraction": adjustment_fraction,
                "max_abs_cycle_adjustment": max_adjustment,
            }
        )
    return out, diagnostics


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

    diagnostics: list[dict[str, Any]] = []
    if not enabled:
        out = data.phase.copy()
        status = "disabled"
    elif data.metadata.get("phase_encoding_status") == "unconfirmed":
        out = data.phase.copy()
        status = "withheld_unconfirmed_encoding"
    elif method == "none":
        out = data.phase.copy()
        status = "pass_through"
    elif method == "skimage_2d":
        out, diagnostics = _skimage_2d_per_zslice(data.phase, params, data.amplitude)
        statuses = {item["status"] for item in diagnostics}
        status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
    elif method == "validated_reference":
        reference = np.asarray(data.metadata.get("phase_reference"), dtype=np.float64)
        if reference.shape != data.phase.shape or not np.isfinite(reference).all():
            raise ValueError("A finite, shape-matched saved phase_reference is required")
        error = np.angle(np.exp(1j * (reference - data.phase)))
        maximum = float(np.max(np.abs(error)))
        if maximum > 1e-9:
            raise ValueError("Saved phase reference does not match the wrapped input modulo 2pi")
        if not data.metadata.get("phase_reference_source"):
            raise ValueError("Saved phase reference requires explicit source provenance")
        out = reference.copy()
        status = "reference_unwrapped"
        diagnostics = [{"slice_index": i, **_wrapped_phase_quality(data.phase[i]),
                        "status": status, "max_modulo_error_radians": maximum,
                        "validation_scope": "historical reference consistency, not physical ground truth"}
                       for i in range(data.n_z)]
    elif method == "itoh_1d":
        out = _itoh_1d_along_z(data.phase)
        status = "unwrapped" if data.phase.shape[0] > 1 else "insufficient_z"
    else:
        raise ValueError(f"Unknown phase-unwrapping method: {method!r}")

    new_meta = deepcopy(data.metadata)
    stage_metadata = dict(params)
    if diagnostics:
        stage_metadata["quality_diagnostics"] = diagnostics
    new_meta.setdefault("pipeline", []).append(
        {"phase_unwrapping": stage_metadata}
    )
    new_meta["phase_unwrapping_status"] = {
        "enabled": enabled,
        "method": method,
        "status": status,
        "quality_diagnostics": diagnostics,
        "reference_source": data.metadata.get("phase_reference_source") if method == "validated_reference" else None,
    }

    return LFIRawData(
        amplitude=data.amplitude.copy(),
        phase=out,
        z_indices=list(data.z_indices),
        target_type=data.target_type,
        metadata=new_meta,
    )
