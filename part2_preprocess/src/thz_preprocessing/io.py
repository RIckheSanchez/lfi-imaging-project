"""Input/output adapters for THz LFI datasets.

This module defines the shared :class:`LFIRawData` data contract used as
the *input* type to :func:`thz_preprocessing.api.run_pipeline`, and
provides temporary file loaders for the pilot datasets supplied to the
project:

* ``data/skin/*.hdf5`` — legacy image-level demodulated scans (single frame).
* ``data/leaf/leaf1.mat`` — stored image stack (MATLAB v5/v6/v7).
* ``data/vegetables/frame-*.hdf5`` — 2026 RDM frame files with image-level
  amplitude/phase datasets and fitted line-of-best-fit waveform arrays.

The contract mirrors teammate A's ``LFIRawData`` dataclass (data-import
module) so that the pipeline can be wired against the shared platform
loader without further changes once that loader ships.

All loaders return :class:`LFIRawData` with shape ``(n_z, H, W)``:

* ``n_z`` — number of frames / sweep samples. ``n_z == 1`` for
  single-frame image-level scans (e.g. current Skin and Vegetable pilots).
* ``H``, ``W`` — spatial raster dimensions.

Phase is always returned in **radians** in ``[-pi, pi)`` (NumPy
convention) so that downstream operators do not need to know how the
underlying file encoded phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import re

import numpy as np


@dataclass
class LFIRawData:
    """Container for one THz LFI scan, post data-import, pre-processing.

    The field layout mirrors the data-import module's output contract
    so that this thesis's pre-processing module can consume the shared
    loader's product directly.
    """

    amplitude: np.ndarray
    """Demodulated amplitude, shape ``(n_z, H, W)``, float."""

    phase: np.ndarray
    """Input wrapped phase in radians, shape ``(n_z, H, W)``. Pipeline
    output may be unwrapped; inspect phase_unwrapping_status and encoding
    provenance. Relative spatial phase is a separate derived product."""

    z_indices: list[int]
    """Per-frame z-axis index (frame id / depth slot). ``len() == n_z``."""

    target_type: str
    """Coarse scan-type tag used to select target-specific defaults
    (``"Skin" | "Leaf" | "Chip" | ...``)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form provenance: source path, original dtype, phase encoding
    convention, anything useful for downstream debugging."""

    @property
    def n_z(self) -> int:
        return self.amplitude.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.amplitude.shape)


@dataclass
class LFIWaveformData:
    """Per-pixel fitted LFI traces from the 2026 RDM HDF5 schema.

    ``RX_LINE_OF_BEST_FIT`` is an acquisition-system intermediate, not the
    raw ADC stream. The first two axes align with the image raster and the
    last axis is the fitted-trace sample axis.
    """

    waveform: np.ndarray
    """Fitted trace array, shape ``(H, W, n_samples)``, float."""

    frame_id: int
    side: str
    target_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.waveform.shape)

    @property
    def n_samples(self) -> int:
        return int(self.waveform.shape[-1])


def _u16_phase_to_radians(
    phase_u16: np.ndarray,
    encoding: str = "0_2pi",
) -> np.ndarray:
    """Convert a uint16-encoded wrapped phase to radians in [-pi, pi).

    Parameters
    ----------
    phase_u16
        Phase array stored as uint16 covering the full ``[0, 65536)``
        range.
    encoding
        How the uint16 range maps to phase:

        * ``"0_2pi"`` (default) — ``[0, 65536) -> [0, 2*pi)``, output
          remapped to ``[-pi, pi)`` for NumPy compatibility.
        * ``"neg_pi_pi"`` — ``[0, 65536) -> [-pi, pi)`` directly.

    Returns
    -------
    np.ndarray
        Phase in radians, ``float64``, shape unchanged, range ``[-pi, pi)``.
    """
    phase = phase_u16.astype(np.float64) / 65536.0  # [0, 1)
    if encoding == "0_2pi":
        phase = phase * (2.0 * np.pi)  # [0, 2*pi)
        return ((phase + np.pi) % (2.0 * np.pi)) - np.pi  # [-pi, pi)
    if encoding == "neg_pi_pi":
        return phase * (2.0 * np.pi) - np.pi
    raise ValueError(f"Unknown phase encoding: {encoding!r}")


def _i16_phase_to_radians(
    phase_i16: np.ndarray,
    encoding: str = "signed_16bit_pi",
) -> np.ndarray:
    """Convert an int16-encoded wrapped phase to radians in [-pi, pi).

    RDM files use an int16 container, but normal complete frames inspected
    so far use signed 14-bit phase values. The scan-start frame 20 spans the
    full int16 range, so both encodings are supported explicitly.
    """
    phase = phase_i16.astype(np.float64)
    if encoding in {"signed_pi", "signed_16bit_pi"}:
        phase = (phase / 32768.0) * np.pi
        return ((phase + np.pi) % (2.0 * np.pi)) - np.pi
    if encoding == "signed_14bit_pi":
        phase = (phase / 8192.0) * np.pi
        return ((phase + np.pi) % (2.0 * np.pi)) - np.pi
    raise ValueError(f"Unknown int16 phase encoding: {encoding!r}")


def _resolve_i16_phase_encoding(
    phase_i16: np.ndarray,
    requested: str,
) -> str:
    """Validate an explicitly selected RDM fixed-point phase scale.

    A value range cannot identify a fixed-point scale: a valid signed-16
    frame can happen to contain only values inside the signed-14 interval.
    The loader therefore requires the acquisition convention explicitly and
    uses observed bounds only as a consistency check.
    """
    if requested == "auto":
        raise ValueError(
            "Automatic int16 phase-scale inference is unsafe. Pass "
            "phase_encoding='signed_14bit_pi' for confirmed complete 2026 "
            "RDM frames, or an explicit diagnostic override."
        )

    resolved = "signed_16bit_pi" if requested == "signed_pi" else requested
    if resolved not in {"signed_14bit_pi", "signed_16bit_pi"}:
        raise ValueError(f"Unknown int16 phase encoding: {requested!r}")

    phase_min = int(np.min(phase_i16))
    phase_max = int(np.max(phase_i16))
    if resolved == "signed_14bit_pi" and (
        phase_min < -8192 or phase_max > 8191
    ):
        raise ValueError(
            "Phase codes fall outside the confirmed signed-14 interval "
            f"[-8192, 8191]: observed [{phase_min}, {phase_max}]. The frame "
            "is anomalous or uses a different acquisition convention; do "
            "not reinterpret it automatically."
        )
    return resolved


def _signed_14bit_phase_to_radians(phase: np.ndarray) -> np.ndarray:
    """Convert the Leaf signed 14-bit fixed-point phase to radians.

    This convention is used by complete 2026 RDM image frames. Leaf's
    saved references establish a different dataset-specific conversion.
    """
    radians = (np.asarray(phase, dtype=np.float64) / 8192.0) * np.pi
    return ((radians + np.pi) % (2.0 * np.pi)) - np.pi


def _leaf_reference_phase_to_radians(phase: np.ndarray) -> np.ndarray:
    """Decode Leaf consistently with all 91 saved U references.

    Every pixel satisfies U = L_P*pi/4096 - pi modulo 2*pi to <2e-14
    radians. This is a dataset-specific reference equivalence, not proof
    of absolute phase calibration or the 2026 HDF5 encoding convention.
    """
    phase = np.asarray(phase, dtype=np.float64)
    if not np.isfinite(phase).all() or np.any((phase < -8192) | (phase > 8191)):
        raise ValueError("Leaf reference encoding requires finite codes in [-8192, 8191]")
    radians = phase * (np.pi / 4096.0) - np.pi
    return ((radians + np.pi) % (2.0 * np.pi)) - np.pi


def _to_python_scalar(value: Any) -> Any:
    """Convert HDF5 / NumPy scalar values into JSON-friendly Python values."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_optional_scalar(h5_file: Any, dataset_path: str) -> Any:
    """Read a scalar HDF5 dataset if present, otherwise return ``None``."""
    try:
        return _to_python_scalar(h5_file[dataset_path][()])
    except KeyError:
        return None


_RDM_FRAME_KEY_RE = re.compile(
    r"^frame_(?P<frame>\d+)_qcl_(?P<qcl>\d+)_(?P<side>left|right)$"
)


def _infer_target_type(path: Path) -> str:
    lowered = {part.lower() for part in path.parts}
    if "vegetables" in lowered or "vegetable" in lowered:
        return "Vegetable"
    if "skin" in lowered:
        return "Skin"
    if "leaf" in lowered:
        return "Leaf"
    return "Scan"


def load_skin_hdf5(
    path: str | Path,
    frame_id: int | None = None,
    qcl: str = "QCL0",
    side: str = "Left",
    phase_encoding: str = "0_2pi",
) -> LFIRawData:
    """Load a single skin scan from the supplied HDF5 layout.

    Expected internal structure (one frame per file, group-per-frame)::

        /frames/{frame_id}/Image Data/{qcl}/{side}/
            IMAGE_AMPLITUDE   uint16   (H, W)
            IMAGE_PHASE       uint16   (H, W)   wrapped phase
            PEAK_TO_PEAK      uint16   (H, W)   (kept in metadata, not
                                                 part of the LFIRawData
                                                 contract)

    Parameters
    ----------
    path
        Path to the ``.hdf5`` file.
    frame_id
        Which ``frames/<id>`` group to read. If ``None``, the first
        group encountered under ``frames/`` is used (the supplied pilot
        scan contains only ``frames/40``).
    qcl, side
        Which laser / side channel to read.
    phase_encoding
        How the uint16 phase is encoded (see :func:`_u16_phase_to_radians`).
        Pending confirmation from the supervisor; default assumes
        ``[0, 2*pi)``.

    Returns
    -------
    LFIRawData
        ``amplitude`` and ``phase`` shaped ``(1, H, W)``. ``z_indices``
        is ``[frame_id]``. ``target_type`` is ``"Skin"``.
    """
    import h5py  # local import keeps optional deps off the critical path

    path = Path(path)
    with h5py.File(path, "r") as f:
        frames = f["frames"]
        if frame_id is None:
            available = sorted(frames.keys(), key=lambda k: int(k))
            if not available:
                raise ValueError(f"No frames found in {path}")
            frame_key = available[0]
            frame_id = int(frame_key)
        else:
            frame_key = str(frame_id)
            if frame_key not in frames:
                raise KeyError(f"Frame {frame_id} not in {path}")

        group_path = f"frames/{frame_key}/Image Data/{qcl}/{side}"
        amp_u16 = f[f"{group_path}/IMAGE_AMPLITUDE"][()]
        phs_u16 = f[f"{group_path}/IMAGE_PHASE"][()]
        try:
            pp_u16 = f[f"{group_path}/PEAK_TO_PEAK"][()]
        except KeyError:
            pp_u16 = None

    amplitude = amp_u16.astype(np.float64)[None, :, :]  # (1, H, W)
    phase = _u16_phase_to_radians(phs_u16, encoding=phase_encoding)[None, :, :]

    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_format": "hdf5",
        "qcl": qcl,
        "side": side,
        "phase_encoding": phase_encoding,
        "original_dtype": "uint16",
        "phase_encoding_status": "unconfirmed",
        "phase_scale_warning": "Legacy Skin scale is an assumption; uint16 reinterpreted as int16 occupies [-8192,8191]. This does not establish the conversion scale.",
    }
    if pp_u16 is not None:
        metadata["peak_to_peak"] = pp_u16  # not in contract, but kept

    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=[int(frame_id)],
        target_type="Skin",
        metadata=metadata,
    )


def load_rdm_hdf5_frame(
    path: str | Path,
    frame_id: int | None = None,
    qcl: int = 1,
    side: str = "left",
    phase_encoding: str = "signed_14bit_pi",
    target_type: str | None = None,
    spatial_slice: tuple[slice, slice] | None = None,
) -> LFIRawData:
    """Load one 2026-style RDM HDF5 frame into ``LFIRawData``.

    Expected internal structure for the inspected Semester 2 vegetable data::

        /data/IMAGE_AMPLITUDE/frame_{id}_qcl_{qcl}_{side}    uint16 (H, W)
        /data/IMAGE_PHASE/frame_{id}_qcl_{qcl}_{side}        int16  (H, W)
        /data/PEAK_TO_PEAK/frame_{id}_qcl_{qcl}_{side}       uint16 (H, W)
        /data/RX_LINE_OF_BEST_FIT/frame_{id}_{side}          int16  (H, W, N)

    ``RX_LINE_OF_BEST_FIT`` is not loaded into :class:`LFIRawData` because the
    downstream contract currently consumes image-level amplitude and phase.
    Use :func:`load_rdm_fitted_waveform` for that acquisition-system
    intermediate. Its dataset name, shape and dtype are also recorded here.
    """
    import h5py  # local import keeps optional deps off the critical path

    path = Path(path)
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    with h5py.File(path, "r") as f:
        amp_group = f["data/IMAGE_AMPLITUDE"]
        candidates: list[tuple[int, str]] = []
        for key in amp_group.keys():
            if key.endswith("_BMP"):
                continue
            match = _RDM_FRAME_KEY_RE.match(key)
            if not match:
                continue
            key_frame = int(match.group("frame"))
            key_qcl = int(match.group("qcl"))
            key_side = match.group("side")
            if key_qcl == qcl and key_side == side:
                candidates.append((key_frame, key))

        if not candidates:
            raise ValueError(
                f"No IMAGE_AMPLITUDE dataset found in {path} for qcl={qcl}, side={side!r}"
            )

        candidates.sort(key=lambda item: item[0])
        if frame_id is None:
            frame_id, dataset_key = candidates[0]
        else:
            matching = [key for key_frame, key in candidates if key_frame == frame_id]
            if not matching:
                available = [key_frame for key_frame, _ in candidates]
                raise KeyError(
                    f"Frame {frame_id} not found in {path}; available frame ids: {available}"
                )
            dataset_key = matching[0]

        amp = f[f"data/IMAGE_AMPLITUDE/{dataset_key}"][()]
        phase_raw = f[f"data/IMAGE_PHASE/{dataset_key}"][()]

        peak_to_peak_path = f"data/PEAK_TO_PEAK/{dataset_key}"
        peak_to_peak_shape = None
        peak_to_peak_dtype = None
        if peak_to_peak_path in f:
            peak_to_peak_shape = tuple(int(v) for v in f[peak_to_peak_path].shape)
            peak_to_peak_dtype = str(f[peak_to_peak_path].dtype)

        waveform_key = f"frame_{frame_id}_{side}"
        waveform_path = f"data/RX_LINE_OF_BEST_FIT/{waveform_key}"
        waveform_shape = None
        waveform_dtype = None
        if waveform_path in f:
            waveform_shape = tuple(int(v) for v in f[waveform_path].shape)
            waveform_dtype = str(f[waveform_path].dtype)

        acquisition = {
            "scan_identifier": _read_optional_scalar(f, "scan_identifier"),
            "timestamp": _read_optional_scalar(f, "thz/info/timestamp"),
            "image_rows": _read_optional_scalar(f, "thz/info/mcp/imageRows"),
            "image_columns": _read_optional_scalar(f, "thz/info/mcp/imageColumns"),
            "depth_frames": _read_optional_scalar(f, "thz/info/mcp/depthFrames"),
            "temperature_c": _read_optional_scalar(f, "thz/info/environment/temperature"),
            "humidity_percent": _read_optional_scalar(f, "thz/info/environment/humidity"),
            "pressure": _read_optional_scalar(f, "thz/info/environment/pressure"),
            "fft_phase_logged": _read_optional_scalar(f, "thz/info/mcp/logFftPhase"),
            "selected_fft_bin": _read_optional_scalar(f, "thz/info/mcp/selFftBin"),
        }

    phase_code_bounds = [int(np.min(phase_raw)), int(np.max(phase_raw))]
    resolved_phase_encoding = _resolve_i16_phase_encoding(
        phase_raw, phase_encoding
    )
    if spatial_slice is not None:
        sh, sw = spatial_slice
        amp = amp[sh, sw]
        phase_raw = phase_raw[sh, sw]

    amplitude = amp.astype(np.float64)[None, :, :]
    phase = _i16_phase_to_radians(
        phase_raw, encoding=resolved_phase_encoding
    )[None, :, :]

    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_format": "hdf5_rdm_2026_frame",
        "schema": "data/IMAGE_*",
        "dataset_key": dataset_key,
        "qcl": qcl,
        "side": side,
        "phase_encoding_requested": phase_encoding,
        "phase_encoding": resolved_phase_encoding,
        "phase_code_bounds_full_frame": phase_code_bounds,
        "phase_unit": "radians",
        "phase_encoding_status": "explicit_validated",
        "amplitude_dtype": str(amp.dtype),
        "phase_dtype": str(phase_raw.dtype),
        "spatial_slice": repr(spatial_slice),
        "peak_to_peak_dataset": peak_to_peak_path if peak_to_peak_shape else None,
        "peak_to_peak_shape": peak_to_peak_shape,
        "peak_to_peak_dtype": peak_to_peak_dtype,
        "waveform_dataset": waveform_path if waveform_shape else None,
        "waveform_shape": waveform_shape,
        "waveform_dtype": waveform_dtype,
        "acquisition": acquisition,
    }

    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=[int(frame_id)],
        target_type=target_type or _infer_target_type(path),
        metadata=metadata,
    )


def _read_experiment_details(path: Path) -> dict[str, Any]:
    """Parse the small RDM experiment-details text file."""
    text = path.read_text(encoding="utf-8", errors="replace")

    def _number(pattern: str) -> float | None:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    return {
        "source_path": str(path),
        "start_z_mm": _number(r"start\s+z\s+position\s*\(mm\)\s*:\s*([-+]?\d+(?:\.\d+)?)"),
        "stop_z_mm": _number(r"stop\s+z\s+position\s*\(mm\)\s*:\s*([-+]?\d+(?:\.\d+)?)"),
        "increment_mm": _number(r"(?:increments?|imcrements?)\s*\(mm\)\s*:\s*([-+]?\d+(?:\.\d+)?)"),
        "text": text.strip(),
    }


def load_rdm_hdf5_volume(
    paths: Sequence[str | Path],
    *,
    qcl: int = 1,
    side: str = "left",
    phase_encoding: str = "signed_14bit_pi",
    target_type: str | None = None,
    spatial_slice: tuple[slice, slice] | None = None,
    experiment_details_path: str | Path | None = None,
    z_reference_frame_id: int | None = None,
) -> LFIRawData:
    """Load multiple compatible RDM frame files as one depth volume.

    Frames are sorted by their internal frame id and concatenated into
    ``(n_frames, H, W)``. A shape mismatch raises an error so an incomplete
    first frame cannot be silently mixed with later full frames.
    """
    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("At least one RDM HDF5 path is required")

    frames = [
        load_rdm_hdf5_frame(
            path,
            qcl=qcl,
            side=side,
            phase_encoding=phase_encoding,
            target_type=target_type,
            spatial_slice=spatial_slice,
        )
        for path in path_list
    ]
    frames.sort(key=lambda frame: frame.z_indices[0])

    expected_shape = frames[0].shape[1:]
    mismatches = [
        (frame.z_indices[0], frame.shape[1:])
        for frame in frames
        if frame.shape[1:] != expected_shape
    ]
    if mismatches:
        all_shapes = [(frame.z_indices[0], frame.shape[1:]) for frame in frames]
        raise ValueError(
            "RDM frames have incompatible spatial shapes; exclude incomplete "
            f"frames or provide a common spatial_slice. Frames: {all_shapes}"
        )

    frame_ids = [int(frame.z_indices[0]) for frame in frames]
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError(f"Duplicate frame ids are not allowed: {frame_ids}")

    resolved_encodings = {
        str(frame.metadata["phase_encoding"]) for frame in frames
    }
    if len(resolved_encodings) != 1:
        raise ValueError(
            "RDM frames resolve to different phase encodings; inspect or "
            f"exclude anomalous frames before stacking: {sorted(resolved_encodings)}"
        )
    resolved_phase_encoding = next(iter(resolved_encodings))

    amplitude = np.concatenate([frame.amplitude for frame in frames], axis=0)
    phase = np.concatenate([frame.phase for frame in frames], axis=0)

    details = None
    z_positions_mm = None
    if experiment_details_path is not None:
        details = _read_experiment_details(Path(experiment_details_path))
        start_z = details.get("start_z_mm")
        increment = details.get("increment_mm")
        if start_z is not None and increment is not None:
            reference_id = (
                int(z_reference_frame_id)
                if z_reference_frame_id is not None
                else min(frame_ids)
            )
            z_positions_mm = [
                round(float(start_z + (frame_id - reference_id) * increment), 10)
                for frame_id in frame_ids
            ]
        else:
            reference_id = None
    else:
        reference_id = None

    metadata: dict[str, Any] = {
        "source_format": "hdf5_rdm_2026_volume",
        "source_paths": [str(frame.metadata["source_path"]) for frame in frames],
        "frame_ids": frame_ids,
        "frame_count": len(frames),
        "spatial_shape": list(expected_shape),
        "qcl": qcl,
        "side": side.lower(),
        "phase_encoding_requested": phase_encoding,
        "phase_encoding": resolved_phase_encoding,
        "phase_unit": "radians",
        "phase_encoding_status": "explicit_validated",
        "spatial_slice": repr(spatial_slice),
        "z_positions_mm": z_positions_mm,
        "z_reference_frame_id": reference_id,
        "experiment_details": details,
        "per_frame_metadata": [frame.metadata for frame in frames],
    }

    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=frame_ids,
        target_type=target_type or frames[0].target_type,
        metadata=metadata,
    )


_RDM_WAVEFORM_KEY_RE = re.compile(
    r"^frame_(?P<frame>\d+)_(?P<side>left|right)$"
)


def load_rdm_fitted_waveform(
    path: str | Path,
    frame_id: int | None = None,
    side: str = "left",
    target_type: str | None = None,
    spatial_slice: tuple[slice, slice] | None = None,
) -> LFIWaveformData:
    """Load the per-pixel fitted LFI traces from a 2026 RDM frame file.

    The stored dataset is identified as a fitted waveform rather than a raw
    ADC waveform because the acquisition metadata enables and logs LOBF
    (line of best fit) while all ``logRx*`` raw/intermediate streams are off.
    For ``frame-20.hdf5`` the native shape is ``(400, 1300, 184)``; therefore
    each image pixel has a 184-sample fitted trace.
    """
    import h5py

    path = Path(path)
    side = side.lower()
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    with h5py.File(path, "r") as f:
        group = f["data/RX_LINE_OF_BEST_FIT"]
        candidates: list[tuple[int, str]] = []
        for key in group.keys():
            match = _RDM_WAVEFORM_KEY_RE.match(key)
            if match and match.group("side") == side:
                candidates.append((int(match.group("frame")), key))

        if not candidates:
            raise ValueError(
                f"No RX_LINE_OF_BEST_FIT dataset found in {path} for side={side!r}"
            )
        candidates.sort(key=lambda item: item[0])

        if frame_id is None:
            frame_id, dataset_key = candidates[0]
        else:
            matches = [key for key_frame, key in candidates if key_frame == frame_id]
            if not matches:
                available = [key_frame for key_frame, _ in candidates]
                raise KeyError(
                    f"Frame {frame_id} not found in {path}; available frame ids: {available}"
                )
            dataset_key = matches[0]

        dataset_path = f"data/RX_LINE_OF_BEST_FIT/{dataset_key}"
        dataset = f[dataset_path]
        if dataset.ndim != 3:
            raise ValueError(
                f"Expected fitted waveform shape (H, W, samples), got {dataset.shape}"
            )
        if spatial_slice is None:
            waveform_raw = dataset[()]
        else:
            sh, sw = spatial_slice
            waveform_raw = dataset[sh, sw, :]

        acquisition_flags = {
            "en_lobf": _read_optional_scalar(f, "thz/info/mcp/enLobf"),
            "log_lobf": _read_optional_scalar(f, "thz/info/mcp/logLobf"),
            "log_rx_raw": _read_optional_scalar(f, "thz/info/mcp/logRxRaw"),
            "log_rx_avg": _read_optional_scalar(f, "thz/info/mcp/logRxAvg"),
            "log_rx_dec": _read_optional_scalar(f, "thz/info/mcp/logRxDec"),
            "log_rx_sub": _read_optional_scalar(f, "thz/info/mcp/logRxSub"),
            "selected_fft_bin": _read_optional_scalar(f, "thz/info/mcp/selFftBin"),
            "fft_phase_logged": _read_optional_scalar(f, "thz/info/mcp/logFftPhase"),
            "fft_magnitude_logged": _read_optional_scalar(f, "thz/info/mcp/logFftMag"),
        }

    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_format": "hdf5_rdm_2026_frame",
        "dataset_path": dataset_path,
        "original_dtype": str(waveform_raw.dtype),
        "spatial_slice": repr(spatial_slice),
        "waveform_kind": "fitted_line_of_best_fit",
        "waveform_semantics_status": "confirmed_from_dataset_name_and_acquisition_flags",
        "axis_order": ["image_row", "image_column", "trace_sample"],
        "sample_count": int(waveform_raw.shape[-1]),
        "acquisition_flags": acquisition_flags,
    }

    return LFIWaveformData(
        waveform=waveform_raw.astype(np.float64),
        frame_id=int(frame_id),
        side=side,
        target_type=target_type or _infer_target_type(path),
        metadata=metadata,
    )


def load_leaf_mat(
    path: str | Path,
    amplitude_var: str = "L_A",
    phase_var: str = "L_P",
    spatial_slice: tuple[slice, slice] | None = None,
    z_slice: slice | None = None,
    phase_encoding: str = "leaf_reference_pi4096_offset_pi",
) -> LFIRawData:
    """Load a single leaf scan from the supplied MATLAB ``.mat`` file.

    The file is the non-HDF5 v5/v6/v7 MATLAB format; whole variables
    are materialised into memory by :mod:`scipy.io`. Use the ``slice``
    arguments to restrict the loaded volume for memory-bound pilots.

    Variable layout (see ``data/leaf/README.md``)::

        L_A    (H, W, n_frames) float64   amplitude
        L_P    (H, W, n_frames) float64   encoded wrapped phase
        L_P_U  (H, W, n_unwr)   float64   group's ad-hoc unwrapped phase
        L_P_U_M (H, W, n_unwr)  float64   modified spatial phase reference

    The returned tensor is transposed to ``(stored_frame_index, H, W)``.
    The physical third-axis coordinate is not established by this file.

    Parameters
    ----------
    path
        Path to ``leaf1.mat``.
    amplitude_var, phase_var
        Variable names to read. Override only if loading an alternate
        layout (e.g. ``"L_P_U"`` instead of ``"L_P"``).
    spatial_slice
        Optional ``(slice_H, slice_W)`` to crop the spatial raster
        before transposing. Recommended for the pilot demo to keep
        memory + runtime small.
    z_slice
        Optional ``slice`` along the stored-image axis.
    phase_encoding
        Default ``"leaf_reference_pi4096_offset_pi"`` is numerically
        validated against all pixels of saved references 0..90. The old
        ``"signed_14bit_pi"`` remains an explicit diagnostic override,
        not a reference-consistent Leaf default. ``"radians"`` is for
        a variable explicitly known to be in radians.

    Returns
    -------
    LFIRawData
        ``amplitude`` and ``phase`` shaped ``(n_z, H, W)``. ``z_indices``
        preserves original stored-image indices after slicing. The leaf
        ``.mat`` does not carry acquisition timestamps, so positions are used as
        the surrogate id. ``target_type`` is ``"Leaf"``.
    """
    import scipy.io as sio

    path = Path(path)
    mat = sio.loadmat(
        path,
        variable_names=[amplitude_var, phase_var],
        squeeze_me=False,
        verify_compressed_data_integrity=False,
    )

    amp_hwz = np.asarray(mat[amplitude_var], dtype=np.float64)
    phs_hwz = np.asarray(mat[phase_var], dtype=np.float64)
    if amp_hwz.shape != phs_hwz.shape:
        raise ValueError(
            f"Amplitude / phase shape mismatch: "
            f"{amp_hwz.shape} vs {phs_hwz.shape}"
        )
    if amp_hwz.ndim != 3:
        raise ValueError("Expected three-dimensional Leaf image arrays")
    stored_indices = np.arange(amp_hwz.shape[2])

    if spatial_slice is not None:
        sh, sw = spatial_slice
        amp_hwz = amp_hwz[sh, sw, :]
        phs_hwz = phs_hwz[sh, sw, :]
    if z_slice is not None:
        amp_hwz = amp_hwz[:, :, z_slice]
        phs_hwz = phs_hwz[:, :, z_slice]
        stored_indices = stored_indices[z_slice]

    amplitude = np.ascontiguousarray(np.transpose(amp_hwz, (2, 0, 1)))
    if phase_encoding == "leaf_reference_pi4096_offset_pi":
        phs_hwz = _leaf_reference_phase_to_radians(phs_hwz)
        phase_unit = "radians"
        encoding_status = "reference_equivalence_validated"
    elif phase_encoding == "signed_14bit_pi":
        phs_hwz = _signed_14bit_phase_to_radians(phs_hwz)
        phase_unit = "radians"
        encoding_status = "legacy_diagnostic_not_reference_consistent"
    elif phase_encoding == "radians":
        phase_unit = "radians"
        encoding_status = "provided_by_caller"
    else:
        raise ValueError(f"Unknown Leaf phase encoding: {phase_encoding!r}")

    phase = np.ascontiguousarray(np.transpose(phs_hwz, (2, 0, 1)))

    n_z = amplitude.shape[0]
    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_format": "matlab_v5",
        "amplitude_var": amplitude_var,
        "phase_var": phase_var,
        "spatial_slice": repr(spatial_slice),
        "z_slice": repr(z_slice),
        "phase_encoding": phase_encoding,
        "phase_unit": phase_unit,
        "phase_encoding_status": encoding_status,
        "axis_order": ["stored_image_index", "spatial_axis_0", "spatial_axis_1"],
        "third_axis_physical_coordinate": "unresolved; no cross-frame processing by default",
        "reference_mapping": "same index 0..90 validated modulo 2pi; some equivalent candidates are ambiguous",
        "reference_mapping_evidence": "output/review_rdm/leaf_semantics_audit.json",
    }

    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=stored_indices.tolist(),
        target_type="Leaf",
        metadata=metadata,
    )


__all__ = [
    "LFIRawData",
    "LFIWaveformData",
    "load_skin_hdf5",
    "load_rdm_hdf5_frame",
    "load_rdm_hdf5_volume",
    "load_rdm_fitted_waveform",
    "load_leaf_mat",
]
