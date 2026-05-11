"""Input/output adapters for THz LFI datasets.

This module defines the shared :class:`LFIRawData` data contract used as
the *input* type to :func:`thz_preprocessing.api.run_pipeline`, and
provides temporary file loaders for the two pilot datasets supplied to
the Progress Seminar:

* ``data/skin/*.hdf5`` — image-level demodulated scans (single frame).
* ``data/leaf/leaf1.mat`` — sweep-axis volumetric scans (MATLAB v5/v6/v7).

The contract mirrors teammate A's ``LFIRawData`` dataclass (data-import
module) so that the pipeline can be wired against the shared platform
loader without further changes once that loader ships.

All loaders return :class:`LFIRawData` with shape ``(n_z, H, W)``:

* ``n_z`` — number of frames / sweep samples. ``n_z == 1`` for
  single-frame image-level scans (e.g. current Skin pilot).
* ``H``, ``W`` — spatial raster dimensions.

Phase is always returned in **radians** in ``[-pi, pi)`` (NumPy
convention) so that downstream operators do not need to know how the
underlying file encoded phase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Wrapped phase in radians, shape ``(n_z, H, W)``, float, range
    ``[-pi, pi)`` by convention."""

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


def load_leaf_mat(
    path: str | Path,
    amplitude_var: str = "L_A",
    phase_var: str = "L_P",
    spatial_slice: tuple[slice, slice] | None = None,
    z_slice: slice | None = None,
) -> LFIRawData:
    """Load a single leaf scan from the supplied MATLAB ``.mat`` file.

    The file is the non-HDF5 v5/v6/v7 MATLAB format; whole variables
    are materialised into memory by :mod:`scipy.io`. Use the ``slice``
    arguments to restrict the loaded volume for memory-bound pilots.

    Variable layout (see ``data/leaf/README.md``)::

        L_A    (H, W, n_sweep)  float64   amplitude
        L_P    (H, W, n_sweep)  float64   wrapped phase
        L_P_U  (H, W, n_unwr)   float64   group's ad-hoc unwrapped phase
        L_P_U_M (H, W, n_unwr)  float64   ditto, with validity mask

    The returned tensor is transposed from ``(H, W, n_sweep)`` to
    ``(n_sweep, H, W)`` to match the :class:`LFIRawData` contract.

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
        Optional ``slice`` along the sweep axis.

    Returns
    -------
    LFIRawData
        ``amplitude`` and ``phase`` shaped ``(n_z, H, W)``. ``z_indices``
        is ``[0, 1, ..., n_z - 1]`` (sweep-sample positions). The leaf
        ``.mat`` does not carry frame indices, so positions are used as
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

    if spatial_slice is not None:
        sh, sw = spatial_slice
        amp_hwz = amp_hwz[sh, sw, :]
        phs_hwz = phs_hwz[sh, sw, :]
    if z_slice is not None:
        amp_hwz = amp_hwz[:, :, z_slice]
        phs_hwz = phs_hwz[:, :, z_slice]

    amplitude = np.ascontiguousarray(np.transpose(amp_hwz, (2, 0, 1)))
    phase = np.ascontiguousarray(np.transpose(phs_hwz, (2, 0, 1)))

    n_z = amplitude.shape[0]
    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_format": "matlab_v5",
        "amplitude_var": amplitude_var,
        "phase_var": phase_var,
        "spatial_slice": repr(spatial_slice),
        "z_slice": repr(z_slice),
        "phase_unit": "radians (assumed)",
    }

    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=list(range(n_z)),
        target_type="Leaf",
        metadata=metadata,
    )


__all__ = ["LFIRawData", "load_skin_hdf5", "load_leaf_mat"]
