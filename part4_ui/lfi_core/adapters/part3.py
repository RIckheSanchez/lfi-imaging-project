"""UI boundary for Mingxuan's unchanged depth backend; no calibration presets."""
import importlib
import logging
import numpy as np
from ..data_types import LFIRawData

CALIBRATION_KEYS = ('z_start_mm', 'z_step_mm', 'first_frame_id')


def calibrated_depths(data, params=None):
    """Validate relative focus positions or a complete user-supplied frame mapping.

    Manual mappings override metadata only when all three fields are supplied.
    An incomplete/invalid mapping must never silently fall back to metadata.
    """
    params = params or {}
    ids = np.asarray(data.z_indices, dtype=float)
    if data.n_slices < 2 or not np.isfinite(ids).all() or len(np.unique(ids)) != len(ids):
        raise ValueError('Reconstruction requires at least two distinct finite frame IDs.')
    given = [params.get(k) is not None for k in CALIBRATION_KEYS]
    if any(given):
        if not all(given):
            raise ValueError('Enter z_start_mm, z_step_mm and first_frame_id together.')
        start, step, first = (float(params[k]) for k in CALIBRATION_KEYS)
        if not np.isfinite([start, step, first]).all() or start <= 0 or step <= 0:
            raise ValueError('Start focus position and focus step must be finite positive numbers.')
        if not first.is_integer():
            raise ValueError('first_frame_id must be a finite integer reference frame ID.')
        depth = start + (ids - first) * step
        if np.any(depth <= 0):
            raise ValueError('The mapping must give every loaded frame a positive relative focus position.')
        spacing = step
    else:
        depth = np.asarray(data.metadata.get('depth_mm', []), dtype=float)
        if depth.shape != ids.shape or not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError('One finite nonnegative metadata depth_mm is required per slice.')
        spacing = None
    ordered = np.sort(depth)
    if not np.isfinite(depth).all() or np.any(np.diff(ordered) <= 0):
        raise ValueError('Slice depths must be finite and unique.')
    # Derive the output grid from real input calibration, never backend defaults.
    spacing = float(np.min(np.diff(ordered))) if spacing is None else spacing
    return depth, spacing


class Part3Reconstructor:
    def __init__(self):
        module = importlib.import_module('part3_reconstruction.gui_backend')
        self.backend = module.DepthReconstructor()

    def reconstruct(self, data: LFIRawData, params: dict) -> LFIRawData:
        if not isinstance(data, LFIRawData):
            raise TypeError('Part 3 adapter requires LFIRawData in (Z, Y, X) order.')
        _, spacing = calibrated_depths(data, params)
        supplied = dict(params or {})
        supplied['output_spacing_mm'] = spacing
        output = self.backend.reconstruct(data, supplied)
        if not isinstance(output, LFIRawData):
            raise TypeError('Part 3 must return LFIRawData.')
        if len(np.unique(output.z_indices)) != output.n_slices:
            raise ValueError('Depth grid is finer than the backend frame-label precision.')
        output.metadata['ui_part3'] = {
            'phase_policy': 'nearest_measured_slice',
            'amplitude_policy': 'linear_depth_interpolation',
            'axes': 'sample_indices', 'z_grid': 'uniform_depth',
            'output_spacing_mm': spacing,
        }
        return output


def register_part3():
    """Optional startup registration; failure leaves the v1.3 identity behavior."""
    from ..backend import register_reconstructor
    try:
        register_reconstructor(Part3Reconstructor())
        return True
    except Exception:
        logging.getLogger(__name__).exception('Part 3 unavailable; retaining v1.3 reconstruction behavior')
        register_reconstructor(None)
        return False
