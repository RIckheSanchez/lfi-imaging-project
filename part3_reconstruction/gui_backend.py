"""Part 3 backend for the desktop viewer (part4_ui). No Qt import here.

``DepthReconstructor`` implements the viewer's Reconstructor protocol,
``reconstruct(data, params) -> LFIRawData``. Each slice is placed at the depth
of the beam focus and the amplitude is resampled linearly onto a uniform depth
grid, so the viewer's index-based 3D view has equal spacing in z even when the
acquisition steps were irregular. Wrapped phase cannot be interpolated, so each
output slice takes the phase of the nearest measured slice.

Depth comes from explicit parameters (z_start_mm, z_step_mm, first_frame_id)
when given, otherwise from ``metadata['depth_mm']`` set by Part 1.
"""
import numpy as np
from .handoff import resample_depth

# 2026 Vegetables scan: frame 20 is the first frame, focus at 30.05 mm, each
# later frame 0.1 mm deeper (confirmed by the acquisition team, October 2026).
VEGETABLES_2026 = {"z_start_mm": 30.05, "z_step_mm": 0.1, "first_frame_id": 20}
DEFAULT_PARAMS = {"output_spacing_mm": 0.1, "max_slices": 400}


def frame_depths(data, params):
    """Return (depth of each slice in mm, description of the source)."""
    ids = np.asarray(data.z_indices, dtype=float)
    keys = ("z_start_mm", "z_step_mm", "first_frame_id")
    given = [params.get(k) is not None for k in keys]
    if any(given):
        if not all(given):
            raise ValueError("give z_start_mm, z_step_mm and first_frame_id together")
        depth = float(params["z_start_mm"]) + (ids - float(params["first_frame_id"])) * float(params["z_step_mm"])
        return depth, "parameters z_start_mm/z_step_mm/first_frame_id"
    depth = data.metadata.get("depth_mm")
    if depth is not None and len(depth) == len(ids):
        depth = np.asarray(depth, dtype=float)
        if np.isfinite(depth).all():
            return depth, "metadata depth_mm"
    raise ValueError("slice depths unknown: pass z_start_mm, z_step_mm and first_frame_id, "
                     "or load data whose metadata has depth_mm")


class DepthReconstructor:
    """Resample a slice stack onto a uniform beam-focus depth grid for the 3D view."""

    def reconstruct(self, data, params):
        p = {**DEFAULT_PARAMS, **(params or {})}
        if data.amplitude.ndim != 3 or data.amplitude.shape[0] < 2:
            raise ValueError("depth reconstruction needs at least two slices")
        depth, source = frame_depths(data, p)
        if len(np.unique(depth)) != len(depth):
            raise ValueError("two slices have the same depth")
        spacing = float(p["output_spacing_mm"])
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("output_spacing_mm must be positive")
        count = int(np.floor((depth.max() - depth.min()) / spacing + 1e-9)) + 1
        if count > int(p["max_slices"]):
            raise ValueError(f"{count} output slices exceed max_slices={p['max_slices']}; "
                             "increase output_spacing_mm")
        amplitude, grid = resample_depth(data.amplitude, depth, spacing)
        order = np.argsort(depth)
        nearest = order[np.abs(grid[:, None] - depth[order][None, :]).argmin(axis=1)]
        phase = np.asarray(data.phase, dtype=np.float32)[nearest]
        metadata = dict(data.metadata)
        metadata.update(
            depth_mm=[float(z) for z in grid],
            part3_reconstruction={
                "method": "linear interpolation of amplitude along depth; phase from nearest slice",
                "depth_source": source, "output_spacing_mm": spacing,
                "source_slice_ids": [data.z_indices[i] for i in order],
                "source_depth_mm": [float(depth[i]) for i in order],
                "phase_source_slice_ids": [data.z_indices[i] for i in nearest]})
        return type(data)(amplitude=amplitude, phase=phase,
                          z_indices=[round(float(z), 4) for z in grid],
                          target_type=data.target_type, file_path=data.file_path, metadata=metadata)
