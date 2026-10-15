"""Reconstruct aligned, depth-resolved slices in physical coordinates.

This does not infer depth from phase, acquisition frame number or frequency.
"""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import shift as spatial_shift
from .data import ImagingData


class VolumeReconstructor:
    def __init__(self, data, *, spacing_xy, z_coordinates, origin_xy=(0., 0.), spatial_unit="mm"):
        if not isinstance(data, ImagingData) or data.amplitude.ndim != 3:
            raise ValueError("data must be ImagingData with shape (z,y,x)")
        self.data = data
        self.dx, self.dy = map(float, spacing_xy)
        self.ox, self.oy = map(float, origin_xy)
        if not np.isfinite([self.dx, self.dy, self.ox, self.oy]).all() or min(self.dx, self.dy) <= 0:
            raise ValueError("spacing must be positive and origin finite")
        z = np.asarray(z_coordinates, dtype=float)
        if z.ndim != 1 or len(z) != data.amplitude.shape[0] or not np.isfinite(z).all():
            raise ValueError("one finite depth coordinate is required per slice")
        order = np.argsort(z)
        self.z = z[order]
        if np.any(np.diff(self.z) <= 0):
            raise ValueError("depth coordinates must be unique")
        if np.array_equal(order, np.arange(len(z))):      # already sorted: no copy
            self.data = ImagingData._wrap(data.amplitude, data.phase, data.metadata)
        else:
            self.data = ImagingData._wrap(data.amplitude[order], data.phase[order], data.metadata)
        self.spatial_unit = spatial_unit
        self.data.metadata.update(spacing_xy=(self.dx, self.dy), origin_xy=(self.ox, self.oy),
                                  spatial_unit=spatial_unit, z_coordinates=self.z.tolist())
        self.registration_shifts = None

    def register_translation(self, reference=0, upsample_factor=10):
        """Opt-in XY translation registration using amplitude; apply to both channels.

        Assumes similar structures across slices. Never run automatically on
        different tissue depths. Missing borders become NaN, never invented data.
        Returns a new volume; the input is unchanged.
        """
        from skimage.registration import phase_cross_correlation
        if not 0 <= reference < len(self.z):
            raise ValueError("reference index out of range")
        if not np.isfinite(self.data.amplitude).all() or not np.isfinite(self.data.phase).all():
            raise ValueError("registration requires finite input")
        a, p = self.data.amplitude.copy(), self.data.phase.copy()
        if self.data.metadata.get("phase_state") != "unwrapped" or self.data.metadata["phase_unit"] != "rad":
            raise ValueError("registration interpolation requires explicitly unwrapped phase")
        shifts = []
        for k in range(len(self.z)):
            delta, _, _ = phase_cross_correlation(a[reference], self.data.amplitude[k], upsample_factor=upsample_factor)
            shifts.append(delta.tolist())
            a[k] = spatial_shift(self.data.amplitude[k], delta, order=1, mode="constant", cval=np.nan, prefilter=False)
            p[k] = spatial_shift(self.data.phase[k], delta, order=1, mode="constant", cval=np.nan, prefilter=False)
        result = VolumeReconstructor(ImagingData._wrap(a, p, self.data.metadata), spacing_xy=(self.dx, self.dy), z_coordinates=self.z, origin_xy=(self.ox, self.oy), spatial_unit=self.spatial_unit)
        result.registration_shifts = shifts
        result.data.metadata["registration_shifts_yx"] = shifts
        result.data.metadata["registration_reference_index"] = reference
        return result

    def resample_isotropic(self, spacing=None, max_voxels=50_000_000):
        """Trilinear interpolation on physical coordinates, including irregular z.

        Output centres remain inside the source extent; the last centre may be
        less than one output voxel short of its upper edge. Trilinear
        interpolation is separable, so each output plane is the linear blend in
        z of the two bracketing source slices, each interpolated bilinearly in
        (y, x). Only the output and two interpolated slices are held in memory.
        Brackets follow RegularGridInterpolator, so NaN source regions remain
        missing, including interpolation neighbours, exactly as in 3D.
        """
        if self.data.metadata.get("phase_state") != "unwrapped" or self.data.metadata["phase_unit"] != "rad":
            raise ValueError("declare phase_state='unwrapped' before interpolating phase")
        if len(self.z) < 2:
            raise ValueError("3D resampling requires at least two depth slices")
        spacing = float(min(self.dx, self.dy, np.diff(self.z).min()) if spacing is None else spacing)
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("output spacing must be finite and positive")
        nz, ny, nx = self.data.amplitude.shape
        source = (self.z, self.oy + np.arange(ny)*self.dy, self.ox + np.arange(nx)*self.dx)
        counts = [int(np.floor((v[-1]-v[0])/spacing + 1e-9))+1 for v in source]
        if np.prod(counts, dtype=object) > max_voxels:
            raise ValueError("requested volume exceeds max_voxels; increase spacing or budget")
        coords = [v[0] + np.arange(n)*spacing for v, n in zip(source, counts)]
        points = np.stack(np.meshgrid(coords[1], coords[2], indexing="ij"), axis=-1)
        # Same bracket choice as RegularGridInterpolator: interval [i, i+1] with
        # i = searchsorted(z, zo) - 1, clipped to the valid range.
        upper = np.clip(np.searchsorted(self.z, coords[0], side="left"), 1, nz - 1)
        lower = upper - 1
        weight = (coords[0] - self.z[lower]) / (self.z[upper] - self.z[lower])
        outputs = []
        for values in (self.data.amplitude, self.data.phase):
            planes = {}
            out = np.empty(counts, dtype=np.float32)
            for k in range(counts[0]):
                for i in (lower[k], upper[k]):
                    if i not in planes:
                        planes[i] = RegularGridInterpolator(source[1:], values[i], method="linear",
                                                            bounds_error=False, fill_value=np.nan)(points)
                for i in [i for i in planes if i < lower[k]]:
                    del planes[i]
                out[k] = planes[lower[k]] * (1 - weight[k]) + planes[upper[k]] * weight[k]
            outputs.append(out)
        meta = dict(self.data.metadata, resampling="trilinear physical-coordinate interpolation", output_spacing=spacing)
        return VolumeReconstructor(ImagingData._wrap(*outputs, meta), spacing_xy=(spacing, spacing), z_coordinates=coords[0], origin_xy=(self.ox, self.oy), spatial_unit=self.spatial_unit)

    def get_orthoslice(self, axis, index, channel="amplitude"):
        axes = {"xy": 0, "xz": 1, "yz": 2}
        if axis not in axes or channel not in {"amplitude", "phase"}:
            raise ValueError("axis must be xy/xz/yz; channel amplitude/phase")
        arr = getattr(self.data, channel)
        if not isinstance(index, (int, np.integer)) or not 0 <= index < arr.shape[axes[axis]]:
            raise ValueError("slice index must be a nonnegative integer within the selected axis")
        return np.take(arr, index, axis=axes[axis]).copy()

    def export_nifti(self, path, channel="amplitude"):
        """Export xyz array with axis-aligned scanner coordinates; no anatomical claim."""
        if channel not in {"amplitude", "phase"}:
            raise ValueError("channel must be amplitude or phase")
        if channel == "phase" and self.data.metadata["phase_unit"] != "rad":
            raise ValueError("phase export requires calibrated radians")
        if len(self.z) < 2 or not np.allclose(np.diff(self.z), np.diff(self.z)[0], rtol=1e-5, atol=1e-8):
            raise ValueError("NIfTI requires uniform z spacing and at least two slices; resample first")
        return write_nifti(path, getattr(self.data, channel),
                           spacing_xyz=(self.dx, self.dy, float(self.z[1]-self.z[0])),
                           origin_xyz=(self.ox, self.oy, float(self.z[0])), spatial_unit=self.spatial_unit,
                           description=f"LFI {channel}; axis-aligned scanner grid; phase radians",
                           sidecar={"channel": channel, "metadata": self.data.metadata})

    def view_napari(self):
        """Optional viewer; caller owns event loop (napari.run())."""
        if len(self.z) < 2 or not np.allclose(np.diff(self.z), np.diff(self.z)[0]):
            raise ValueError("viewer requires a uniform z grid; resample first")
        try:
            import napari
        except ImportError as exc:
            raise ImportError('Optional viewer requires: pip install "napari[all]"') from exc
        viewer = napari.Viewer(ndisplay=3)
        scale = (float(self.z[1]-self.z[0]), self.dy, self.dx)
        translate = (float(self.z[0]), self.oy, self.ox)
        for name in ("amplitude", "phase"):
            viewer.add_image(getattr(self.data, name), name=name, scale=scale, translate=translate, colormap="inferno" if name == "amplitude" else "viridis", visible=name == "amplitude")
        return viewer


def write_nifti(path, array_zyx, *, spacing_xyz, origin_xyz, spatial_unit, description, sidecar):
    """Write a (z, y, x) array as an xyz NIfTI with qform = sform and a JSON sidecar."""
    import nibabel as nib
    if spatial_unit not in {"mm", "micron", "meter"}:
        raise ValueError("NIfTI needs an explicit supported physical unit")
    path = Path(path)
    if not (str(path).endswith(".nii") or str(path).endswith(".nii.gz")):
        raise ValueError("NIfTI path must end in .nii or .nii.gz")
    affine = np.diag([*map(float, spacing_xyz), 1.])
    affine[:3, 3] = origin_xyz
    image = nib.Nifti1Image(np.asarray(array_zyx).transpose(2, 1, 0), affine)
    image.header.set_xyzt_units(spatial_unit)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    image.header["descrip"] = description[:79]
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)
    path.with_name(path.name + ".json").write_text(json.dumps({
        **sidecar, "array_order": "xyz", "spatial_unit": spatial_unit, "affine": affine.tolist(),
        "orientation": "axis-aligned scanner coordinates; not anatomical registration"
    }, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")
    return path
