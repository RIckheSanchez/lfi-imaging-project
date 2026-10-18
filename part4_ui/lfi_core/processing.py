"""Pure array operations shared by desktop and legacy frontends."""
import numpy as np

def finite_values(values):
    """Return finite float64 samples; reject channels with no measured samples."""
    array = np.asarray(values, dtype=np.float64)
    samples = array[np.isfinite(array)]
    if not samples.size:
        raise ValueError('Channel has no finite values (all NaN or non-finite).')
    return samples


def finite_range(values):
    samples = finite_values(values)
    return float(samples.min()), float(samples.max())


def normalized_slice(data, z_index, channel='amplitude'):
    """Return normalized [0,1] slice for the given channel ('amplitude' or 'phase')."""
    slice_data = data.get_slice(z_index)
    arr = np.asarray(slice_data[channel], dtype=np.float64)
    s_min, s_max = finite_range(arr)
    normalized = (arr - s_min) / (s_max - s_min + 1e-10)
    return normalized, s_min, s_max

def downsampled_volume(data, max_points=200_000):
    """Cache and downsample 3D amplitude volume for fast rendering."""
    amp   = np.asarray(data.amplitude, dtype=np.float64)
    total = amp.size
    step  = max(1, int((total / max_points) ** (1 / 3)))
    sampled = amp[::step, ::step, ::step]
    Z, Y, X = np.mgrid[
        0:amp.shape[0]:step,
        0:amp.shape[1]:step,
        0:amp.shape[2]:step
    ]
    return X, Y, Z, sampled, step


def adjust_slice(normalized, brightness=60, contrast=75):
    """Apply the legacy normalized brightness/contrast transform."""
    normalized = np.asarray(normalized, dtype=np.float64)
    return np.clip((normalized - .5) * (contrast / 50) + .5 + (brightness - 50) / 100, 0, 1)


def isosurface_thresholds(values, percentile=90):
    """Lower requested percentile (default 90) and fixed upper 99.5th."""
    if not 0 <= percentile < 99.5:
        raise ValueError('Isomin percentile must be at least 0 and below 99.5.')
    samples = finite_values(values)
    return float(np.percentile(samples, percentile)), float(np.percentile(samples, 99.5))



def prepare_3d(data, mode='volume', percentile=90, colormap='Viridis',
               brightness=60, contrast=75, max_points=200_000):
    """Prepare XYZ volume pixels or three uncapped meshes off the GUI thread.

    Sampling matches the legacy frontend. Inputs/mesh coordinates use sensor
    pixels, not a claimed physical Z calibration. marching_cubes is imported
    only for isosurfaces; neither Qt nor OpenGL is needed to prepare results.
    """
    from .export import get_export_colormap
    _, _, _, values, step = downsampled_volume(data, max_points)
    if min(values.shape) < 2:
        raise ValueError('3D rendering requires at least two samples along every axis.')
    finite = np.isfinite(values)
    samples = finite_values(values)
    if mode not in ('volume', 'isosurface'):
        raise ValueError('Unknown 3D rendering mode.')
    result = dict(mode=mode, original_shape=data.amplitude.shape, step=step,
                  sampled_points=values.size, values_sum=float(samples.sum()))
    minimum, maximum = finite_range(values)
    cmap = get_export_colormap(colormap)
    if mode == 'volume':
        lower, upper = map(float, np.percentile(samples, [1, 99.5]))
        result['normalization_range'] = (lower, upper)
        normalized = (np.clip(values, lower, upper) - lower) / (upper - lower + 1e-10)
        adjusted = adjust_slice(normalized, brightness, contrast)
        rgba = cmap(adjusted)
        rgba[..., 3] = np.where(finite, adjusted**2 * .35, 0.0)
        # GLVolumeItem expects (X, Y, Z, RGBA); HDF5 is (Z, Y, X).
        result['rgba'] = np.ascontiguousarray((rgba.transpose(2, 1, 0, 3) * 255).astype(np.uint8))
    else:
        from skimage.measure import marching_cubes
        low, high = isosurface_thresholds(values, percentile)
        if not minimum < low < high <= maximum:
            raise ValueError('The selected percentiles do not define distinct isosurfaces. Try another percentile.')
        # Missing samples are excluded from every marching-cubes stencil.
        # A finite implementation buffer is required by skimage, but the mask
        # prevents these placeholders from becoming surfaces or measurements.
        from scipy.ndimage import binary_erosion
        mesh_mask = binary_erosion(finite, structure=np.ones((3,3,3)), border_value=1)
        if not mesh_mask.any():
            raise ValueError('No finite region is large enough for an isosurface.')
        mesh_values = np.where(finite, values, minimum)
        meshes = []
        for level, opacity in zip(np.linspace(low, high, 3), (.12, .30, .60)):
            level = min(float(level), float(np.nextafter(np.float32(maximum), np.float32(minimum))))
            vertices, faces, _, _ = marching_cubes(mesh_values, level=level, spacing=(step, step, step), mask=mesh_mask)
            # No boundary caps are added. Reverse ZYX vertices to sensor XYZ.
            color_value = adjust_slice((level-minimum)/(maximum-minimum+1e-10), brightness, contrast)
            color = tuple(cmap(color_value)[:3]) + (opacity,)
            meshes.append((np.ascontiguousarray(vertices[:, ::-1]), faces, color))
        result.update(meshes=meshes, thresholds=(low, high))
    return result
