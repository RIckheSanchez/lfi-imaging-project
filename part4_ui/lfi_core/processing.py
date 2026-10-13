"""Pure array operations shared by desktop and legacy frontends."""
import numpy as np

def normalized_slice(data, z_index, channel='amplitude'):
    """Return normalized [0,1] slice for the given channel ('amplitude' or 'phase')."""
    slice_data = data.get_slice(z_index)
    arr = slice_data[channel]
    s_min, s_max = arr.min(), arr.max()
    normalized = (arr - s_min) / (s_max - s_min + 1e-10)
    return normalized, s_min, s_max

def downsampled_volume(data, max_points=200_000):
    """Cache and downsample 3D amplitude volume for fast rendering."""
    amp   = data.amplitude
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
    return np.clip((normalized - .5) * (contrast / 50) + .5 + (brightness - 50) / 100, 0, 1)


def isosurface_thresholds(values, percentile=60):
    """Lower requested percentile and fixed upper 95th percentile."""
    return float(np.percentile(values, percentile)), float(np.percentile(values, 95))
