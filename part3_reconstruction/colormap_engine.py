"""Display transforms only: never change the scientific input arrays."""
import numpy as np
from matplotlib import colormaps


def normalise(arr, strategy="percentile", pct_lo=2.0, pct_hi=98.0, limits=None):
    """Return float32 [0,1]; preserve missing pixels as NaN.

    Log uses log1p(value - lower_bound), so baseline-corrected negative
    amplitudes are supported. Explicit limits allow consistent comparisons.
    """
    a = np.asarray(arr, dtype=np.float64)
    if strategy not in {"linear", "percentile", "log"}:
        raise ValueError("strategy must be linear, percentile or log")
    if not 0 <= pct_lo < pct_hi <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    valid = np.isfinite(a)
    if not valid.any():
        raise ValueError("array contains no finite values")
    if limits is None:
        lo, hi = (np.percentile(a[valid], [pct_lo, pct_hi])
                  if strategy == "percentile" else (a[valid].min(), a[valid].max()))
    else:
        lo, hi = limits
    if not np.isfinite([lo, hi]).all() or hi < lo:
        raise ValueError("limits must be finite and increasing")
    out = np.full(a.shape, np.nan, dtype=np.float32)
    if hi == lo:
        out[valid] = 0
    else:
        values = np.clip(a[valid], lo, hi) - lo
        out[valid] = (np.log1p(values) / np.log1p(hi - lo)
                      if strategy == "log" else values / (hi - lo))
    return out


def fold_phase(phase):
    """Radians -> [-pi, pi), with nonfinite inputs masked."""
    phase = np.asarray(phase, dtype=np.float64)
    out = np.full(phase.shape, np.nan)
    valid = np.isfinite(phase)
    out[valid] = (phase[valid] + np.pi) % (2 * np.pi) - np.pi
    return out


def resolve_cmap(name):
    return colormaps["twilight" if name == "phase" else name].copy()


class ColormapEngine:
    normalise = staticmethod(normalise)
    fold_phase = staticmethod(fold_phase)
    resolve_cmap = staticmethod(resolve_cmap)
