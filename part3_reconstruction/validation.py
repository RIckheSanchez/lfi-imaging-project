"""Reference-based metrics, never an unexplained standalone quality score."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(reference, candidate, *, data_range):
    a, b = np.asarray(reference), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 7:
        raise ValueError("metrics require matching 2D images at least 7x7")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(data_range) or data_range <= 0:
        raise ValueError("metrics require finite images and positive data_range")
    return {"psnr": float("inf") if np.array_equal(a, b) else float(peak_signal_noise_ratio(a, b, data_range=data_range)),
            "ssim": float(structural_similarity(a, b, data_range=data_range))}


def dice_score(reference_mask, candidate_mask):
    a, b = np.asarray(reference_mask, dtype=bool), np.asarray(candidate_mask, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("mask shapes must match")
    total = a.sum() + b.sum()
    return 1.0 if total == 0 else float(2 * np.count_nonzero(a & b) / total)
