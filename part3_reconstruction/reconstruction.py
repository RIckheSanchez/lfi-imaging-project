"""Explicit adapters for the original team entry points."""
import numpy as np
from .data import ImagingData
from .volume_reconstructor import VolumeReconstructor
from .image_generator import ImageGenerator
from .validation import image_metrics


def reconstruct_image(processed_signal, *, phase_mode="cyclic"):
    """Return a dual-channel figure. Legacy clean_signal input is ambiguous."""
    if not isinstance(processed_signal, ImagingData):
        raise TypeError("Pass ImagingData(amplitude, phase, metadata) in yx order")
    return ImageGenerator(processed_signal).generate_comparison(phase_mode=phase_mode)


def reconstruct_3d(signals, z_levels, *, spacing_xy, spatial_unit="mm"):
    if not signals or not all(isinstance(s, ImagingData) and s.amplitude.ndim == 2 for s in signals):
        raise TypeError("signals must be a nonempty sequence of 2D ImagingData")
    if len({(s.metadata.get("phase_state"), s.metadata["phase_unit"]) for s in signals}) != 1:
        raise ValueError("slices must have consistent phase semantics")
    data = ImagingData(np.stack([s.amplitude for s in signals]), np.stack([s.phase for s in signals]), signals[0].metadata)
    return VolumeReconstructor(data, spacing_xy=spacing_xy, z_coordinates=z_levels, spatial_unit=spatial_unit)


def calculate_quality_score(image, reference=None, *, data_range=None):
    if reference is None or data_range is None:
        raise ValueError("quality requires a reference and explicit data_range")
    return image_metrics(reference, image, data_range=data_range)["ssim"]
