"""Inactive integration example. Import/register explicitly to try it."""
from dataclasses import replace
import numpy as np
from part4_ui.lfi_core.data_types import LFIRawData


class GaussianPreprocessor:
    """Apply a separable Gaussian filter in XY; leave the Z axis unchanged."""
    def preprocess(self, data: LFIRawData, params: dict) -> LFIRawData:
        sigma = float(params.get('sigma', 1.0))
        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError('sigma must be a positive finite number.')
        radius = max(1, int(np.ceil(3 * sigma)))
        coordinates = np.arange(-radius, radius + 1)
        kernel = np.exp(-(coordinates**2) / (2*sigma**2))
        kernel /= kernel.sum()
        def smooth(values):
            result = values.astype(float, copy=True)
            for axis in (1, 2):
                def filter_line(line):
                    padded = np.pad(line, radius, mode='reflect')
                    return np.convolve(padded, kernel, mode='valid')
                result = np.apply_along_axis(filter_line, axis, result)
            return result
        return replace(data, amplitude=smooth(data.amplitude), phase=smooth(data.phase))
