"""
data_types.py
Standard output type for LFI data
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class LFIRawData:
    """
    Standard LFI scan output (single file or multi-file volume)

    Attributes:
        amplitude:    amplitude array, shape (n_slices, height, width)
        phase:        phase array, shape (n_slices, height, width)
        z_indices:    z / frame id list, e.g. [21, 22, ..., 100]
        target_type:  target type, e.g. 'Chip', 'Leaf', 'Skin'
        file_path:    source file path; for volumes, the data directory
        metadata:     extra info (depth_mm, z_start_mm, z_step_mm, ...)
        waveform:     optional fitted waveform (n, H, W, n_samples); off by default
    """
    amplitude:   np.ndarray
    phase:       np.ndarray
    z_indices:   list
    target_type: str
    file_path:   str
    metadata:    dict = field(default_factory=dict)
    waveform:    Optional[np.ndarray] = None

    def __post_init__(self):
        # basic checks
        if self.amplitude.shape != self.phase.shape:
            raise ValueError(
                f"amplitude and phase shapes differ: "
                f"{self.amplitude.shape} vs {self.phase.shape}"
            )
        if len(self.z_indices) != self.amplitude.shape[0]:
            raise ValueError(
                f"z_indices count ({len(self.z_indices)}) "
                f"vs number of slices ({self.amplitude.shape[0]}) mismatch"
            )
        if self.waveform is not None and self.waveform.shape[:3] != self.amplitude.shape:
            raise ValueError(
                f"waveform first three dims {self.waveform.shape[:3]} "
                f"vs amplitude {self.amplitude.shape} mismatch"
            )

    @property
    def n_slices(self) -> int:
        """number of z slices"""
        return self.amplitude.shape[0]

    @property
    def height(self) -> int:
        return self.amplitude.shape[1]

    @property
    def width(self) -> int:
        return self.amplitude.shape[2]

    def get_slice(self, z_index: int) -> dict:
        """
        Get amplitude and phase for one slice

        Args:
            z_index: z index, e.g. 530

        Returns:
            {'amplitude': 2D array, 'phase': 2D array}
        """
        if z_index not in self.z_indices:
            raise KeyError(f"z_index {z_index} not found; available: {self.z_indices}")
        i = self.z_indices.index(z_index)
        return {
            'amplitude': self.amplitude[i],
            'phase':     self.phase[i],
        }

    def summary(self) -> str:
        """Print a short summary"""
        return (
            f"LFIRawData Summary\n"
            f"  target_type : {self.target_type}\n"
            f"  file_path   : {self.file_path}\n"
            f"  amplitude   : shape={self.amplitude.shape}, dtype={self.amplitude.dtype}\n"
            f"  phase       : shape={self.phase.shape},     dtype={self.phase.dtype}\n"
            f"  z_indices   : {self.z_indices[0]} → {self.z_indices[-1]} "
            f"({self.n_slices} slices)\n"
            f"  waveform    : "
            f"{'None' if self.waveform is None else self.waveform.shape}\n"
            f"  metadata    : {self.metadata}"
        )
