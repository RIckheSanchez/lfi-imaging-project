"""
Shared data types for LFI Imaging Project

This module defines the data structures passed between different parts.
All team members should use these types for consistency.
"""

from dataclasses import dataclass, fields
from typing import Optional, Dict, List
import numpy as np


@dataclass
class LFIRawData:
    """
    Raw data from Part 1 (Zeyu Wang)
    Output of data loading module
    
    Data format from Zeyu's loader:
    - amplitude: shape (70, 1400, 400) - amplitude data at different z-depths
    - phase: shape (70, 1400, 400) - phase data at different z-depths
    - 70 z-slices, 1400x400 XY scan resolution
    """
    # Core LFI signal data
    amplitude: np.ndarray          # Amplitude signal, shape: (z_slices, x, y)
    phase: np.ndarray              # Phase signal, shape: (z_slices, x, y)
    
    # Coordinate information
    x_coords: Optional[np.ndarray] = None    # X coordinates
    y_coords: Optional[np.ndarray] = None    # Y coordinates
    z_coords: Optional[np.ndarray] = None    # Z-depth coordinates
    
    # Metadata
    target_type: Optional[str] = None        # 'Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom'
    scan_params: Optional[Dict] = None       # Scanning parameters
    
    # Optional fields
    timestamp: Optional[str] = None
    file_path: Optional[str] = None
    
    def __repr__(self):
        return (f"LFIRawData(amplitude_shape={self.amplitude.shape}, "
                f"phase_shape={self.phase.shape}, "
                f"target={self.target_type})")
    
    @property
    def num_z_slices(self) -> int:
        """Number of z-depth slices"""
        return self.amplitude.shape[0]
    
    @property
    def xy_resolution(self) -> tuple:
        """XY scan resolution (width, height)"""
        return (self.amplitude.shape[1], self.amplitude.shape[2])


@dataclass
class ProcessedSignal:
    """
    Processed signal from Part 2 (Haobo Yu)
    Output of signal pre-processing module
    """
    # Cleaned signal
    clean_signal: np.ndarray       # Filtered and denoised signal
    
    # Processing information
    noise_level: float             # Estimated noise level removed
    filter_params: Dict            # Parameters used in filtering
    
    # Original data reference
    original_shape: tuple          # Shape of original signal
    target_type: str               # Preserved from raw data
    
    def __repr__(self):
        return (f"ProcessedSignal(signal_shape={self.clean_signal.shape}, "
                f"noise_level={self.noise_level:.4f})")


@dataclass
class ReconstructedImage:
    """
    Reconstructed image from Part 3 (Mingxuan Tang)
    Output of image reconstruction module
    """
    # Image data
    image_2d: np.ndarray           # 2D reconstructed image
    slices_3d: Optional[List[np.ndarray]] = None  # 3D slices if available
    
    # Quality metrics
    quality_score: float = 0.0     # Reconstruction quality metric
    resolution: tuple = None       # (x_res, y_res) in appropriate units
    
    # Processing information
    reconstruction_method: str = "default"
    processing_time: float = 0.0   # Time taken in seconds
    
    # Metadata
    target_type: str = ""
    z_levels: Optional[List[float]] = None  # Z-depth levels if 3D
    
    def __repr__(self):
        slices_info = f", {len(self.slices_3d)} slices" if self.slices_3d else ""
        return (f"ReconstructedImage(shape={self.image_2d.shape}"
                f"{slices_info}, quality={self.quality_score:.2f})")


# Utility functions for data conversion
def raw_to_dict(data: LFIRawData) -> Dict:
    """Convert LFIRawData to dictionary (for serialization); inverse of dict_to_raw"""
    return {f.name: getattr(data, f.name) for f in fields(data)}


def dict_to_raw(d: Dict) -> LFIRawData:
    """Convert dictionary to LFIRawData (for deserialization)"""
    return LFIRawData(**d)
