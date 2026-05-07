"""
Shared data types for LFI Imaging Project

This module defines the data structures passed between different parts.
All team members should use these types for consistency.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import numpy as np


@dataclass
class LFIRawData:
    """
    Raw data from Part 1 (Zeyu)
    Output of data loading module
    """
    # Core signal data
    signal: np.ndarray              # Raw LFI signal, shape: (X, Y) or (X, Y, Z)
    
    # Coordinate information
    x_coords: np.ndarray           # X coordinates
    y_coords: np.ndarray           # Y coordinates
    z_depth: float                 # Current z-depth value
    
    # Metadata
    target_type: str               # 'Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom'
    scan_params: Dict              # Scanning parameters
    
    # Optional fields
    timestamp: Optional[str] = None
    file_path: Optional[str] = None
    
    def __repr__(self):
        return (f"LFIRawData(signal_shape={self.signal.shape}, "
                f"target={self.target_type}, z_depth={self.z_depth})")


@dataclass
class ProcessedSignal:
    """
    Processed signal from Part 2 (Mingxuan)
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
    Reconstructed image from Part 3 (Haobo)
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
    """Convert LFIRawData to dictionary (for serialization)"""
    return {
        'signal': data.signal,
        'x_coords': data.x_coords,
        'y_coords': data.y_coords,
        'z_depth': data.z_depth,
        'target_type': data.target_type,
        'scan_params': data.scan_params,
        'timestamp': data.timestamp,
        'file_path': data.file_path
    }


def dict_to_raw(d: Dict) -> LFIRawData:
    """Convert dictionary to LFIRawData (for deserialization)"""
    return LFIRawData(**d)
