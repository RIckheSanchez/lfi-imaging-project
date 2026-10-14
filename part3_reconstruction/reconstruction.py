"""
Part 3: Image Generation & Reconstruction
Responsible: Mingxuan Tang

This module handles image reconstruction from processed LFI signals.
"""

from typing import List
from shared.data_types import ProcessedSignal, ReconstructedImage
import numpy as np


def reconstruct_image(processed_signal: ProcessedSignal) -> ReconstructedImage:
    """
    Reconstruct 2D image from processed signal
    
    Args:
        processed_signal: Cleaned signal from Part 2
        
    Returns:
        ReconstructedImage with 2D image data
        
    TODO: Implement reconstruction algorithm
    """
    raise NotImplementedError("TODO: Mingxuan to implement")


def reconstruct_3d(signals: List[ProcessedSignal], 
                   z_levels: List[float]) -> ReconstructedImage:
    """
    Reconstruct 3D volume from multiple z-depth scans
    
    Args:
        signals: List of processed signals at different depths
        z_levels: Corresponding z-depth values
        
    Returns:
        ReconstructedImage with 3D data
        
    TODO: Implement 3D tomographic reconstruction
    """
    raise NotImplementedError("TODO: Mingxuan to implement")


def calculate_quality_score(image: np.ndarray) -> float:
    """
    Calculate reconstruction quality metric
    
    Args:
        image: Reconstructed image
        
    Returns:
        Quality score (0-1)
    """
    raise NotImplementedError("TODO: Mingxuan to implement")
