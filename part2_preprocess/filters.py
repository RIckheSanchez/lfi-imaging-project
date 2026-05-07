"""
Part 2: Signal Pre-processing
Responsible: Mingxuan Tang

This module handles filtering and noise reduction of LFI signals.
"""

from shared.data_types import LFIRawData, ProcessedSignal
import numpy as np


def preprocess_signal(raw_data: LFIRawData) -> ProcessedSignal:
    """
    Apply filtering and noise reduction to raw signal
    
    Args:
        raw_data: Raw LFI data from Part 1
        
    Returns:
        ProcessedSignal with cleaned signal
        
    TODO: Implement signal processing pipeline
    """
    raise NotImplementedError("TODO: Mingxuan to implement")


def apply_filter(signal: np.ndarray, 
                 filter_type: str = 'lowpass',
                 cutoff: float = 0.5) -> np.ndarray:
    """
    Apply specific filter to signal
    
    Args:
        signal: Input signal array
        filter_type: Type of filter ('lowpass', 'highpass', 'bandpass')
        cutoff: Cutoff frequency
        
    Returns:
        Filtered signal
        
    TODO: Implement various filter types
    """
    raise NotImplementedError("TODO: Mingxuan to implement")


def estimate_noise_level(signal: np.ndarray) -> float:
    """
    Estimate noise level in signal
    
    Args:
        signal: Input signal
        
    Returns:
        Estimated noise level
    """
    raise NotImplementedError("TODO: Mingxuan to implement")
