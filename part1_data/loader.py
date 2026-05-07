"""
Part 1: Data Import & Management
Responsible: Zeyu Wang

This module handles loading and parsing LFI data from HDF5 and CSV files.
"""

from typing import List, Optional
from shared.data_types import LFIRawData
import numpy as np


def load_single_scan(filepath: str) -> LFIRawData:
    """
    Load a single LFI scan from file
    
    Args:
        filepath: Path to HDF5 or CSV file
        
    Returns:
        LFIRawData object with signal and metadata
        
    Raises:
        FileNotFoundError: If file doesn't exist
        DataFormatError: If file format is invalid
    
    TODO: Implement after exploring dataset structure
    """
    # Placeholder implementation
    raise NotImplementedError("TODO: Zeyu to implement after data exploration")


def load_multiple_scans(directory: str, 
                        target_type: Optional[str] = None) -> List[LFIRawData]:
    """
    Load multiple scans from a directory
    
    Args:
        directory: Path to directory containing scan files
        target_type: Optional filter by target type
        
    Returns:
        List of LFIRawData objects
    
    TODO: Implement batch loading
    """
    raise NotImplementedError("TODO: Zeyu to implement")


def get_available_targets(directory: str) -> List[str]:
    """
    Get list of available target types in directory
    
    Args:
        directory: Path to data directory
        
    Returns:
        List of target type names (e.g., ['Chip', 'Leaf', 'Skin'])
    """
    raise NotImplementedError("TODO: Zeyu to implement")
