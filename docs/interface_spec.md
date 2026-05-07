# Module Interface Specification

## Overview

This document defines the interfaces between different modules in the LFI Imaging Project.
All team members must follow these specifications to ensure smooth integration.

---

## Part 1: Data Import & Management (Zeyu Wang)

### Module: `part1_data`

### Main Functions

#### `load_single_scan(filepath: str) -> LFIRawData`

**Purpose:** Load a single LFI scan from HDF5/CSV file

**Parameters:**
- `filepath` (str): Path to the data file

**Returns:**
- `LFIRawData`: Object containing signal data, coordinates, and metadata

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `DataFormatError`: If file format is invalid

**Example:**
```python
from part1_data.loader import load_single_scan

data = load_single_scan('datasets/Chip_001.h5')
print(data.amplitude.shape)  # (70, 1400, 400)
print(data.phase.shape)      # (70, 1400, 400)
print(data.num_z_slices)     # 70
print(data.xy_resolution)    # (1400, 400)
print(data.target_type)      # 'Chip'
```

**Data Format:**
- `amplitude`: numpy array with shape (70, 1400, 400)
  - 70 z-depth slices
  - 1400 x 400 XY scan resolution
- `phase`: numpy array with shape (70, 1400, 400)
  - Same dimensions as amplitude
  - Phase information from LFI

#### `load_multiple_scans(directory: str, target_type: str = None) -> List[LFIRawData]`

**Purpose:** Load multiple scans from a directory

**Parameters:**
- `directory` (str): Path to directory containing scan files
- `target_type` (str, optional): Filter by target type

**Returns:**
- List of `LFIRawData` objects

---

## Part 2: Signal Pre-processing (Mingxuan Tang)

### Module: `part2_preprocess`

### Main Functions

#### `preprocess_signal(raw_data: LFIRawData) -> ProcessedSignal`

**Purpose:** Apply filtering and noise reduction to raw signal

**Parameters:**
- `raw_data` (LFIRawData): Raw data from Part 1

**Returns:**
- `ProcessedSignal`: Cleaned signal with processing metadata

**Example:**
```python
from part1_data import load_single_scan
from part2_preprocess import preprocess_signal

raw = load_single_scan('data.h5')
processed = preprocess_signal(raw)
print(processed.noise_level)  # 0.05
```

#### `apply_custom_filter(signal: np.ndarray, filter_params: dict) -> np.ndarray`

**Purpose:** Apply custom filter with specified parameters

**Parameters:**
- `signal` (np.ndarray): Input signal
- `filter_params` (dict): Filter configuration

**Returns:**
- Filtered signal array

---

## Part 3: Image Generation & Reconstruction (Haobo Yu)

### Module: `part3_reconstruction`

### Main Functions

#### `reconstruct_image(processed_signal: ProcessedSignal) -> ReconstructedImage`

**Purpose:** Reconstruct image from processed signal

**Parameters:**
- `processed_signal` (ProcessedSignal): Cleaned signal from Part 2

**Returns:**
- `ReconstructedImage`: Reconstructed 2D/3D image data

**Example:**
```python
from part2_preprocess import preprocess_signal
from part3_reconstruction import reconstruct_image

processed = preprocess_signal(raw_data)
result = reconstruct_image(processed)
print(result.image_2d.shape)  # (512, 512)
```

#### `reconstruct_3d(signals: List[ProcessedSignal], z_levels: List[float]) -> ReconstructedImage`

**Purpose:** Reconstruct 3D volume from multiple z-depth scans

**Parameters:**
- `signals` (List[ProcessedSignal]): List of processed signals at different depths
- `z_levels` (List[float]): Corresponding z-depth values

**Returns:**
- `ReconstructedImage` with 3D data

---

## Part 4: UI Design & Integration (He Zhenyu / Vincent)

### Module: `part4_ui`

### Main Application

#### `main.py`

**Purpose:** Main UI application that integrates all modules

**Features:**
- File selection dialog
- Progress indication during processing
- Image display and visualization
- Export functionality

**Example Pipeline:**
```python
from part1_data import load_single_scan
from part2_preprocess import preprocess_signal
from part3_reconstruction import reconstruct_image

# Complete processing pipeline
def process_lfi_file(filepath):
    # Step 1: Load data
    raw_data = load_single_scan(filepath)
    
    # Step 2: Preprocess
    processed = preprocess_signal(raw_data)
    
    # Step 3: Reconstruct
    result = reconstruct_image(processed)
    
    return result

# Use in UI
result = process_lfi_file('datasets/Chip_001.h5')
display_image(result.image_2d)
```

---

## Data Flow Diagram

```
┌─────────────┐     LFIRawData      ┌──────────────┐
│   Part 1    │ ──────────────────> │    Part 2    │
│ Data Import │                     │ Preprocessing │
└─────────────┘                     └──────────────┘
                                           │
                                           │ ProcessedSignal
                                           ▼
                                    ┌──────────────┐
                                    │    Part 3    │
                                    │Reconstruction│
                                    └──────────────┘
                                           │
                                           │ ReconstructedImage
                                           ▼
                                    ┌──────────────┐
                                    │    Part 4    │
                                    │   UI & Display│
                                    └──────────────┘
```

---

## Testing Requirements

Each module must include:

1. **Unit tests** for all public functions
2. **Sample data** or mock data generators
3. **Documentation** with usage examples

### Test File Naming Convention

- `tests/test_part1.py` - Tests for Part 1
- `tests/test_part2.py` - Tests for Part 2
- `tests/test_part3.py` - Tests for Part 3
- `tests/test_part4.py` - Tests for Part 4

---

## Error Handling

All modules should use consistent error handling:

```python
class LFIError(Exception):
    """Base exception for LFI project"""
    pass

class DataLoadError(LFIError):
    """Error loading data"""
    pass

class ProcessingError(LFIError):
    """Error during signal processing"""
    pass

class ReconstructionError(LFIError):
    """Error during image reconstruction"""
    pass
```

---

## Version Compatibility

- Python: >= 3.8
- NumPy: >= 1.21.0
- See `requirements.txt` for full dependencies

---

## Questions?

If any interface is unclear or needs modification:
1. Discuss in team meeting
2. Update this document
3. Notify all team members of changes
