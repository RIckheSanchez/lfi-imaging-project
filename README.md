# LFI Imaging Project - MEGA

**Imaging using Laser Feedback Interferometry**

University of Queensland - Final Year Thesis Project  
Supervisor: Karl Bertling

## Team Members

| Part | Module | Responsible | Branch | Status |
|------|--------|-------------|---------|--------|
| Part 1 | Data Import & Management | Zeyu Wang | `zeyu/part1-data` | ✅ **Complete** |
| Part 2 | Signal Pre-processing | Yu Haobo | `haobo/part2-preprocess` | 🔄 In Progress |
| Part 3 | Image Generation & Reconstruction | Tang Mingxuan | `mingxuan/part3-reconstruction` | ⏳ Waiting |
| Part 4 | UI Design & Usability Evaluation | He Zhenyu (Vincent) | `vincent/part4-ui` | ⏳ Waiting |

## Project Structure

```
lfi-imaging-project/
├── part1_data/              # Data import and management (Zeyu)
│   ├── __init__.py
│   ├── loader.py
│   ├── data_types.py
│   └── utils.py
├── part2_preprocess/        # Signal pre-processing (Mingxuan)
│   ├── __init__.py
│   ├── filters.py
│   └── noise_reduction.py
├── part3_reconstruction/    # Image reconstruction (Haobo)
│   ├── __init__.py
│   ├── reconstruction.py
│   └── tomography.py
├── part4_ui/                # UI and integration (Vincent)
│   ├── __init__.py
│   ├── main.py
│   └── widgets.py
├── shared/                  # Shared types and utilities
│   ├── __init__.py
│   └── data_types.py
├── tests/                   # Unit tests
│   ├── test_part1.py
│   ├── test_part2.py
│   ├── test_part3.py
│   └── test_part4.py
├── docs/                    # Documentation
│   └── interface_spec.md
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/lfi-imaging-project.git
cd lfi-imaging-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Switch to your branch
```bash
# For Zeyu
git checkout zeyu/part1-data

# For Mingxuan
git checkout mingxuan/part2-preprocess

# For Haobo
git checkout haobo/part3-reconstruction

# For Vincent
git checkout vincent/part4-ui
```

## Development Workflow

### Working on your part
```bash
# 1. Make sure you're on your branch
git checkout your-branch-name

# 2. Make changes and commit
git add .
git commit -m "Description of changes"

# 3. Push to your branch
git push origin your-branch-name
```

### Integration workflow
```bash
# 1. Make sure your branch is up to date
git checkout your-branch-name
git pull origin main

# 2. Test your changes
python -m pytest tests/

# 3. Create pull request on GitHub
# Review by at least one team member before merging
```

## Branching Strategy

- `main` - Stable release branch (protected)
- `dev` - Integration branch for testing
- `zeyu/part1-data` - Zeyu's development branch
- `mingxuan/part2-preprocess` - Mingxuan's development branch
- `haobo/part3-reconstruction` - Haobo's development branch
- `vincent/part4-ui` - Vincent's development branch

## Module Interfaces

### Part 1 Output (Zeyu → Mingxuan)
```python
from shared.data_types import LFIRawData

data = load_single_scan('file.h5')
# Returns: LFIRawData with signal, coordinates, metadata
```

### Part 2 Output (Mingxuan → Haobo)
```python
from shared.data_types import ProcessedSignal

processed = preprocess_signal(raw_data)
# Returns: ProcessedSignal with cleaned signal
```

### Part 3 Output (Haobo → Vincent)
```python
from shared.data_types import ReconstructedImage

image = reconstruct_image(processed_signal)
# Returns: ReconstructedImage with 2D/3D data
```

### Part 4 Integration (Vincent)
```python
# UI calls all three modules in sequence
from part1_data import load_single_scan
from part2_preprocess import preprocess_signal
from part3_reconstruction import reconstruct_image

# Pipeline
data = load_single_scan(filepath)
processed = preprocess_signal(data)
result = reconstruct_image(processed)
# Display in UI
```

## Testing

Run all tests:
```bash
pytest tests/
```

Run specific module tests:
```bash
pytest tests/test_part1.py
```

## Documentation

See `docs/` folder for detailed documentation:
- Interface specifications
- Data format definitions
- API references

## Contact

For questions about:
- **Data loading**: Contact Zeyu Wang
- **Signal processing**: Contact Mingxuan Tang
- **Image reconstruction**: Contact Haobo Yu
- **UI/Integration**: Contact He Zhenyu (Vincent)
- **Overall project**: Contact supervisor Karl Bertling

## License

University of Queensland - Academic Use Only
