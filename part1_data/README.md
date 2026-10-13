# Part 1 — LFI data loader

Public API: `output_interface.py` (see `docs/INTEGRATION.md`).

```bash
pip install -r requirements.txt
python3 -m pytest -q tests/
python3 output_interface.py examples/Marble --out /tmp/lfi_exported
```

```python
from output_interface import load_single_scan, load_volume, get_phase_radians

scan = load_single_scan("examples/Marble/maeble10%_0.hdf5")
# Vegetables volume (after placing RDM frames under a folder):
# vol = load_volume("path/to/2026_Vegetables")
# phase = get_phase_radians(vol)
```

Do not call underscore-prefixed helpers in `loader.py` from downstream code.
