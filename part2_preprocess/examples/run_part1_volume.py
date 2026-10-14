"""Load a 2026_Vegetables folder with Part 1's ``load_volume`` and run Part 2.

Usage (from the repository root, with Part 1 present):
    python part2_preprocess/examples/run_part1_volume.py path/to/2026_Vegetables
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from part1_data import load_volume  # noqa: E402
from part2_preprocess.pipeline import preprocess_part1  # noqa: E402

folder = sys.argv[1]
start = time.perf_counter()
volume = load_volume(folder)
print("loaded", volume.amplitude.shape, "frames", volume.z_indices,
      "in %.1f s" % (time.perf_counter() - start))
start = time.perf_counter()
out = preprocess_part1(volume)
products = out.metadata["spatial_products"]
print("pre-processed in %.1f s" % (time.perf_counter() - start))
print("phase unwrapping:", out.metadata["phase_unwrapping_status"]["status"])
print("phase released for", int(np.asarray(products["phase_correction_available"]).sum()),
      "of", len(volume.z_indices), "frames")
