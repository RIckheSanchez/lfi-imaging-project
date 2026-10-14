"""Measure wall time and peak traced memory for the main Part 3 operations.

tracemalloc sees NumPy/SciPy array allocations made through Python's allocator;
it is a lower bound on process RSS, reported as such.
"""
import platform
import time
import tracemalloc
from pathlib import Path
import numpy as np
from .data import ImagingData, DataLoader
from .image_generator import ImageGenerator
from .volume_reconstructor import VolumeReconstructor
from .integration import figure_to_png


def _measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"seconds": round(elapsed, 4), "peak_traced_mb": round(peak / 2**20, 2)}


def synthetic_volume(nz, ny, nx, seed=0):
    rng = np.random.default_rng(seed)
    z, y, x = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij", sparse=True)
    amplitude = np.exp(-((x - nx/2)**2 + (y - ny/2)**2) / (nx*ny/8)) + 0.01*rng.standard_normal((nz, ny, nx))
    phase = 0.01*x + 0.02*y + 0.5*z + np.zeros((nz, ny, nx))
    return ImagingData(amplitude, phase, {"phase_state": "unwrapped", "phase_unit": "rad"})


def run(shapes=((10, 128, 128), (20, 256, 256), (40, 400, 400)), z_step=2.0, real_files=(), out_dir=None):
    rows = []
    for shape in shapes:
        data = synthetic_volume(*shape)
        vol = VolumeReconstructor(data, spacing_xy=(1, 1), z_coordinates=np.arange(shape[0]) * z_step)
        row = {"case": "synthetic", "shape_zyx": list(shape)}
        _, row["render_2d_comparison"] = _measure(lambda: figure_to_png(ImageGenerator(data.slice(shape[0]//2)).generate_comparison()))
        resampled, row["resample_to_1mm"] = _measure(lambda: vol.resample_isotropic(1.0, max_voxels=10**9))
        row["resampled_shape_zyx"] = list(resampled.data.amplitude.shape)
        if out_dir is not None:
            path = Path(out_dir) / f"bench_{'x'.join(map(str, shape))}.nii.gz"
            _, row["export_nifti"] = _measure(lambda: resampled.export_nifti(path))
        rows.append(row)
    for path in real_files:
        for side in ("left", "right"):
            row = {"case": "uq_frame", "file": Path(path).name, "side": side}
            data, row["load"] = _measure(lambda: DataLoader.load_uq_frame(path, side=side))
            row["shape_yx"] = list(data.amplitude.shape)
            _, row["render_2d_comparison"] = _measure(lambda: figure_to_png(ImageGenerator(data).generate_comparison(phase_mode="sequential")))
            rows.append(row)
    return {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "machine": platform.machine(), "system": platform.system()},
            "note": "render_2d_comparison includes PNG encoding at 150 dpi; peak_traced_mb is tracemalloc peak (lower bound on RSS)", "results": rows}
