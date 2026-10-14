"""
benchmark.py - load performance: time, peak RSS, float32 comparison, batch

Usage:
    MPLBACKEND=Agg python3 benchmark.py
    python3 benchmark.py --leaf-partial   # Leaf: only 3 frames (faster)
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np

from loader import find_data_files, load_single_scan, load_volume

_HERE = Path(__file__).resolve().parent
VEG = _HERE / "data" / "2026_Vegetables"
LEAF = _HERE / "data" / "Leaf" / "leaf1.mat"
MARBLE = _HERE / "data" / "Marble"


def _rss_mb() -> float:
    # macOS/BSD: ru_maxrss is bytes; Linux: KB
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 ** 2)
    return rss / 1024


def _row(name: str, elapsed: float, shape, extra: str = ""):
    mem = _rss_mb()
    print(f"{name:<36} {elapsed:>7.2f}s  peakRSS≈{mem:>7.0f}MB  {shape}  {extra}")


def bench_marble():
    files = find_data_files(MARBLE, recursive=False)
    if not files:
        print("skip Marble: no files")
        return
    t0 = time.perf_counter()
    d = load_single_scan(str(files[0]))
    _row("Marble single", time.perf_counter() - t0, d.amplitude.shape)


def bench_vegetables_batch():
    files = find_data_files(VEG, recursive=False)
    files = [p for p in files if p.suffix.lower() in (".hdf5", ".h5")]
    if len(files) < 2:
        print("skip Vegetables batch: not enough files")
        return
    t0 = time.perf_counter()
    n = 0
    for p in files:
        try:
            load_single_scan(str(p), exclude_anomalous=False)
            n += 1
        except Exception as e:
            print(f"  failed {p.name}: {e}")
    _row(f"Vegetables batch ({n} files)", time.perf_counter() - t0, f"{n} files")


def bench_volume():
    if not VEG.exists():
        print("skip load_volume")
        return
    frames = [21, 22, 28]
    if not all((VEG / f"frame-{i}.hdf5").exists() for i in frames):
        print("skip load_volume: missing frame 21/22/28")
        return
    t0 = time.perf_counter()
    vol = load_volume(str(VEG), frame_ids=frames)
    _row("load_volume 3 frames", time.perf_counter() - t0, vol.amplitude.shape)


def bench_leaf(partial: bool):
    if not LEAF.exists():
        print("skip Leaf: no leaf1.mat")
        return
    t0 = time.perf_counter()
    if partial:
        d = load_single_scan(str(LEAF), frame_ids=[0, 10, 20])
        tag = "Leaf partial 3 frames"
    else:
        d = load_single_scan(str(LEAF))
        tag = "Leaf full 475 frames"
    elapsed = time.perf_counter() - t0
    f64 = d.amplitude.nbytes + d.phase.nbytes
    f32 = (d.amplitude.astype(np.float32).nbytes + d.phase.astype(np.float32).nbytes)
    _row(tag, elapsed, d.amplitude.shape, f"f64≈{f64/1e6:.0f}MB  f32≈{f32/1e6:.0f}MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf-partial", action="store_true", help="Leaf: only 3 frames")
    ap.add_argument("--leaf-full", action="store_true", help="Leaf full load (slow, high RAM)")
    args = ap.parse_args()

    print("=" * 72)
    print(f"{'Case':<36} {'Time':>8}  {'Memory':>16}  Shape")
    print("=" * 72)
    bench_marble()
    bench_vegetables_batch()
    bench_volume()
    if args.leaf_full:
        bench_leaf(partial=False)
    else:
        bench_leaf(partial=True if args.leaf_partial or LEAF.exists() else True)
        if LEAF.exists() and not args.leaf_partial and not args.leaf_full:
            print("(default Leaf is partial; use --leaf-full for all frames)")
    print("=" * 72)


if __name__ == "__main__":
    main()
