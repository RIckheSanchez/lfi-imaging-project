"""
inspect_real.py
real datafilesstructure(HDF5 / CBOR), side-by-side loader

Usage:
    python3 inspect_real.py data/2026_Vegetables
"""

import sys
from pathlib import Path

import h5py
import numpy as np


def inspect_hdf5(fp: Path) -> None:
    print(f"\n{'=' * 70}\nHDF5: {fp.name}  ({fp.stat().st_size / 1e6:.1f} MB)\n{'=' * 70}")
    with h5py.File(fp, "r") as f:
        print(f"top-level keys: {list(f.keys())}")
        print(f"files attrs: {dict(f.attrs)}")

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                line = f"  {name}: shape={obj.shape}, dtype={obj.dtype}"
                leaf = name.rsplit("/", 1)[-1]
                if leaf in ("IMAGE_PHASE", "IMAGE_AMPLITUDE") or obj.ndim >= 2:
                    a = obj[...]
                    if a.size and np.issubdtype(a.dtype, np.number):
                        line += f", min={a.min()}, max={a.max()}"
                print(line)
            else:
                attrs = dict(obj.attrs)
                print(f"  [{name}/]" + (f" attrs={attrs}" if attrs else ""))

        f.visititems(visit)


def _summarise(obj, depth=0, max_depth=4):
    pad = "  " * depth
    if depth > max_depth:
        print(f"{pad}...")
        return
    if isinstance(obj, dict):
        for k, v in list(obj.items())[:50]:
            if isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__}({len(v)})")
                _summarise(v, depth + 1, max_depth)
            else:
                s = repr(v)
                print(f"{pad}{k}: {s[:120]}{'...' if len(s) > 120 else ''}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:5]):
            print(f"{pad}[{i}]: {type(v).__name__}")
            _summarise(v, depth + 1, max_depth)
        if len(obj) > 5:
            print(f"{pad}... total {len(obj)}  items")


def inspect_cbor(fp: Path) -> None:
    print(f"\n{'=' * 70}\nCBOR: {fp.name}  ({fp.stat().st_size / 1e6:.1f} MB)\n{'=' * 70}")
    try:
        import cbor2
    except ImportError:
        print("cbor2 missing; run: pip3 install cbor2")
        return
    with open(fp, "rb") as f:
        obj = cbor2.load(f)
    print(f"top-level type: {type(obj).__name__}")
    _summarise(obj)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    files = sorted(root.rglob("*"))
    for fp in files:
        if fp.suffix in (".hdf5", ".h5"):
            inspect_hdf5(fp)
        elif fp.suffix == ".cbor":
            inspect_cbor(fp)


if __name__ == "__main__":
    main()
