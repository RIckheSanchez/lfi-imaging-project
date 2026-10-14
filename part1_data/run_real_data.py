"""
run_real_data.py
Batch-run the loader on RDM-downloaded real data and write a report

Usage:
    python3 run_real_data.py data
    python3 run_real_data.py data --out results_real

Output (default results_real/)::
    report.md               (share this with your teammate)
    report.csv              tabular version of the same content
    previews/*.png          per-file amplitude/phase preview
    structure_<files>.txt  HDF5 structure dump for failed loads (to fix the loader)

: keep RDM folder names after download (e.g. 2026_Vegetables), 
      the loader uses folder names for dataset identity and thus the phase formula.
"""

import argparse
import csv
import os
import traceback
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np

from loader import load_single_scan, _infer_target_type


COLUMNS = [
    "file", "target_type", "status", "file_format", "shape",
    "frame_ids", "anomalous_frame_ids", "original_dtype",
    "phase_encoding_used", "phase_converted",
    "amp_min", "amp_max", "phase_min", "phase_max", "error",
]


def _find_data_files(root: Path) -> list:
    from loader import find_data_files
    return find_data_files(root, recursive=True)


def _dump_structure(fp: Path, out_dir: Path) -> None:
    """Dump HDF5 trees for failed files to help fix the loader."""
    out = out_dir / f"structure_{fp.stem}.txt"
    lines = [f"files: {fp}", ""]
    try:
        with h5py.File(fp, "r") as f:
            lines.append(f"top-level keys: {list(f.keys())}")
            lines.append(f"files attrs: {dict(f.attrs)}")
            lines.append("")

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    lines.append(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    lines.append(f"  [{name}/]")
            f.visititems(visit)
    except Exception as e:
        lines.append(f"cannot open with h5py (maybe .mat v5 or corrupt): {e}")
    out.write_text("\n".join(lines), encoding="utf-8")


def _save_preview(data, fp: Path, preview_dir: Path) -> None:
    normal = [
        i for i, z in enumerate(data.z_indices)
        if z not in data.metadata.get("anomalous_frame_ids", [])
    ]
    idx = normal[len(normal) // 2] if normal else 0
    z = data.z_indices[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(data.amplitude[idx], cmap="hot", aspect="auto")
    axes[0].set_title(f"Amplitude - id={z}")
    plt.colorbar(im0, ax=axes[0])

    phase_label = "Phase (rad)" if data.metadata.get("phase_converted") else "Phase (raw codes)"
    im1 = axes[1].imshow(data.phase[idx], cmap="RdBu", aspect="auto")
    axes[1].set_title(f"{phase_label} - id={z}")
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle(
        f"{data.target_type} | {fp.name} | "
        f"encoding={data.metadata.get('phase_encoding_used')}"
    )
    plt.tight_layout()
    plt.savefig(preview_dir / f"{fp.parent.name}_{fp.stem}.png", dpi=100)
    plt.close(fig)


def _run_one(fp: Path, out_dir: Path, preview_dir: Path) -> dict:
    row = {c: "" for c in COLUMNS}
    row["file"] = str(fp)
    row["target_type"] = _infer_target_type(str(fp))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # keep anomalous frames (e.g. Vegetables 20), flagged in metadata, not silently dropped
            data = load_single_scan(str(fp), exclude_anomalous=False)
    except Exception as e:
        row["status"] = "FAILED"
        row["error"] = f"{type(e).__name__}: {e}"
        if fp.suffix != ".mat":
            _dump_structure(fp, out_dir)
        traceback.print_exc()
        return row

    m = data.metadata
    row.update({
        "status": "OK",
        "target_type": data.target_type,
        "file_format": m.get("file_format"),
        "shape": "x".join(str(s) for s in data.amplitude.shape),
        "frame_ids": _short_list(m.get("frame_ids", [])),
        "anomalous_frame_ids": m.get("anomalous_frame_ids", []),
        "original_dtype": m.get("original_dtype"),
        "phase_encoding_used": m.get("phase_encoding_used"),
        "phase_converted": m.get("phase_converted"),
        "amp_min": f"{float(np.nanmin(data.amplitude)):.4g}",
        "amp_max": f"{float(np.nanmax(data.amplitude)):.4g}",
        "phase_min": f"{float(np.nanmin(data.phase)):.4g}",
        "phase_max": f"{float(np.nanmax(data.phase)):.4g}",
    })
    _save_preview(data, fp, preview_dir)
    return row


def _short_list(ids: list) -> str:
    if len(ids) <= 6:
        return str(ids)
    return f"[{ids[0]}, {ids[1]}, ..., {ids[-1]}] ({len(ids)} items)"


def _write_reports(rows: list, out_dir: Path, data_root: Path) -> None:
    with open(out_dir / "report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_ok = sum(r["status"] == "OK" for r in rows)
    n_fail = sum(r["status"] == "FAILED" for r in rows)
    n_skip = sum(r["status"] == "SKIPPED" for r in rows)

    md_cols = [
        "file", "target_type", "status", "shape", "frame_ids",
        "anomalous_frame_ids", "original_dtype",
        "phase_encoding_used", "phase_converted", "error",
    ]
    lines = [
        "# Real-data loader run results",
        "",
        f"data directory: `{data_root.resolve()}`",
        "",
        f"total {len(rows)} files: ok {n_ok}, failed {n_fail}, skip {n_skip}",
        "",
        "| " + " | ".join(md_cols) + " |",
        "|" + "---|" * len(md_cols),
    ]
    for r in rows:
        cells = []
        for c in md_cols:
            v = r[c]
            if c == "file":
                v = Path(v).relative_to(data_root) if data_root in Path(v).parents else v
            cells.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "notes:",
        "- Phase formula from dataset identity (folder name) lookup; do not infer from value ranges.",
        "- `phase_converted=False` means phase kept as raw codes: formula unconfirmed or file has anomalous frames.",
        "- `anomalous_frame_ids` non-emptyframes( Vegetables frame 20)load, must not be treated as normal data.",
        "- HDF5 structure dumps for failed files: `structure_*.txt`.",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Batch-run loader on real data")
    parser.add_argument("data_dir", help="RDM-downloaded data root")
    parser.add_argument("--out", default="results_real", help="output directory")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    if not data_root.exists():
        raise SystemExit(f"data directory does not exist: {data_root}")

    out_dir = Path(args.out)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    files = _find_data_files(data_root)
    if not files:
        raise SystemExit(f"{data_root} found no .hdf5 / .h5 / .mat files")

    print(f"Found {len(files)}  files\n")
    rows = []
    for i, fp in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fp}")
        row = _run_one(fp, out_dir, preview_dir)
        print(f"    → {row['status']} {row['error']}")
        rows.append(row)

    _write_reports(rows, out_dir, data_root)
    print(f"\ndone.report: {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
