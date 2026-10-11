"""Process full spatial images: Skin, every complete Vegetables frame and 91 Leaf images.

Usage: PYTHONPATH=src .venv/bin/python -m examples.run_spatial_processing
Outputs go to output/spatial_v06; the superseded v0.5 batch stays in output/spatial_v05.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thz_preprocessing import api, load_leaf_mat, load_skin_hdf5, load_rdm_hdf5_frame
from thz_preprocessing.io import LFIRawData
from thz_preprocessing.display import oriented, pixel_aspect
from examples.run_pilot import _json_ready


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/spatial_v06"


def render_spatial(raw, processed, output, index=0):
    """Greyscale panels at the physical aspect ratio; phase panels state their status."""
    p = processed.metadata["spatial_products"]
    target = raw.target_type
    view = lambda a: oriented(a, target)
    aspect = pixel_aspect(view(raw.amplitude[index]).shape, raw.metadata)
    status = processed.metadata["phase_unwrapping_status"]
    slice_status = (status["quality_diagnostics"][index]["status"]
                    if status["quality_diagnostics"] else status["status"])
    phase_ok = bool(p["phase_correction_available"][index])

    def limits(values, low=1, high=99):
        finite = values[np.isfinite(values)]
        return np.percentile(finite, [low, high])

    amp = raw.amplitude[index]
    lo, hi = limits(amp)
    panels = [
        (amp, "Raw amplitude", "gray", lo, hi),
        (p["amplitude_corrected"][index], "amplitude_corrected: line offsets removed, contrast kept", "gray", lo, hi),
        (p["amplitude_detail"][index], "amplitude_detail: minus Gaussian background (relative view)", "gray",
         *limits(p["amplitude_detail"][index], 2, 98)),
        (raw.phase[index], "Raw wrapped phase (rad)", "twilight", -np.pi, np.pi),
    ]
    if phase_ok:
        unwrapped = processed.phase[index]
        panels += [
            (unwrapped, "Unwrapped phase (%s)" % slice_status, "magma", *limits(unwrapped, 0.5, 99.5)),
            (p["phase_corrected"][index], "phase_corrected: minus background and line offsets (relative)", "gray",
             *limits(p["phase_corrected"][index], 2, 98)),
        ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 7.6), constrained_layout=True)
    fig.suptitle("%s | source image %s | v%s" % (target, raw.z_indices[index],
                                                processed.metadata["version"]), fontsize=13)
    for ax, (values, title, cmap, low, high) in zip(axes.flat, panels):
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("white")
        im = ax.imshow(view(values), cmap=cm, vmin=low, vmax=high, aspect=aspect, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.03)
    if not phase_ok:
        quality = status["quality_diagnostics"][index] if status["quality_diagnostics"] else {}
        reason = ("Phase kept wrapped\nstatus: %s\nresidues in signal region: %.2f%% (gate %.2f%%)"
                  % (slice_status, 100 * quality.get("residue_fraction", float("nan")),
                     100 * quality.get("max_residue_fraction", float("nan"))))
        for ax in axes.flat[4:]:
            ax.axis("off")
        axes.flat[4].text(0.5, 0.5, reason, transform=axes.flat[4].transAxes, ha="center", va="center", fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_case(name, raw, figure=False):
    out = api.run_pipeline(raw)
    products = out.metadata["spatial_products"]
    np.savez_compressed(OUTPUT / (name + ".npz"), amplitude=out.amplitude, phase=out.phase,
                        z_indices=np.array(out.z_indices), **products)
    metadata = {"name": name, "shape": list(raw.shape), "metadata": _json_ready(out.metadata)}
    (OUTPUT / (name + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    if figure:
        render_spatial(raw, out, OUTPUT / (name + ".png"))
    status = out.metadata["phase_unwrapping_status"]
    return {"name": name, "source_indices": raw.z_indices,
            "phase_status": status["status"],
            "phase_quality": status["quality_diagnostics"],
            "spatial_diagnostics": out.metadata["spatial_correction_status"]["diagnostics"],
            "pipeline_seconds": out.metadata["timings_seconds"]["total"]}


def vegetable_frame_ids():
    """Complete local 2026 Vegetables frames; frame 20 is an anomalous start-up frame."""
    ids = sorted(int(p.stem.split("-")[1]) for p in (ROOT / "data/vegetables").glob("frame-*.hdf5"))
    return [i for i in ids if i != 20]


def main(leaf_count=91):
    if not 1 <= leaf_count <= 475:
        raise ValueError("Use 1..475 Leaf images")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    skin = load_skin_hdf5(ROOT / "data/skin/2025-11-28_10-14-26.594507929.hdf5")
    cases.append(save_case("skin_frame40_spatial", skin, figure=True))
    for i in vegetable_frame_ids():
        raw = load_rdm_hdf5_frame(ROOT / ("data/vegetables/frame-%d.hdf5" % i))
        cases.append(save_case("vegetable_frame%d_spatial" % i, raw,
                               figure=i in {21, 80, 130, 140}))
    print("Skin and Vegetables spatial outputs saved", flush=True)
    # Leaf phase is decoded on the supervisor-stated scale and unwrapped by the module.
    leaf = load_leaf_mat(ROOT / "data/leaf/leaf1.mat", z_slice=slice(0, leaf_count))
    for i in range(leaf_count):
        raw = LFIRawData(leaf.amplitude[i:i+1], leaf.phase[i:i+1], [leaf.z_indices[i]],
                         "Leaf", dict(leaf.metadata))
        cases.append(save_case("leaf_frame%03d_spatial" % i, raw, figure=i in {0, 45, 90}))
        if i % 10 == 0 or i == leaf_count - 1:
            print("Leaf full-resolution frames completed: %d/%d" % (i + 1, leaf_count), flush=True)
    released = {}
    for case in cases:
        kind = case["name"].split("_")[0]
        released.setdefault(kind, [0, 0])
        released[kind][1] += 1
        released[kind][0] += case["phase_status"] in {"unwrapped", "already_continuous"}
    summary = {
        "version": "0.6.1", "leaf_frame_count": leaf_count,
        "spatial_dimensions_leaf": [1200, 400], "cases": cases,
        "phase_released_of_total": {k: "%d/%d" % tuple(v) for k, v in released.items()},
        "leaf_phase_scale": "code * pi / 8192 (supervisor-stated); the saved L_P_U / L_P_U_M use twice this scale",
        "notch_status": "not applied without verified artefact frequencies",
        "calibration_status": "no verified physical pixel size, acquisition timestamps or reference ROI",
        "channel_contract": "amplitude/phase are measurement channels; amplitude_corrected keeps object contrast; amplitude_detail and phase_corrected are relative views; inspect phase_correction_available and signal_mask",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-count", type=int, default=91)
    main(parser.parse_args().leaf_count)
