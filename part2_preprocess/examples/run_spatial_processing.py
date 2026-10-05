"""Process full spatial images, including all 91 reference-matched Leaf frames.

Usage: PYTHONPATH=src .venv/bin/python -m examples.run_spatial_processing
Outputs are kept separately from the old v0.4 pilot archives.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from thz_preprocessing import api, load_leaf_mat, load_skin_hdf5, load_rdm_hdf5_frame
from thz_preprocessing.io import LFIRawData
from examples.run_pilot import _json_ready


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/spatial_v05"


def render_spatial(raw, processed, output, index=0):
    p = processed.metadata["spatial_products"]
    amp = raw.amplitude[index]
    background = p["amplitude_background"][index]
    corrected = p["amplitude_corrected"][index]
    lo, hi = np.percentile(amp, [1, 99])
    limit = max(float(np.percentile(np.abs(corrected), 99)), 1e-9)
    phase_ok = p["phase_correction_available"][index]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8), constrained_layout=True)
    fig.suptitle("%s | source image %s | spatial correction (exploratory)" %
                 (raw.target_type, raw.z_indices[index]), fontsize=14)
    panels = [(amp, "Raw amplitude", "viridis", lo, hi),
              (background, "Estimated spatial background (raw scale)", "viridis", lo, hi),
              (corrected, "Background + line corrected amplitude (detail scale)", "RdBu_r", -limit, limit),
              (raw.phase[index], "Phase under unconfirmed scale" if raw.metadata.get("phase_encoding_status") == "unconfirmed"
               else "Raw wrapped phase (radians)", "twilight", -np.pi, np.pi)]
    if phase_ok:
        phase = p["phase_corrected"][index]
        phase_bg = p["phase_background"][index]
        phase_limit = max(float(np.percentile(np.abs(phase), 99)), 1e-9)
        bg_limit = max(float(np.percentile(np.abs(phase_bg), 99)), 1e-9)
        panels += [(phase_bg, "Removed phase background (radians)", "RdBu_r", -bg_limit, bg_limit),
                   (phase, "Relative phase after background + line correction", "RdBu_r", -phase_limit, phase_limit)]
    else:
        stripe = p["amplitude_stripe"][index]
        stripe_limit = max(float(np.percentile(np.abs(stripe), 99)), 1e-9)
        panels += [(stripe, "Removed amplitude line offsets", "RdBu_r", -stripe_limit, stripe_limit)]
        axes[1, 2].axis("off")
        reason = "Encoding scale is unconfirmed" if raw.metadata.get("phase_encoding_status") == "unconfirmed" else "Unwrap quality gate withheld processing"
        axes[1, 2].text(0.5, 0.5, "Phase correction withheld\n%s\nInput phase is preserved" % reason,
                        transform=axes[1, 2].transAxes, ha="center", va="center", fontsize=12)
    for ax, (values, title, cmap, low, high) in zip(axes.flat, panels):
        display = values.T if raw.target_type == "Leaf" else values
        im = ax.imshow(display, cmap=cmap, vmin=low, vmax=high, aspect="auto",
                       interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.065, pad=0.04)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_case(name, raw, reference_u=None, reference_m=None, figure=False):
    overrides = None
    if reference_u is not None:
        raw.metadata["phase_reference"] = reference_u[None, :, :]
        raw.metadata["phase_reference_source"] = {
            "file": str(ROOT / "data/leaf/leaf1.mat"), "variable": "L_P_U",
            "stored_reference_index": raw.z_indices[0],
            "source_status": "saved group output, not physical ground truth",
        }
        overrides = {"phase_unwrapping": {"method": "validated_reference"}}
    out = api.run_pipeline(raw, config_overrides=overrides)
    products = out.metadata["spatial_products"]
    np.savez_compressed(OUTPUT / (name + ".npz"), amplitude=out.amplitude, phase=out.phase,
                        z_indices=np.array(out.z_indices), **products)
    metrics = {}
    if reference_u is not None and products["phase_correction_available"][0]:
        delta = out.phase[0] - reference_u
        offset = 2 * np.pi * round(float(np.median(delta)) / (2 * np.pi))
        metrics["unwrapped_reference_rmse_after_global_cycle_alignment"] = float(
            np.sqrt(np.mean((delta - offset)**2)))
        metrics["reference_global_cycle_offset_radians"] = offset
        corrected = products["phase_corrected"][0]
        background_only = products["phase_background_corrected"][0]
        metrics["background_only_reference_rmse_radians"] = float(
            np.sqrt(np.mean((background_only - reference_m)**2)))
        metrics["background_only_reference_max_error_radians"] = float(
            np.abs(background_only - reference_m).max())
        metrics["relative_phase_correlation_with_saved_u_m"] = float(
            np.corrcoef(corrected.ravel(), reference_m.ravel())[0, 1])
        metrics["relative_phase_rmse_with_saved_u_m"] = float(
            np.sqrt(np.mean((corrected - reference_m)**2)))
        metrics["reference_note"] = "Historical reference agreement, not physical or clinical ground truth"
    metadata = {"name": name, "shape": list(raw.shape), "reference_comparison": metrics,
                "metadata": _json_ready(out.metadata)}
    (OUTPUT / (name + "_metadata.json")).write_text(json.dumps(metadata, indent=2) + "\n")
    if figure:
        render_spatial(raw, out, OUTPUT / (name + ".png"))
    return {"name": name, "source_indices": raw.z_indices,
            "phase_status": out.metadata["phase_unwrapping_status"]["status"],
            "spatial_diagnostics": out.metadata["spatial_correction_status"]["diagnostics"],
            "reference_comparison": metrics,
            "pipeline_seconds": out.metadata["timings_seconds"]["total"]}


def main(leaf_count=91):
    if not 1 <= leaf_count <= 91:
        raise ValueError("Use 1..91 frames with saved reference correspondence")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    skin = load_skin_hdf5(ROOT / "data/skin/2025-11-28_10-14-26.594507929.hdf5")
    cases.append(save_case("skin_frame40_spatial", skin, figure=True))
    for i in [21, 22]:
        raw = load_rdm_hdf5_frame(ROOT / ("data/vegetables/frame-%d.hdf5" % i))
        cases.append(save_case("vegetable_frame%d_spatial" % i, raw, figure=True))
    print("Skin and Vegetables spatial outputs saved", flush=True)
    leaf = load_leaf_mat(ROOT / "data/leaf/leaf1.mat", z_slice=slice(0, leaf_count))
    reference = loadmat(ROOT / "data/leaf/leaf1.mat", variable_names=["L_P_U", "L_P_U_M"])
    for i in range(leaf_count):
        raw = LFIRawData(leaf.amplitude[i:i+1], leaf.phase[i:i+1], [leaf.z_indices[i]],
                         "Leaf", dict(leaf.metadata))
        cases.append(save_case("leaf_frame%03d_spatial" % i, raw,
                                reference["L_P_U"][:, :, i], reference["L_P_U_M"][:, :, i],
                                figure=i in {0, 45, 90}))
        if i % 10 == 0 or i == leaf_count - 1:
            print("Leaf full-resolution frames completed: %d/%d" % (i + 1, leaf_count), flush=True)
    metrics = [c["reference_comparison"] for c in cases if c["reference_comparison"]]
    summary = {
        "version": "0.5.0", "leaf_frame_count": leaf_count,
        "spatial_dimensions_leaf": [1200, 400], "cases": cases,
        "max_reference_unwrap_rmse": max((m["unwrapped_reference_rmse_after_global_cycle_alignment"] for m in metrics), default=None),
        "reference_reuse_note": "Leaf U is reused after modulo validation; zero unwrap RMSE is expected, not independent algorithm validation",
        "max_background_only_reference_error_radians": max((m["background_only_reference_max_error_radians"] for m in metrics), default=None),
        "median_corrected_phase_reference_correlation": float(np.median(
            [m["relative_phase_correlation_with_saved_u_m"] for m in metrics])) if metrics else None,
        "notch_status": "not applied without verified artefact frequencies",
        "calibration_status": "no verified physical pixel size, acquisition timestamps or reference ROI",
        "channel_contract": "amplitude/phase remain measurement channels; *_corrected are relative spatial products; inspect phase_correction_available",
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leaf-count", type=int, default=91)
    main(parser.parse_args().leaf_count)
