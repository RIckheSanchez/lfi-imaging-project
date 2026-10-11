"""Quantitative evaluation of the pre-processing module, v0.6 (thesis Chapter 5).

Usage:
    PYTHONPATH=src .venv/bin/python -m examples.evaluate

Requires the v0.6 spatial products (``python -m examples.run_spatial_processing``)
and the local Skin, Leaf and Vegetables data. Writes tables (CSV), figures (PNG)
and ``evaluation_summary.json`` to ``output/evaluation_v06/``.

Experiments
-----------
E1   Repeatability: run the pipeline twice per case, compare SHA-256 of outputs.
E2   Computational cost: load time, pipeline time and peak traced memory.
E3   Leaf historical-reference agreement over all 91 matched frames.
E4   Controlled artefact injection (known ground truth) with stage ablation.
E5   Background-scale sensitivity: synthetic recovery and Leaf U_M agreement.
E6   Real-data indicators: line offsets removed, contrast and detail retained.
E7   Slice-wise noise reduction: noise estimate versus edge preservation.
E8   Wrapped-phase residue density, whole image and inside the signal mask.
E9   Vegetables depth series: volume cost, continuity, indicators versus depth.
E10  Skin phase encoding: evidence for the signed-14 scale.
E11  Independent 2D unwrap versus the group's saved Leaf reference.
Plus two review figures: a three-dataset gallery and v0.5 versus v0.6.

No metric here is clinical or physical ground truth. Saved Leaf references are
historical group outputs; injected artefacts are synthetic by construction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import tracemalloc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.io import loadmat

from thz_preprocessing import api, load_leaf_mat, load_rdm_hdf5_frame, load_skin_hdf5
from thz_preprocessing.display import oriented, pixel_aspect
from thz_preprocessing.io import LFIRawData
from thz_preprocessing.phase_unwrapping import _wrapped_phase_quality
from thz_preprocessing.spatial_correction import correct_image, estimate_line_offsets

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / "output/spatial_v06"
PREVIOUS = ROOT / "output/spatial_v05"  # superseded batch, used only for the review figure
OUT = ROOT / "output/evaluation_v06"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LEAF_MAT = ROOT / "data/leaf/leaf1.mat"
SKIN_H5 = ROOT / "data/skin/2025-11-28_10-14-26.594507929.hdf5"
VEG_H5 = str(ROOT / "data/vegetables") + "/frame-%d.hdf5"
LEAF_PARAMS = {"background_sigma_pixels": 20.0, "background_boundary_mode": "nearest",
               "background_truncate": 2.0, "stripe_axis": "rows",
               "stripe_window_lines": 15}
ASPECT = {"Leaf": 1.5, "Skin": 1.9651163, "Vegetable": 1.9651163}


# ---------------------------------------------------------------- helpers

def write_csv(name, rows):
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def corr(a, b):
    return float(np.corrcoef(np.ravel(a), np.ravel(b))[0, 1])


def highpass(image, sigma=2.0):
    return image - ndimage.gaussian_filter(image, sigma, mode="reflect")


def line_offset_std(image, valid=None):
    """Std of the robust per-line offsets (see estimate_line_offsets), input units."""
    work = np.where(np.isfinite(image), image, 0.0)
    ok = np.isfinite(image) if valid is None else (valid & np.isfinite(image))
    return float(estimate_line_offsets(work, "rows", 15, None if ok.all() else ok).std())


def background_std(image, sigma=50.0, valid=None):
    """Std of the very-low-pass (sigma=50 px) component, in input units."""
    if valid is None:
        return float(ndimage.gaussian_filter(image, sigma, mode="nearest").std())
    weights = ndimage.gaussian_filter(valid.astype(float), sigma, mode="nearest")
    low = ndimage.gaussian_filter(np.where(valid, image, 0.0), sigma, mode="nearest") / np.maximum(weights, 1e-9)
    return float(low[valid].std())


def show(ax, image, target, title, cmap="gray", limits=None, percentiles=(1, 99)):
    """Greyscale panel at the physical aspect ratio."""
    view = oriented(image, target)
    finite = view[np.isfinite(view)]
    low, high = limits if limits is not None else np.percentile(finite, percentiles)
    colours = plt.get_cmap(cmap).copy()
    colours.set_bad("white")
    ax.imshow(view, cmap=colours, vmin=low, vmax=high, interpolation="nearest",
              aspect=pixel_aspect(view.shape, {"display_aspect_ratio": ASPECT[target]}))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def noise_sigma(image):
    """Immerkaer (1996) fast noise standard deviation estimate."""
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = ndimage.convolve(np.asarray(image, dtype=np.float64), kernel, mode="reflect")
    h, w = image.shape
    return float(np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)) * np.abs(response[1:-1, 1:-1]).sum())


def gradient_magnitude(image):
    return np.hypot(ndimage.sobel(image, 0, mode="reflect"), ndimage.sobel(image, 1, mode="reflect"))


def digest(result: LFIRawData):
    h = hashlib.sha256()
    h.update(result.amplitude.tobytes())
    h.update(result.phase.tobytes())
    for key in sorted(result.metadata.get("spatial_products", {})):
        h.update(key.encode())
        h.update(np.ascontiguousarray(result.metadata["spatial_products"][key]).tobytes())
    return h.hexdigest()


def load_cases():
    """Real cases for E1/E2/E7. Leaf uses one full-resolution matched frame."""
    cases = []
    t0 = time.perf_counter()
    skin = load_skin_hdf5(SKIN_H5)
    cases.append(("Skin frame 40", skin, None, time.perf_counter() - t0))
    for i in [21, 130]:
        t0 = time.perf_counter()
        veg = load_rdm_hdf5_frame(VEG_H5 % i)
        cases.append(("Vegetables frame %d" % i, veg, None, time.perf_counter() - t0))
    t0 = time.perf_counter()
    leaf = load_leaf_mat(LEAF_MAT, z_slice=slice(45, 46))
    ref = loadmat(LEAF_MAT, variable_names=["L_P_U"])["L_P_U"][:, :, 45]
    leaf.metadata["phase_reference"] = ref[None]
    leaf.metadata["phase_reference_source"] = {"file": str(LEAF_MAT), "variable": "L_P_U",
                                               "stored_reference_index": 45}
    cases.append(("Leaf frame 45", leaf, {"phase_unwrapping": {"method": "validated_reference"}},
                  time.perf_counter() - t0))
    return cases


def vegetable_frame_ids():
    ids = sorted(int(p.stem.split("-")[1]) for p in (ROOT / "data/vegetables").glob("frame-*.hdf5"))
    return [i for i in ids if i != 20]


# ---------------------------------------------------------------- E1, E2

def e1_e2(cases):
    rows = []
    for name, data, overrides, load_s in cases:
        run_pipeline = lambda: api.run_pipeline(data, config_overrides=overrides)
        run_pipeline()  # warm-up so imports are excluded from timing
        tracemalloc.start()
        t0 = time.perf_counter()
        first = run_pipeline()
        seconds = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        second = run_pipeline()
        rows.append({
            "case": name, "shape": "x".join(map(str, data.shape)),
            "pixels": int(np.prod(data.shape)),
            "load_seconds": round(load_s, 3), "pipeline_seconds": round(seconds, 3),
            "peak_traced_memory_mb": round(peak / 2**20, 1),
            "input_mb": round((data.amplitude.nbytes + data.phase.nbytes) / 2**20, 1),
            "byte_identical_repeat": digest(first) == digest(second),
            "sha256_prefix": digest(first)[:16],
            "phase_status": first.metadata["phase_unwrapping_status"]["status"],
        })
    write_csv("e1_e2_repeatability_runtime.csv", rows)

    manifest = json.loads((SPATIAL / "manifest.json").read_text())
    leaf_times = [c["pipeline_seconds"] for c in manifest["cases"] if c["name"].startswith("leaf")]
    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    ax.barh([r["case"] for r in rows], [r["pipeline_seconds"] for r in rows], color="#4C72B0")
    for i, r in enumerate(rows):
        ax.text(r["pipeline_seconds"], i, "  %.2f s, %.0f MB" % (r["pipeline_seconds"], r["peak_traced_memory_mb"]),
                va="center", fontsize=9)
    ax.axvline(30, color="grey", ls="--", lw=1)
    ax.text(30, len(rows) - 0.5, " 30 s target", color="grey", fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("Pipeline time per image (s, log scale)")
    ax.set_title("E2 pipeline cost per full-resolution image")
    fig.savefig(FIGURES / "e2_runtime.png", dpi=180)
    plt.close(fig)
    return {"cases": rows, "all_byte_identical": all(r["byte_identical_repeat"] for r in rows),
            "leaf_91_batch_pipeline_seconds": {"median": float(np.median(leaf_times)),
                                               "max": float(np.max(leaf_times)),
                                               "total": float(np.sum(leaf_times))}}


# ---------------------------------------------------------------- E3

def e3_leaf_reference():
    rows = []
    for i in range(91):
        meta = json.loads((SPATIAL / ("leaf_frame%03d_spatial_metadata.json" % i)).read_text())
        m = meta["reference_comparison"]
        rows.append({"stored_index": i,
                     "background_only_max_error_rad": m["background_only_reference_max_error_radians"],
                     "background_only_rmse_rad": m["background_only_reference_rmse_radians"],
                     "line_corrected_vs_um_corr": m["relative_phase_correlation_with_saved_u_m"],
                     "line_corrected_vs_um_rmse_rad": m["relative_phase_rmse_with_saved_u_m"]})
    write_csv("e3_leaf_reference_agreement.csv", rows)
    idx = [r["stored_index"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].semilogy(idx, [r["background_only_max_error_rad"] for r in rows], ".", color="#4C72B0")
    axes[0].set_title("Background-only output vs saved U_M (max |error|)")
    axes[0].set_xlabel("Stored Leaf index")
    axes[0].set_ylabel("radians")
    axes[1].plot(idx, [r["line_corrected_vs_um_rmse_rad"] for r in rows], ".", color="#C44E52")
    axes[1].set_title("Additional line correction: deviation from U_M")
    axes[1].set_xlabel("Stored Leaf index")
    axes[1].set_ylabel("RMSE (radians)")
    fig.suptitle("E3 agreement with historical Leaf references (not ground truth)")
    fig.savefig(FIGURES / "e3_leaf_reference.png", dpi=180)
    plt.close(fig)
    col = lambda k: np.array([r[k] for r in rows])
    return {"frames": len(rows),
            "background_only_max_error_rad": float(col("background_only_max_error_rad").max()),
            "line_corrected_vs_um_corr_median": float(np.median(col("line_corrected_vs_um_corr"))),
            "line_corrected_vs_um_corr_min": float(col("line_corrected_vs_um_corr").min()),
            "line_corrected_vs_um_rmse_median": float(np.median(col("line_corrected_vs_um_rmse_rad")))}


# ---------------------------------------------------------------- E4, E5

def make_injection(base, seed=0):
    """Add known background, row offsets, periodic fringe and noise to ``base``."""
    rng = np.random.default_rng(seed)
    h, w = base.shape
    yy, xx = np.mgrid[0:h, 0:w] / np.array([h, w])[:, None, None]
    scale = base.std()
    background = scale * (3.0 * (xx - 0.5) + 2.0 * (yy - 0.5) ** 2
                          + 2.5 * np.exp(-((xx - 0.3) ** 2 + (yy - 0.6) ** 2) / 0.05))
    rows = scale * 0.6 * rng.standard_normal(h)
    stripe = np.repeat(rows[:, None], w, axis=1)
    f_row, f_col = 0.0, 0.125  # cycles/pixel, a column-direction fringe
    fringe = scale * 0.5 * np.cos(2 * np.pi * (f_row * np.arange(h)[:, None] + f_col * np.arange(w)[None, :]))
    noise = scale * 0.1 * rng.standard_normal(base.shape)
    return base + background + stripe + fringe + noise, {"fringe": [f_row, f_col],
                                                         "components": (background, stripe, fringe, noise)}


def injection_metrics(base, output, fringe_freq):
    b = base - base.mean()
    o = output - output.mean()
    spectrum = np.abs(np.fft.fft2(o)) ** 2
    fy = int(round(fringe_freq[0] * o.shape[0])) % o.shape[0]
    fx = int(round(fringe_freq[1] * o.shape[1])) % o.shape[1]
    return {"rmse_to_truth_rel": rmse(o, b) / b.std(),
            "corr_to_truth": corr(o, b),
            "fine_detail_corr": corr(highpass(o), highpass(b)),
            "stripe_std_rel": line_offset_std(o) / b.std(),
            "fringe_power_fraction": float(spectrum[fy, fx] / spectrum.sum())}


ABLATION = [
    ("injected input (no correction)", None),
    ("background only", {"stripe_axis": "none"}),
    ("background + line", {}),
    ("background + line + notch", {"fft_notches": [[0.0, 0.125]],
                                   "notch_width_cycles_per_pixel": 0.004}),
]


def e4_e5(leaf_um, leaf_u):
    # Base truth: a saved, already background-corrected Leaf image (realistic texture).
    base = leaf_um[:, :, 45].astype(np.float64)
    injected, info = make_injection(base)
    rows, panels = [], [("Truth: saved U_M[45]", base), ("Injected input", injected)]
    for label, override in ABLATION:
        if override is None:
            out = injected
        else:
            out = correct_image(injected, {**LEAF_PARAMS, **override})["corrected"]
            panels.append((label, out))
        rows.append({"configuration": label, **{k: round(v, 4) for k, v in
                                                injection_metrics(base, out, info["fringe"]).items()}})
    # Control: the uncorrupted truth through the full chain isolates signal loss.
    control = correct_image(base, {**LEAF_PARAMS, **ABLATION[-1][1]})["corrected"]
    rows.append({"configuration": "control: truth through full chain", **{k: round(v, 4) for k, v in
                                                                         injection_metrics(base, control, info["fringe"]).items()}})
    rows.append({"configuration": "truth (reference)", **{k: round(v, 4) for k, v in
                                                         injection_metrics(base, base, info["fringe"]).items()}})
    write_csv("e4_injection_ablation.csv", rows)

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 6), constrained_layout=True)
    lim = np.percentile(np.abs(base - base.mean()), 99)
    for ax, (title, img) in zip(axes, panels):
        shown = img - img.mean()
        if title == "Injected input":
            vmax = np.percentile(np.abs(shown), 99)
        else:
            vmax = lim
        ax.imshow(shown, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("E4 controlled artefact injection on a real Leaf texture (known truth)")
    fig.savefig(FIGURES / "e4_injection_ablation.png", dpi=150)
    plt.close(fig)

    # E5 sensitivity to background scale.
    sigmas = [5, 10, 15, 20, 25, 32, 50, 80]
    leaf_frames = list(range(0, 91, 9))
    sweep = []
    for s in sigmas:
        params = {**LEAF_PARAMS, "background_sigma_pixels": float(s)}
        synth = correct_image(injected, params)["corrected"]
        um_rmse = []
        for i in leaf_frames:
            u = leaf_u[:, :, i]
            bg = ndimage.gaussian_filter(u, s, mode="nearest", truncate=2.0)
            um_rmse.append(rmse(u - bg, leaf_um[:, :, i]))
        m = injection_metrics(base, synth, info["fringe"])
        sweep.append({"background_sigma_pixels": s,
                      "synthetic_rmse_to_truth_rel": round(m["rmse_to_truth_rel"], 4),
                      "synthetic_corr_to_truth": round(m["corr_to_truth"], 4),
                      "leaf_um_rmse_rad_median": float(np.median(um_rmse))})
    write_csv("e5_background_sigma_sweep.csv", sweep)
    fig, ax1 = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    ax1.plot(sigmas, [r["synthetic_rmse_to_truth_rel"] for r in sweep], "o-", color="#4C72B0",
             label="Synthetic: RMSE to truth / std(truth)")
    ax1.set_xlabel("Background Gaussian sigma (pixels)")
    ax1.set_ylabel("Synthetic relative RMSE", color="#4C72B0")
    ax2 = ax1.twinx()
    ax2.semilogy(sigmas, [max(r["leaf_um_rmse_rad_median"], 1e-15) for r in sweep], "s--", color="#C44E52",
                 label="Leaf: RMSE to saved U_M")
    ax2.set_ylabel("Leaf U_M RMSE (rad, log)", color="#C44E52")
    ax1.set_title("E5 sensitivity to background scale (Leaf frames 0, 9, ..., 90)")
    fig.savefig(FIGURES / "e5_sigma_sweep.png", dpi=180)
    plt.close(fig)
    return {"injection": rows, "sigma_sweep": sweep, "base_image": "saved L_P_U_M index 45",
            "leaf_frames_in_sweep": leaf_frames}


# ---------------------------------------------------------------- E6

def e6_real_artefacts():
    """What the spatial step changed on every real image (no ground truth)."""
    rows = []
    names = ["skin_frame40_spatial"] + ["vegetable_frame%d_spatial" % i for i in vegetable_frame_ids()] + \
            ["leaf_frame%03d_spatial" % i for i in range(91)]
    for name in names:
        d = np.load(SPATIAL / (name + ".npz"))
        mask = d["signal_mask"][0]
        valid = None if mask.all() else mask
        before, after, detail = d["amplitude_pre_spatial"][0], d["amplitude_corrected"][0], d["amplitude_detail"][0]
        rows.append({
            "image": name.replace("_spatial", ""), "channel": "amplitude",
            "line_offset_std_ratio": line_offset_std(after) / max(line_offset_std(before), 1e-12),
            "low_frequency_ratio_corrected": background_std(after) / background_std(before),
            "low_frequency_ratio_relative_view": background_std(np.nan_to_num(detail), valid=valid)
                                                 / background_std(before, valid=valid),
            "fine_detail_corr": corr(highpass(before), highpass(after)),
        })
        if bool(d["phase_correction_available"][0]):
            unwrapped, relative = d["phase_pre_spatial"][0], d["phase_corrected"][0]
            ok = np.isfinite(unwrapped) & np.isfinite(relative)
            before_p, after_p = np.where(ok, unwrapped, 0.0), np.where(ok, relative, 0.0)
            pv = None if ok.all() else ok
            rows.append({
                "image": name.replace("_spatial", ""), "channel": "phase",
                "line_offset_std_ratio": line_offset_std(relative, pv) / max(line_offset_std(unwrapped, pv), 1e-12),
                "low_frequency_ratio_corrected": float("nan"),
                "low_frequency_ratio_relative_view": background_std(after_p, valid=pv) / background_std(before_p, valid=pv),
                # Compare away from the mask edge, where zero filling would dominate a high-pass.
                "fine_detail_corr": (lambda inner: corr(highpass(before_p)[inner], highpass(after_p)[inner]))(
                    ndimage.binary_erosion(ok, iterations=8)),
            })
    write_csv("e6_real_indicators.csv", rows)

    groups = {}
    for r in rows:
        key = ("Leaf" if r["image"].startswith("leaf") else r["image"].split("_")[0].capitalize()) + " " + r["channel"]
        groups.setdefault(key, []).append(r)
    summary = []
    for key, items in groups.items():
        med = lambda k: float(np.nanmedian([i[k] for i in items])) if not all(np.isnan(i[k]) for i in items) else float("nan")
        summary.append({"group": key, "n_images": len(items),
                        "line_offset_std_ratio_median": med("line_offset_std_ratio"),
                        "low_frequency_ratio_corrected_median": med("low_frequency_ratio_corrected"),
                        "low_frequency_ratio_relative_view_median": med("low_frequency_ratio_relative_view"),
                        "fine_detail_corr_median": med("fine_detail_corr"),
                        "fine_detail_corr_min": float(min(i["fine_detail_corr"] for i in items))})
    write_csv("e6_real_summary.csv", summary)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    x = np.arange(len(summary))
    labels = [g["group"] for g in summary]
    axes[0].bar(x, [g["line_offset_std_ratio_median"] for g in summary], 0.6, color="#51247A")
    axes[0].set_title("Scan-line offsets after / before", fontsize=10)
    amp = [g for g in summary if g["group"].endswith("amplitude")]
    xa = np.arange(len(amp))
    axes[1].bar(xa - 0.2, [g["low_frequency_ratio_corrected_median"] for g in amp], 0.4,
                color="#51247A", label="amplitude_corrected (kept)")
    axes[1].bar(xa + 0.2, [g["low_frequency_ratio_relative_view_median"] for g in amp], 0.4,
                color="#FBB800", label="amplitude_detail (removed)")
    axes[1].set_xticks(xa, [g["group"] for g in amp], rotation=20, fontsize=8)
    axes[1].set_title("Low-frequency content after / before (sigma=50 px)", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[2].bar(x, [g["fine_detail_corr_median"] for g in summary], color="#55A868")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Fine-detail correlation (high-pass, sigma=2)", fontsize=10)
    for ax in (axes[0], axes[2]):
        ax.set_xticks(x, labels, rotation=20, fontsize=8)
    for ax in axes[:2]:
        ax.axhline(1, color="grey", ls="--", lw=1)
    fig.suptitle("E6 real-data indicators (medians; no ground truth)")
    fig.savefig(FIGURES / "e6_real_indicators.png", dpi=180)
    plt.close(fig)
    return summary


# ---------------------------------------------------------------- E7

def e7_noise_reduction(cases):
    from thz_preprocessing import noise_reduction
    rows = []
    configs = [("none", {"method": "none"}), ("median 3x3", {"method": "median2d", "median_window": 3}),
               ("median 5x5", {"method": "median2d", "median_window": 5}),
               ("gaussian s=1", {"method": "gaussian2d", "gaussian_sigma": 1.0}),
               ("gaussian s=2", {"method": "gaussian2d", "gaussian_sigma": 2.0})]
    for name, data, _, _ in cases:
        raw = data.amplitude[0].astype(np.float64)
        g_raw = gradient_magnitude(raw)
        n_raw = noise_sigma(raw)
        for label, params in configs:
            out = noise_reduction.apply(data, params).amplitude[0].astype(np.float64)
            rows.append({"case": name, "filter": label,
                         "hf_noise_estimate_rel": round(noise_sigma(out) / n_raw, 4),
                         "edge_corr": round(corr(gradient_magnitude(out), g_raw), 4),
                         "mean_abs_change_rel": round(float(np.mean(np.abs(out - raw)) / raw.std()), 4)})
    write_csv("e7_noise_reduction.csv", rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    markers = dict(zip([c[0] for c in cases], "osD^"))
    colours = dict(zip([c[0] for c in configs], ["#8C8C8C", "#4C72B0", "#1F3B73", "#DD8452", "#8C3B0F"]))
    for r in rows:
        ax.scatter(r["hf_noise_estimate_rel"], r["edge_corr"], marker=markers[r["case"]],
                   color=colours[r["filter"]], s=40)
    for f, c in colours.items():
        ax.scatter([], [], color=c, label=f)
    for k, m in markers.items():
        ax.scatter([], [], marker=m, color="k", facecolors="none", label=k)
    ax.set_xlabel("Immerkaer high-frequency estimate after / before (log)")
    ax.set_xscale("log")
    ax.set_ylabel("Gradient-magnitude correlation with raw")
    ax.set_title("E7 amplitude noise reduction trade-off")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIGURES / "e7_noise_tradeoff.png", dpi=180)
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- E8

def _quality(name):
    meta = json.loads((SPATIAL / (name + "_metadata.json")).read_text())
    status = meta["metadata"]["phase_unwrapping_status"]
    return status["status"], status["quality_diagnostics"][0]


def e8_residues():
    """Residue density of the wrapped phase, whole image and inside the signal mask."""
    rows = []
    status, q = _quality("skin_frame40_spatial")
    rows.append({"image": "Skin 40", "whole_image": q["residue_fraction_whole_image"],
                 "signal_region": q["residue_fraction"], "signal_fraction": q["signal_fraction"], "status": status})
    veg_ids = vegetable_frame_ids()
    for i in veg_ids:
        status, q = _quality("vegetable_frame%d_spatial" % i)
        rows.append({"image": "Vegetables %d" % i, "whole_image": q["residue_fraction_whole_image"],
                     "signal_region": q["residue_fraction"], "signal_fraction": q["signal_fraction"], "status": status})
    for i in range(91):
        status, q = _quality("leaf_frame%03d_spatial" % i)
        rows.append({"image": "Leaf %d" % i, "whole_image": q["residue_fraction"],
                     "signal_region": float("nan"), "signal_fraction": 1.0, "status": status})
    write_csv("e8_phase_residues.csv", rows)
    leaf = [r["whole_image"] for r in rows if r["image"].startswith("Leaf")]
    veg = [r for r in rows if r["image"].startswith("Vegetables")]
    skin = rows[0]
    gate = 0.001
    fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
    ax.plot(range(91), leaf, ".", color="#55A868", label="Leaf, whole image (stored index)")
    ax.plot(veg_ids, [r["whole_image"] for r in veg], "s", color="#C44E52", ms=4, label="Vegetables, whole image (frame id)")
    ax.plot(veg_ids, [r["signal_region"] for r in veg], "s", mfc="none", color="#C44E52", ms=5,
            label="Vegetables, signal region")
    ax.plot([40], [skin["whole_image"]], "D", color="#51247A", ms=6, label="Skin, whole image")
    ax.plot([40], [skin["signal_region"]], "D", mfc="none", color="#51247A", ms=7, label="Skin, signal region")
    ax.axhline(gate, color="k", ls="--", lw=1, label="release gate 0.001")
    ax.set_yscale("log")
    ax.set_xlabel("Leaf stored index / Vegetables or Skin frame id")
    ax.set_ylabel("Fraction of 2x2 cells with a residue")
    ax.set_title("E8 wrapped-phase residue density")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(FIGURES / "e8_phase_residues.png", dpi=180)
    plt.close(fig)
    col = lambda k: [r[k] for r in veg]
    return {"gate": gate, "skin": skin, "vegetable_frames": veg_ids,
            "vegetable_whole_min": float(min(col("whole_image"))), "vegetable_whole_max": float(max(col("whole_image"))),
            "vegetable_signal_min": float(min(col("signal_region"))), "vegetable_signal_max": float(max(col("signal_region"))),
            "vegetable_signal_median": float(np.median(col("signal_region"))),
            "vegetable_released": int(sum(r["status"] == "unwrapped" for r in veg)),
            "leaf_median": float(np.median(leaf)), "leaf_min": float(min(leaf)), "leaf_max": float(max(leaf))}


# ---------------------------------------------------------------- E9

def e9_depth_series():
    """Multi-frame Vegetables depth series: volume cost and depth continuity."""
    from thz_preprocessing import load_rdm_hdf5_volume
    ids = vegetable_frame_ids()
    run = [i for i in ids if 21 <= i <= 30]
    paths = [VEG_H5 % i for i in run]
    t0 = time.perf_counter()
    volume = load_rdm_hdf5_volume(paths, target_type="Vegetable",
                                  experiment_details_path=ROOT / "data/vegetables/experiment_details.txt",
                                  z_reference_frame_id=20)
    load_s = time.perf_counter() - t0
    api.run_pipeline(volume)
    tracemalloc.start()
    t0 = time.perf_counter()
    first = api.run_pipeline(volume)
    seconds = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    identical = digest(first) == digest(api.run_pipeline(volume))
    # Processing a frame alone must equal processing it inside the volume.
    single = api.run_pipeline(load_rdm_hdf5_frame(paths[3]))
    independent = bool(np.array_equal(single.metadata["spatial_products"]["amplitude_corrected"][0],
                                      first.metadata["spatial_products"]["amplitude_corrected"][3]))
    products = first.metadata["spatial_products"]
    adjacent = [{"frame_pair": "%d-%d" % (run[k], run[k + 1]),
                 "raw_amplitude_corr": corr(volume.amplitude[k], volume.amplitude[k + 1]),
                 "corrected_amplitude_corr": corr(products["amplitude_corrected"][k],
                                                  products["amplitude_corrected"][k + 1])}
                for k in range(len(run) - 1)]
    write_csv("e9_adjacent_frame_correlation.csv", adjacent)

    per_frame = []
    step_mm, z0 = 0.1, 30.05  # experiment_details.txt; frame 20 is the scan start
    for i in ids:
        d = np.load(SPATIAL / ("vegetable_frame%d_spatial.npz" % i))
        _, q = _quality("vegetable_frame%d_spatial" % i)
        raw = d["raw_amplitude"][0]
        smooth = ndimage.gaussian_filter(raw, 2)
        per_frame.append({"frame": i, "z_mm": round(z0 + step_mm * (i - 20), 2),
                          "mean_amplitude": float(raw.mean()),
                          "pixel_noise_rel": float((raw - smooth).std() / raw.mean()),
                          "residue_whole_image": q["residue_fraction_whole_image"],
                          "residue_signal_region": q["residue_fraction"],
                          "signal_fraction": q["signal_fraction"],
                          "line_offset_std_rel": line_offset_std(d["amplitude_pre_spatial"][0]) / raw.mean()})
    write_csv("e9_vegetable_depth_series.csv", per_frame)

    z = [r["z_mm"] for r in per_frame]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    axes[0].plot(z, [r["mean_amplitude"] for r in per_frame], "o-", color="#51247A")
    axes[0].set_ylabel("Mean raw amplitude (codes)")
    axes[0].set_title("Signal level vs depth")
    axes[1].plot(z, [r["pixel_noise_rel"] for r in per_frame], "o-", color="#C44E52")
    axes[1].set_ylabel("High-frequency std / mean amplitude")
    axes[1].set_title("Relative pixel noise vs depth")
    axes[2].plot(z, [r["residue_whole_image"] for r in per_frame], "s-", color="#C44E52", label="whole image")
    axes[2].plot(z, [r["residue_signal_region"] for r in per_frame], "s-", mfc="none", color="#C44E52",
                 label="signal region")
    axes[2].axhline(0.001, color="k", ls="--", lw=1, label="gate 0.001")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Phase residue fraction vs depth")
    for ax in axes:
        ax.set_xlabel("z position (mm)")
    fig.suptitle("E9 Vegetables depth series (%d frames)" % len(ids))
    fig.savefig(FIGURES / "e9_depth_series.png", dpi=180)
    plt.close(fig)
    col = lambda k: [r[k] for r in adjacent]
    best = max(per_frame, key=lambda r: r["mean_amplitude"])
    return {"frames": ids, "volume_frames": run, "volume_shape": list(volume.shape),
            "volume_load_seconds": round(load_s, 2), "volume_pipeline_seconds": round(seconds, 2),
            "volume_peak_traced_memory_mb": round(peak / 2**20, 1),
            "volume_input_mb": round((volume.amplitude.nbytes + volume.phase.nbytes) / 2**20, 1),
            "volume_byte_identical_repeat": identical,
            "frame_result_independent_of_volume": independent,
            "volume_phase_status": first.metadata["phase_unwrapping_status"]["status"],
            "adjacent_raw_corr_median": float(np.median(col("raw_amplitude_corr"))),
            "adjacent_corrected_corr_median": float(np.median(col("corrected_amplitude_corr"))),
            "adjacent_corrected_corr_min": float(min(col("corrected_amplitude_corr"))),
            "strongest_frame": best["frame"], "strongest_frame_z_mm": best["z_mm"],
            "mean_amplitude_first_last_peak": [per_frame[0]["mean_amplitude"], per_frame[-1]["mean_amplitude"],
                                               best["mean_amplitude"]],
            "pixel_noise_rel_range": [min(r["pixel_noise_rel"] for r in per_frame),
                                      max(r["pixel_noise_rel"] for r in per_frame)]}


# ---------------------------------------------------------------- E10

def e10_skin_encoding():
    """Which fixed-point scale turns Skin phase wraps into 2*pi steps?"""
    import h5py

    with h5py.File(SKIN_H5, "r") as f:
        codes = f["frames/40/Image Data/QCL0/Left/IMAGE_PHASE"][()]
    signed = codes.view(np.int16)
    wrap = lambda x: np.angle(np.exp(1j * x))
    scalings = {"uint16 / 65536 * 2pi (old assumption)": wrap(codes / 65536.0 * 2 * np.pi),
                "int16 / 8192 * pi (signed-14)": wrap(signed / 8192.0 * np.pi)}
    bins = [1.0, 1.3, 1.8, 3.0, 5.5, 2 * np.pi + 1e-9]
    labels = ["1.0-1.3", "1.3-1.8 (pi/2)", "1.8-3.0", "3.0-5.5", "5.5-2pi (wrap)"]
    rows = []
    for name, phase in scalings.items():
        steps = np.abs(np.diff(phase, axis=1))
        counts, _ = np.histogram(steps[steps > 1.0], bins=bins)
        rows.append({"scaling": name, "phase_min": float(phase.min()), "phase_max": float(phase.max()),
                     "large_steps": int(counts.sum()), **{l: int(c) for l, c in zip(labels, counts)}})
    write_csv("e10_skin_encoding_steps.csv", rows)

    d = np.load(SPATIAL / "skin_frame40_spatial.npz")
    status, q = _quality("skin_frame40_spatial")
    fig, axes = plt.subplots(1, 4, figsize=(17, 2.9), constrained_layout=True)
    show(axes[0], d["raw_amplitude"][0], "Skin", "Amplitude")
    show(axes[1], d["raw_phase"][0], "Skin", "Wrapped phase, signed-14 scale", cmap="twilight", limits=(-np.pi, np.pi))
    show(axes[2], d["phase"][0], "Skin", "Unwrapped inside the signal region", cmap="magma", percentiles=(0.5, 99.5))
    show(axes[3], d["phase_corrected"][0], "Skin", "phase_corrected (relative)", percentiles=(5, 99))
    fig.suptitle("E10 Skin phase under the signed-14 scale (numerical evidence, not a supervisor confirmation)")
    fig.savefig(FIGURES / "e10_skin_phase.png", dpi=170)
    plt.close(fig)
    unwrapped = d["phase"][0]
    return {"code_range_as_int16": [int(signed.min()), int(signed.max())],
            "distinct_codes": int(np.unique(codes).size), "step_histogram": rows,
            "status": status, "residue_signal_region": q["residue_fraction"],
            "residue_whole_image": q["residue_fraction_whole_image"], "signal_fraction": q["signal_fraction"],
            "unwrapped_span_rad": float(np.nanmax(unwrapped) - np.nanmin(unwrapped))}


# ---------------------------------------------------------------- E11

def e11_independent_unwrap(leaf_u):
    """Does an independent 2D unwrap reproduce the group's saved Leaf result?"""
    from skimage.restoration import unwrap_phase

    def discontinuity(a):
        return float(((np.abs(np.diff(a, axis=0)) > np.pi).sum() + (np.abs(np.diff(a, axis=1)) > np.pi).sum())
                     / (2 * a.size))

    rows = []
    for i in range(0, 91, 9):
        d = np.load(SPATIAL / ("leaf_frame%03d_spatial.npz" % i))
        saved = leaf_u[:, :, i]
        own = unwrap_phase(d["raw_phase"][0], rng=0)
        cycles = np.rint((own - saved) / (2 * np.pi)).astype(int)
        _, counts = np.unique(cycles, return_counts=True)
        rows.append({"stored_index": i, "same_cycle_share": float(counts.max() / cycles.size),
                     "distinct_cycle_offsets": int(counts.size),
                     "saved_reference_discontinuity_fraction": discontinuity(saved),
                     "own_unwrap_discontinuity_fraction": discontinuity(own),
                     "residue_fraction": _wrapped_phase_quality(d["raw_phase"][0])["residue_fraction"]})
    write_csv("e11_independent_unwrap_vs_reference.csv", rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    ax.bar([r["stored_index"] for r in rows], [r["same_cycle_share"] for r in rows], 6, color="#51247A")
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Leaf stored index")
    ax.set_ylabel("Share of pixels on the same 2pi cycle")
    ax.set_title("E11 independent unwrap vs the group's saved U")
    fig.savefig(FIGURES / "e11_independent_unwrap.png", dpi=180)
    plt.close(fig)
    share = [r["same_cycle_share"] for r in rows]
    return {"frames": [r["stored_index"] for r in rows], "same_cycle_share_min": float(min(share)),
            "same_cycle_share_max": float(max(share)), "same_cycle_share_median": float(np.median(share)),
            "saved_discontinuity_range": [min(r["saved_reference_discontinuity_fraction"] for r in rows),
                                          max(r["saved_reference_discontinuity_fraction"] for r in rows)],
            "own_discontinuity_range": [min(r["own_unwrap_discontinuity_fraction"] for r in rows),
                                        max(r["own_unwrap_discontinuity_fraction"] for r in rows)]}


# ---------------------------------------------------------------- review figures

def review_figures():
    """Gallery of the three datasets, and what v0.5 did to an amplitude image."""
    cases = [("Leaf", "leaf_frame045_spatial", "Leaf, stored image 45"),
             ("Vegetable", "vegetable_frame130_spatial", "Vegetables, frame 130 (z = 41.05 mm)"),
             ("Skin", "skin_frame40_spatial", "Skin sample, frame 40")]
    fig, axes = plt.subplots(3, 3, figsize=(16, 10.2), constrained_layout=True)
    for row, (target, name, label) in zip(axes, cases):
        d = np.load(SPATIAL / (name + ".npz"))
        raw = d["raw_amplitude"][0]
        limits = np.percentile(raw, [1, 99])
        show(row[0], raw, target, label + ": raw amplitude", limits=limits)
        show(row[1], d["amplitude_corrected"][0], target, "amplitude_corrected (contrast kept)", limits=limits)
        show(row[2], d["amplitude_detail"][0], target, "amplitude_detail (relative view)", percentiles=(2, 98))
    fig.savefig(FIGURES / "gallery_three_datasets.png", dpi=150)
    plt.close(fig)

    old_path = PREVIOUS / "vegetable_frame130_spatial.npz"
    if not old_path.exists():
        return {"v05_comparison": "superseded batch not present"}
    old, new = np.load(old_path), np.load(SPATIAL / "vegetable_frame130_spatial.npz")
    raw = new["raw_amplitude"][0]
    limits = np.percentile(raw, [1, 99])
    fig, axes = plt.subplots(2, 3, figsize=(16, 6.6), constrained_layout=True)
    show(axes[0, 0], raw, "Vegetable", "Raw amplitude, frame 130", limits=limits)
    show(axes[0, 1], old["amplitude_corrected"][0], "Vegetable",
         "v0.5 amplitude_corrected: background sigma=32 removed", percentiles=(1, 99))
    show(axes[0, 2], new["amplitude_corrected"][0], "Vegetable", "v0.6 amplitude_corrected", limits=limits)
    span = np.percentile(np.abs(old["amplitude_stripe"][0]), 99.5)
    show(axes[1, 0], old["amplitude_background"][0], "Vegetable", "v0.5 removed 'background' (holds the objects)",
         limits=limits)
    show(axes[1, 1], old["amplitude_stripe"][0], "Vegetable", "v0.5 removed 'stripes' (real horizontal structure)",
         cmap="RdBu_r", limits=(-span, span))
    show(axes[1, 2], new["amplitude_stripe"][0], "Vegetable", "v0.6 removed line offsets (same colour scale)",
         cmap="RdBu_r", limits=(-span, span))
    fig.savefig(FIGURES / "review_v05_vs_v06_vegetable130.png", dpi=150)
    plt.close(fig)

    def object_contrast(image, mask):
        return float(np.nanmean(image[mask]) - np.nanmean(image[~mask]))

    mask = new["signal_mask"][0]
    return {"frame": 130,
            "contrast_raw": object_contrast(raw, mask),
            "contrast_v05": object_contrast(old["amplitude_corrected"][0], mask),
            "contrast_v06": object_contrast(new["amplitude_corrected"][0], mask),
            "v05_stripe_std": float(old["amplitude_stripe"][0].std()),
            "v06_stripe_std": float(new["amplitude_stripe"][0].std()),
            "raw_std": float(raw.std())}


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    print("Loading cases ...", flush=True)
    cases = load_cases()
    summary = {"version_evaluated": "0.6.0"}
    print("E1/E2 repeatability and runtime ...", flush=True)
    summary["e1_e2"] = e1_e2(cases)
    print("E3 Leaf reference agreement ...", flush=True)
    summary["e3"] = e3_leaf_reference()
    print("E4/E5 injection and sigma sweep ...", flush=True)
    refs = loadmat(LEAF_MAT, variable_names=["L_P_U", "L_P_U_M"])
    summary["e4_e5"] = e4_e5(refs["L_P_U_M"], refs["L_P_U"])
    print("E11 independent unwrap ...", flush=True)
    summary["e11"] = e11_independent_unwrap(refs["L_P_U"])
    del refs
    print("E6 real-data indicators ...", flush=True)
    summary["e6"] = e6_real_artefacts()
    print("E7 noise reduction ...", flush=True)
    summary["e7"] = e7_noise_reduction(cases)
    print("E8 phase residues ...", flush=True)
    summary["e8"] = e8_residues()
    print("E9 Vegetables depth series ...", flush=True)
    summary["e9"] = e9_depth_series()
    print("E10 Skin encoding ...", flush=True)
    summary["e10"] = e10_skin_encoding()
    print("Review figures ...", flush=True)
    summary["review"] = review_figures()
    summary["scope_note"] = ("Synthetic injection and saved historical references only; "
                             "no physical calibration or clinical ground truth.")
    (OUT / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: summary[k] for k in ["e3", "e8", "e9", "e10", "e11", "review"]}, indent=2, default=float))


if __name__ == "__main__":
    main()
