"""Run the Person 3 module over a whole Person 2 handoff and record the evidence.

Outputs (in ``out_dir``): handoff_report.json, report.html (one self-contained
page with every figure and table), the figures as PNG for the thesis, and a
depth-resampled amplitude volume. Lateral axes stay in pixels because the
handoff records no pixel size; z is in millimetres from the stacked file.
"""
import json
import time
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import median_filter
from .image_generator import ImageGenerator
from .handoff import (load_frame, load_volume, resample_depth, mirror_geometry,
                      aperture_aspect, FRAME_ASPECT)

REQUIRED = ("amplitude_corrected", "amplitude_detail", "raw_amplitude", "raw_phase", "phase_corrected",
            "signal_mask", "phase_correction_available", "z_indices", "amplitude_pre_spatial",
            "amplitude_stripe", "amplitude_notch_removed", "amplitude_background")


def stripe_index(image, window=15):
    """Line-offset strength: std of row medians about their running median, / 2-98% range.

    Rows are the scan lines of the 400x1300 rasters (Person 2 stripe_axis='rows').
    """
    a = np.asarray(image, dtype=float)
    rows = np.nanmedian(a, axis=1)
    residual = rows - median_filter(rows, size=window, mode="nearest")
    lo, hi = np.nanpercentile(a, [2, 98])
    return float(np.std(residual) / (hi - lo)) if hi > lo else 0.0


def audit_frames(data_dir):
    """Open every frame NPZ, check fields and Person 2's decomposition identities."""
    records = []
    for path in sorted(Path(data_dir).glob("*_spatial.npz")):
        with np.load(path, allow_pickle=False) as f:
            missing = [k for k in REQUIRED if k not in f.files]
            if missing:
                records.append({"file": path.name, "missing": missing})
                continue
            corrected, mask = f["amplitude_corrected"][0], f["signal_mask"][0]
            rebuilt = corrected + f["amplitude_stripe"][0] + f["amplitude_notch_removed"][0]
            detail_sum = f["amplitude_detail"][0] + f["amplitude_background"][0]
            available = bool(f["phase_correction_available"][0])
            phase = f["phase_corrected"][0]
            records.append({
                "file": path.name, "missing": [], "shape": list(corrected.shape),
                "frame_id": int(f["z_indices"][0]), "phase_available": available,
                "signal_fraction": float(mask.mean()),
                "phase_finite_fraction_in_mask": float(np.isfinite(phase[mask]).mean()) if available and mask.any() else 0.0,
                "phase_finite_outside_mask": int(np.isfinite(phase[~mask]).sum()) if available else 0,
                "amplitude_corrected_range": [float(np.nanmin(corrected)), float(np.nanmax(corrected))],
                "amplitude_corrected_negative_fraction": float((corrected < 0).mean()),
                "identity_pre_spatial_max_error": float(np.nanmax(np.abs(rebuilt - f["amplitude_pre_spatial"][0]))),
                "identity_detail_max_error_in_mask": float(np.nanmax(np.abs(detail_sum - corrected)[mask])) if mask.any() else 0.0,
                "stripe_index_raw": stripe_index(f["raw_amplitude"][0]),
                "stripe_index_corrected": stripe_index(corrected),
            })
    return records


def _gray(ax, image, aspect, title, *, pct=(1, 99)):
    lo, hi = np.nanpercentile(image, pct)
    ax.imshow(image, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=aspect, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (pixel)")
    ax.set_ylabel("y (pixel)")


def figure_amplitude_channels(root, out):
    """Same Vegetables slice: raw vs corrected vs detail amplitude."""
    frames = {ch: load_frame(root / "data/vegetable_frame130_spatial.npz", amplitude_channel=ch)
              for ch in ("raw_amplitude", "amplitude_corrected", "amplitude_detail")}
    fig = Figure(figsize=(8, 9.5), layout="constrained")
    axes = fig.subplots(3, 1)
    for ax, (ch, d) in zip(axes, frames.items()):
        _gray(ax, d.amplitude, d.metadata["pixel_aspect_yx"], f"Vegetables frame 130: {ch} (stripe index {stripe_index(d.amplitude):.4f})")
    fig.savefig(out / "handoff_veg130_amplitude_channels.png", dpi=150)


def figure_phase_cases(root, out):
    """Leaf (released phase), Skin (released phase) and Vegetables (wrapped) with ImageGenerator."""
    cases = [("leaf_frame045_spatial.npz", "sequential"), ("skin_frame40_spatial.npz", "sequential"),
             ("vegetable_frame130_spatial.npz", "cyclic")]
    timings = {}
    for name, mode in cases:
        d = load_frame(root / "data" / name)
        stem = "handoff_" + name.replace("_spatial.npz", "")
        start = time.perf_counter()
        gen = ImageGenerator(d)
        gen.generate_comparison(phase_mode=mode, export=out / f"{stem}_comparison.png")
        timings[stem] = time.perf_counter() - start
        if d.metadata["phase_state"] == "unwrapped":
            gen.generate_comparison("hsl_fusion", export=out / f"{stem}_fusion.png")
    return timings


def nifti_study(volume, grid, pixel_aspect_yx, nominal_pixel_mm, out):
    """Export the depth-resampled amplitude as NIfTI with a nominal lateral spacing and check it.

    The lateral spacing is an estimate (window diameter / its width in pixels), not a
    calibration; the header description and the JSON sidecar say so. The file is read
    back with SimpleITK, the ITK library that ITK-SNAP uses, when it is installed.
    """
    from .volume_reconstructor import write_nifti
    dx = float(nominal_pixel_mm)
    dy = dx * float(pixel_aspect_yx)
    dz = float(grid[1] - grid[0])
    path = write_nifti(out / "veg_amplitude_depth_nominal_mm.nii.gz", volume, spacing_xyz=(dx, dy, dz),
                       origin_xyz=(0.0, 0.0, float(grid[0])), spatial_unit="mm",
                       description="LFI amplitude_corrected; nominal lateral mm (not calibrated); z=focus",
                       sidecar={"channel": "amplitude_corrected", "z": "beam focus position (mm)",
                                "lateral_spacing": "nominal estimate from the ~5 cm window, not calibrated",
                                "nominal_pixel_mm_x": dx, "pixel_aspect_yx": float(pixel_aspect_yx)})
    expected = {"size_xyz": [volume.shape[2], volume.shape[1], volume.shape[0]],
                "spacing_xyz_mm": [dx, dy, dz], "origin_xyz_mm": [0.0, 0.0, float(grid[0])]}
    result = {"file": path.name, "expected": expected, "nominal_pixel_mm": dx,
              "note": "lateral spacing is a nominal estimate; z is the beam focus position"}
    try:
        import SimpleITK as sitk
    except ImportError:
        result["simpleitk"] = "not installed; check skipped"
        return result
    image = sitk.ReadImage(str(path))
    got = {"size_xyz": list(image.GetSize()), "spacing_xyz_mm": list(image.GetSpacing()),
           "origin_xyz_mm": list(image.GetOrigin()), "direction": list(image.GetDirection())}
    array = sitk.GetArrayViewFromImage(image)          # (z, y, x)
    k = volume.shape[0] // 2
    result["simpleitk"] = {
        **got, "version": sitk.Version_VersionString(),
        "size_matches": got["size_xyz"] == expected["size_xyz"],
        "spacing_matches": bool(np.allclose(got["spacing_xyz_mm"], expected["spacing_xyz_mm"], rtol=1e-5)),
        "origin_matches": bool(np.allclose(got["origin_xyz_mm"], expected["origin_xyz_mm"], atol=1e-5)),
        # NIfTI's RAS axes appear in ITK's LPS convention as diag(-1, -1, 1).
        "direction_axis_aligned": bool(np.allclose(np.abs(got["direction"]), np.eye(3).ravel())),
        "middle_slice_equal": bool(np.allclose(array[k], volume[k], equal_nan=True))}
    return result


def napari_screenshot(volume, grid, pixel_aspect_yx, out, nominal_pixel_mm=None):
    """3D rendering of the depth-resampled amplitude in napari, saved as PNG (optional).

    The window must be shown: on Windows a hidden napari canvas renders black. With
    a nominal pixel size all three axes are in mm, so the depth extent is to scale.
    """
    try:
        import napari
        from qtpy.QtWidgets import QApplication
    except ImportError:
        return {"napari": "not installed; skipped"}
    step = 4
    small = volume[:, ::step, ::step]
    dz = float(grid[1] - grid[0])
    if nominal_pixel_mm is None:
        scale, units = (dz, step * float(pixel_aspect_yx), float(step)), "z in mm; y, x in pixel widths"
    else:
        dx = float(nominal_pixel_mm)
        scale, units = (dz, step * dx * float(pixel_aspect_yx), step * dx), "z, y, x in mm (lateral nominal)"
    viewer = napari.Viewer(ndisplay=3, show=True)
    try:
        lo, hi = np.nanpercentile(small, [1, 99.5])
        viewer.add_image(small, name="amplitude_corrected", colormap="gray", contrast_limits=(lo, hi),
                         scale=scale, rendering="mip")
        viewer.reset_view()
        viewer.camera.angles = (-30, 40, 60)
        viewer.camera.zoom *= 1.1
        for _ in range(10):
            QApplication.processEvents()
        path = out / "handoff_napari_3d.png"
        viewer.screenshot(str(path), canvas_only=True, flash=False)
        return {"napari": napari.__version__, "file": path.name, "rendering": "maximum intensity projection",
                "downsampled_yx": step, "units": units}
    finally:
        viewer.close()


def depth_study(root, out, spacing=0.1, nominal_pixel_mm=None, napari=False):
    data, z, ids, mask = load_volume(root / "vegetables_volume_21frames.npz")
    start = time.perf_counter()
    volume, grid = resample_depth(data.amplitude, z, spacing)
    elapsed = time.perf_counter() - start
    # Per-slice signal strength along z (raw amplitude, inside and over the whole frame).
    raw, _, _, _ = load_volume(root / "vegetables_volume_21frames.npz", amplitude_channel="raw_amplitude")
    profile = [{"frame_id": int(i), "z_mm": float(zz), "raw_mean": float(raw.amplitude[k].mean()),
                "signal_fraction": float(mask[k].mean()), "stripe_index_corrected": stripe_index(data.amplitude[k])}
               for k, (i, zz) in enumerate(zip(ids, z))]
    np.savez_compressed(out / "veg_amplitude_depth_resampled.npz", amplitude=volume, z_mm=grid,
                        note="amplitude_corrected, linear in z only; x/y in pixels (no lateral calibration)")
    # Orthogonal sections: XY at the strongest frame, XZ/YZ through the frame centre.
    k_best = int(np.argmax([p["raw_mean"] for p in profile]))
    ny, nx = volume.shape[1:]
    jy, ix = ny // 2, nx // 2
    lo, hi = np.nanpercentile(volume, [1, 99])
    fig = Figure(figsize=(11, 8), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :])
    kz = int(np.argmin(np.abs(grid - z[k_best])))
    ax.imshow(volume[kz], cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect=data.metadata["pixel_aspect_yx"])
    ax.axhline(jy, color="tab:orange", lw=.8); ax.axvline(ix, color="tab:cyan", lw=.8)
    ax.set(title=f"XY at z = {grid[kz]:.2f} mm (frame {ids[k_best]})", xlabel="x (pixel)", ylabel="y (pixel)")
    for col, (section, label, color) in enumerate([(volume[:, jy, :], f"XZ at y = {jy} px", "tab:orange"),
                                                   (volume[:, :, ix], f"YZ at x = {ix} px", "tab:cyan")]):
        ax = fig.add_subplot(gs[1, col])
        n_lat = section.shape[1]
        ax.imshow(section, cmap="gray", vmin=lo, vmax=hi, origin="lower", aspect="auto",
                  extent=(-.5, n_lat - .5, grid[0] - spacing / 2, grid[-1] + spacing / 2))
        ax.set_title(label, color=color, fontsize=10)
        ax.set(xlabel=("x" if col == 0 else "y") + " (pixel)", ylabel="z (mm)")
    fig.savefig(out / "handoff_veg_depth_sections.png", dpi=150)
    fig = Figure(figsize=(7, 3.6), layout="constrained")
    ax = fig.subplots()
    ax.plot([p["z_mm"] for p in profile], [p["raw_mean"] for p in profile], "o-", ms=4)
    for p in profile:
        if p["frame_id"] in (21, 30, 100, 130, 140):
            ax.annotate(str(p["frame_id"]), (p["z_mm"], p["raw_mean"]), textcoords="offset points", xytext=(0, 6), fontsize=8, ha="center")
    ax.set(xlabel="z (mm, beam focus position; deeper to the right)", ylabel="mean raw amplitude (a.u.)",
           title="Vegetables stack: signal strength against focus position")
    ax.grid(alpha=.3)
    fig.savefig(out / "handoff_veg_depth_profile.png", dpi=150)
    result = {"input_shape_zyx": list(data.amplitude.shape), "z_mm_input": z.tolist(), "frame_ids": ids.tolist(),
              "output_spacing_mm": spacing, "output_shape_zyx": list(volume.shape),
              "output_z_range_mm": [float(grid[0]), float(grid[-1])], "resample_seconds": elapsed,
              "output_megabytes": volume.nbytes / 2**20, "strongest_frame": int(ids[k_best]),
              "profile": profile, "note": "amplitude-only; wrapped raw_phase is not interpolated"}
    if nominal_pixel_mm is not None:
        result["nifti"] = nifti_study(volume, grid, data.metadata["pixel_aspect_yx"], nominal_pixel_mm, out)
    if napari:
        result["napari"] = napari_screenshot(volume, grid, data.metadata["pixel_aspect_yx"], out, nominal_pixel_mm)
    return result


def geometry_study(root, out, raw_files=()):
    """Pixel shape from the scan-mirror settings; Skin aperture as an independent consistency check."""
    mirror = [mirror_geometry(p) for p in raw_files]
    skin = load_frame(root / "data/skin_frame40_spatial.npz", amplitude_channel="raw_amplitude")
    aperture, regions = aperture_aspect(skin.amplitude, details=True)
    aperture["assumed_frame_aspect"] = skin.metadata["frame_aspect_ratio"]
    fig = Figure(figsize=(9, 5.2), layout="constrained")
    ax = fig.subplots()
    lo, hi = np.percentile(skin.amplitude, [1, 99])
    ax.imshow(skin.amplitude, cmap="gray", vmin=lo, vmax=hi, aspect=skin.metadata["pixel_aspect_yx"])
    for (fraction, region), colour in zip(sorted(regions.items()), ("tab:cyan", "tab:orange", "tab:red")):
        ax.contour(region, levels=[.5], colors=[colour], linewidths=.9)
        ax.plot([], [], color=colour, label=f"threshold {fraction:g} x Otsu")
    ax.legend(loc="lower right", fontsize=8)
    ax.set(xlabel="x (pixel)", ylabel="y (pixel)",
           title=f"Skin aperture: implied frame ratio {aperture['frame_aspect_min']:.2f}-{aperture['frame_aspect_max']:.2f} "
                 f"(assumed {aperture['assumed_frame_aspect']:.3f})")
    fig.savefig(out / "handoff_geometry_skin_aperture.png", dpi=150)
    return {"mirror": mirror, "skin_aperture": aperture,
            "note": "mirror steps fix pixel shape only; their length at the sample is not recorded"}


def run(root, out_dir, raw_files=(), nominal_pixel_mm=None, napari=False):
    root, out = Path(root), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    records = audit_frames(root / "data")
    ok = [r for r in records if not r["missing"]]
    by_kind = {}
    for kind in FRAME_ASPECT:
        rs = [r for r in ok if r["file"].startswith(kind)]
        if not rs:
            continue
        by_kind[kind] = {
            "frames": len(rs), "shape": rs[0]["shape"],
            "phase_available": sum(r["phase_available"] for r in rs),
            "phase_available_frame_ids": [r["frame_id"] for r in rs if r["phase_available"]],
            "signal_fraction_range": [min(r["signal_fraction"] for r in rs), max(r["signal_fraction"] for r in rs)],
            "max_identity_error_pre_spatial": max(r["identity_pre_spatial_max_error"] for r in rs),
            "max_identity_error_detail": max(r["identity_detail_max_error_in_mask"] for r in rs),
            "phase_values_outside_mask": sum(r["phase_finite_outside_mask"] for r in rs),
            "amplitude_corrected_negative_fraction_max": max(r["amplitude_corrected_negative_fraction"] for r in rs),
            "stripe_index_raw_median": float(np.median([r["stripe_index_raw"] for r in rs])),
            "stripe_index_corrected_median": float(np.median([r["stripe_index_corrected"] for r in rs])),
            "stripe_reduced_frames": sum(r["stripe_index_corrected"] < r["stripe_index_raw"] for r in rs),
        }
    figure_amplitude_channels(root, out)
    render = figure_phase_cases(root, out)
    depth = depth_study(root, out, nominal_pixel_mm=nominal_pixel_mm, napari=napari)
    geometry = geometry_study(root, out, raw_files)
    report = {"handoff": str(root), "files_read": len(records), "files_with_missing_fields": [r for r in records if r["missing"]],
              "summary_by_sample": by_kind, "render_seconds": render, "depth_resampling": depth,
              "geometry": geometry,
              "elapsed_seconds": time.perf_counter() - started, "frames": records,
              "scope": "I/O, rendering and depth resampling on real pre-processed data; no physical calibration or ground truth"}
    (out / "handoff_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    from .report_html import write_html_report
    write_html_report(report, out)
    return report
