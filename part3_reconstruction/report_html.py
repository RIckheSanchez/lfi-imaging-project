"""Render a handoff report (the dict from handoff_report.run) as one self-contained HTML page.

Figures are embedded as base64 PNG (downscaled for size), so the page can be
emailed or opened offline. Numbers come from the report dict only.
"""
import base64
import html
import io
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FIGURES = {
    "channels": ("handoff_veg130_amplitude_channels.png",
                 "Vegetables frame 130: raw_amplitude, amplitude_corrected and amplitude_detail "
                 "(white = outside signal_mask). Grey scale, 1st-99th percentile, true pixel aspect."),
    "leaf": ("handoff_leaf_frame045_comparison.png",
             "Leaf image 45 (transposed): amplitude_corrected and released phase_corrected "
             "(relative, sequential colour map). Phase is missing along the veins."),
    "skin": ("handoff_skin_frame40_comparison.png",
             "Skin image (frame 40): amplitude_corrected and released relative phase."),
    "skin_fusion": ("handoff_skin_frame40_fusion.png",
                    "Skin HSL fusion, lightness limited to [0, 0.5] so bright regions keep their phase colour."),
    "veg_phase": ("handoff_vegetable_frame130_comparison.png",
                  "Vegetables frame 130: amplitude and wrapped raw_phase (cyclic colour map); "
                  "Vegetables phase is not released."),
    "aperture": ("handoff_geometry_skin_aperture.png",
                 "Skin raw amplitude shown with the assumed pixel aspect; contours: largest bright region "
                 "at three thresholds, whose outermost left/right points were fitted."),
    "sections": ("handoff_veg_depth_sections.png",
                 "Vegetables stack after depth-only resampling to a uniform z grid: XY plane at the "
                 "strongest slice and XZ/YZ sections. Lateral axes in pixels."),
    "profile": ("handoff_veg_depth_profile.png",
                "Mean raw amplitude of each Vegetables slice against its recorded scan position."),
}

CSS = """
body{font-family:Georgia,'Times New Roman',serif;max-width:1000px;margin:0 auto;padding:24px 16px 64px;
     color:#1d1d1f;background:#fff;line-height:1.5}
h1{font-size:1.7em;margin-bottom:.2em}h2{font-size:1.3em;margin-top:2em;border-bottom:1px solid #ccc;padding-bottom:.2em}
h3{font-size:1.05em;margin-top:1.4em}
.meta{color:#555;font-size:.9em}.scope{background:#f6f6f2;border-left:4px solid #b8860b;padding:8px 12px;font-size:.93em}
table{border-collapse:collapse;margin:.8em 0;font-size:.88em;width:100%}
th,td{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#f2f2f2}td.n{text-align:right;font-variant-numeric:tabular-nums}
figure{margin:1.2em 0}figure img{max-width:100%;height:auto;border:1px solid #e3e3e3}
figcaption{font-size:.88em;color:#444;margin-top:.3em}
code{font-family:Consolas,monospace;font-size:.9em;background:#f4f4f4;padding:0 3px}
.ok{color:#1b7f3b;font-weight:bold}.warn{color:#b35c00;font-weight:bold}
details summary{cursor:pointer;margin:.6em 0}
@media print{body{max-width:none}details{display:block}figure{break-inside:avoid}}
"""


def _e(value):
    return html.escape(str(value))


def _table(headers, rows, numeric=()):
    """Headers and cells are trusted HTML built in this module; file names are escaped by callers."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f'<td class="n">{c}</td>' if i in numeric else f"<td>{c}</td>"
                                    for i, c in enumerate(row)) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _figure(out, key, number, max_width=1400):
    name, caption = FIGURES[key]
    path = Path(out) / name
    if not path.exists():
        return f'<p class="warn">Figure {number} missing: {_e(name)}</p>'
    from PIL import Image
    with Image.open(path) as im:
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        buffer = io.BytesIO()
        im.save(buffer, format="PNG", optimize=True)
    data = base64.b64encode(buffer.getvalue()).decode()
    return (f'<figure><img alt="{_e(caption)}" src="data:image/png;base64,{data}">'
            f"<figcaption><b>Figure {number}.</b> {_e(caption)} <code>{_e(name)}</code></figcaption></figure>")


def _provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent, timeout=10).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    versions = []
    for module in ("numpy", "scipy", "matplotlib", "skimage", "h5py"):
        try:
            versions.append(f"{module} {__import__(module).__version__}")
        except ImportError:
            pass
    return commit, f"Python {platform.python_version()}; " + ", ".join(versions)


def write_html_report(report, out_dir, filename="report.html"):
    out = Path(out_dir)
    s, depth, geo = report["summary_by_sample"], report["depth_resampling"], report.get("geometry", {})
    commit, versions = _provenance()
    total = sum(v["frames"] for v in s.values())
    released = sum(v["phase_available"] for v in s.values())
    max_identity = max(max(v["max_identity_error_pre_spatial"], v["max_identity_error_detail"]) for v in s.values())
    outside = sum(v["phase_values_outside_mask"] for v in s.values())
    parts = [f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Handoff check report</title>
<style>{CSS}</style></head><body>
<h1>Person 2 handoff check &mdash; Person 3 imaging module</h1>
<p class="meta">Generated {datetime.now():%Y-%m-%d %H:%M} &middot; handoff <code>{_e(report['handoff'])}</code><br>
Code commit <code>{_e(commit)}</code> &middot; {_e(versions)} &middot; run time {report['elapsed_seconds']:.0f} s<br>
Command: <code>{_e(' '.join(sys.argv) or 'python -m part3_reconstruction handoff')}</code></p>
<p class="scope"><b>Scope.</b> {_e(report['scope'])}. Lateral axes are in pixels because no pixel size is recorded.</p>"""]

    missing = report["files_with_missing_fields"]
    mirror = geo.get("mirror", [])
    aperture = geo.get("skin_aperture")
    findings = [
        f"<b>{report['files_read']}</b> files read; "
        + (f'<span class="ok">no missing fields</span>' if not missing else f'<span class="warn">{len(missing)} with missing fields</span>'),
        f"Decomposition identities hold to <b>{max_identity:.2g}</b> (floating-point rounding).",
        f"Released phase: <b>{released}</b> of {total} images; phase values outside <code>signal_mask</code>: <b>{outside}</b>.",
    ]
    veg = s.get("vegetable")
    if veg:
        drop = 1 - veg["stripe_index_corrected_median"] / veg["stripe_index_raw_median"]
        findings.append(f"<code>amplitude_corrected</code> lowers the Vegetables stripe index in "
                        f"<b>{veg['stripe_reduced_frames']} of {veg['frames']}</b> slices (median &minus;{drop:.0%}).")
    if mirror:
        m0 = mirror[0]
        findings.append(f"Mirror settings give a frame ratio of ({m0['imageColumns']}&times;{m0['columnStep']})/"
                        f"({m0['imageRows'] // 2}&times;{m0['rowStepSize']}) = <b>{m0['aspect_from_integer_steps']:.5f}</b>, "
                        + ("equal to the stored value in all files" if all(m["matches_recorded"] for m in mirror)
                           else '<span class="warn">not equal to the stored value in every file</span>') + ".")
    if aperture:
        findings.append(f"Skin aperture implies a frame ratio of <b>{aperture['frame_aspect_min']:.2f}&ndash;"
                        f"{aperture['frame_aspect_max']:.2f}</b> across {len(aperture['variants'])} fits "
                        f"(assumed {aperture['assumed_frame_aspect']:.3f}): "
                        + ('<span class="ok">consistent</span>' if aperture['frame_aspect_min'] <= aperture['assumed_frame_aspect']
                           <= aperture['frame_aspect_max'] else '<span class="warn">not confirmed</span>') + ".")
    findings.append(f"Vegetables stack resampled along z to {depth['output_spacing_mm']} mm: "
                    f"{' &times; '.join(map(str, depth['output_shape_zyx']))} voxels in {depth['resample_seconds']:.2f} s; "
                    f"strongest slice frame {depth['strongest_frame']}.")
    parts.append("<h2>1. Key results</h2><ul>" + "".join(f"<li>{x}</li>" for x in findings) + "</ul>")

    rows = [[_e(k.capitalize()), v["frames"], "&times;".join(map(str, v["shape"])), v["phase_available"],
             f"{v['signal_fraction_range'][0]:.3f}&ndash;{v['signal_fraction_range'][1]:.3f}",
             f"{max(v['max_identity_error_pre_spatial'], v['max_identity_error_detail']):.2g}",
             f"{v['stripe_index_raw_median']:.4f} &rarr; {v['stripe_index_corrected_median']:.4f}",
             f"{v['stripe_reduced_frames']}/{v['frames']}"] for k, v in s.items()]
    parts.append("<h2>2. File audit</h2>"
                 "<p>Every frame file was opened and checked for the expected fields, the signal fraction, "
                 "phase values outside the mask and Person 2's two decomposition identities. The stripe index is the "
                 "standard deviation of row medians about their 15-row running median, divided by the 2&ndash;98% range.</p>"
                 + _table(["Sample", "Images", "Shape", "Phase released", "Signal fraction", "Max identity error",
                           "Stripe index raw &rarr; corrected (median)", "Stripe lower"], rows, numeric=(1, 3)))
    if missing:
        parts.append("<p class=\"warn\">Files with missing fields:</p>" + _table(
            ["File", "Missing"], [[_e(m["file"]), _e(", ".join(m["missing"]))] for m in missing]))

    n = 1
    parts.append("<h2>3. Amplitude channels</h2>" + _figure(out, "channels", n)); n += 1
    parts.append("<h2>4. Phase and fusion displays</h2>")
    for key in ("leaf", "skin", "skin_fusion", "veg_phase"):
        parts.append(_figure(out, key, n)); n += 1
    render = report.get("render_seconds", {})
    if render:
        parts.append("<p>Rendering time per comparison figure, including PNG export: "
                     + ", ".join(f"{_e(k.replace('handoff_', ''))} {v:.2f} s" for k, v in render.items()) + ".</p>")

    parts.append("<h2>5. Pixel geometry</h2>")
    if mirror:
        parts.append("<h3>5.1 Scan-mirror settings (raw HDF5, <code>thz/info/mcp</code>)</h3>"
                     "<p>Adjacent columns are <code>columnStep</code> mirror steps apart and adjacent rows "
                     "<code>rowStepSize</code> steps apart; the recorded rows form a left and a right image. "
                     "The stored ratio ignores the fractional step settings, shown in the last ratio column.</p>"
                     + _table(["File", "Columns", "Rows (both sides)", "columnStep", "rowStepSize", "Fract. col/row",
                               "Stored ratio", "From steps", "With fractions", "Match", "Pixel dy/dx"],
                              [[_e(m["file"]), m["imageColumns"], m["imageRows"], m["columnStep"], m["rowStepSize"],
                                f"{m['columnStepFract256']}/256, {m['rowStepFract256']}/256",
                                f"{m['aspect_recorded']:.6f}", f"{m['aspect_from_integer_steps']:.6f}",
                                f"{m['aspect_with_fractional_steps']:.6f}",
                                '<span class="ok">yes</span>' if m["matches_recorded"] else '<span class="warn">no</span>',
                                f"{m['pixel_aspect_yx']:.4f}"] for m in mirror], numeric=(1, 2, 3, 4, 6, 7, 8, 10)))
    else:
        parts.append("<p class=\"warn\">No raw HDF5 files given (<code>--raw</code>); mirror settings not checked.</p>")
    if aperture:
        parts.append("<h3>5.2 Consistency check with the Skin aperture</h3>"
                     "<p>The Skin file stores no mirror settings; Person 2 assumed the 2026 raster. Its circular "
                     "aperture gives an independent estimate: for each threshold the largest bright region is taken, "
                     "and the outermost left and right boundary pixel of every row is fitted with an axis-aligned "
                     "ellipse. The pixel aspect that makes it round converts to a frame width/height ratio. Fits use "
                     "all rows or only the middle rows, because the aperture is cut off at the top and bottom. The "
                     "spread shows how strongly the image constrains the ratio. Selecting edge points near an assumed "
                     "circle was rejected because it biases the result towards round.</p>"
                     + _table(["Threshold (x Otsu)", "Rows", "Points", "Semi-axes x / y (px)", "Pixel dy/dx",
                               "Implied frame ratio"],
                              [[f"{v['threshold_fraction_of_otsu']:g}", _e(v["rows"]), v["points"],
                                " / ".join(f"{w:.1f}" for w in v["semi_axes_px_xy"]), f"{v['pixel_aspect_yx']:.3f}",
                                f"{v['frame_aspect']:.3f}"] for v in aperture["variants"]], numeric=(0, 2, 4, 5))
                     + f"<p>Implied frame ratio {aperture['frame_aspect_min']:.3f}&ndash;{aperture['frame_aspect_max']:.3f} "
                     f"(median {aperture['frame_aspect_median']:.3f}); assumed {aperture['assumed_frame_aspect']:.4f}.</p>"
                     + _figure(out, "aperture", n)); n += 1
    parts.append(f"<p><i>{_e(geo.get('note', ''))}</i></p>")

    parts.append("<h2>6. Depth resampling (Vegetables stack)</h2>"
                 f"<p>Input {' &times; '.join(map(str, depth['input_shape_zyx']))}, z = "
                 f"{depth['z_mm_input'][0]:.2f}&ndash;{depth['z_mm_input'][-1]:.2f} mm (non-uniform). Output "
                 f"{' &times; '.join(map(str, depth['output_shape_zyx']))} at {depth['output_spacing_mm']} mm, "
                 f"{depth['output_megabytes']:.0f} MB, {depth['resample_seconds']:.2f} s. {_e(depth['note'])}.</p>")
    for key in ("sections", "profile"):
        parts.append(_figure(out, key, n)); n += 1
    parts.append(_table(["Frame", "z (mm)", "Mean raw amplitude", "Signal fraction", "Stripe index (corrected)"],
                        [[p["frame_id"], f"{p['z_mm']:.2f}", f"{p['raw_mean']:.1f}", f"{p['signal_fraction']:.3f}",
                          f"{p['stripe_index_corrected']:.4f}"] for p in depth["profile"]], numeric=(0, 1, 2, 3, 4)))

    parts.append("<h2>7. Open questions</h2><ul>"
                 "<li>Pixel size in millimetres: the length of one mirror step at the sample is not recorded.</li>"
                 "<li>Phase scale of &pi;/8192 rad per code: supported by the code range, not yet confirmed.</li>"
                 "<li>Meaning of z: whether a scan position changes the imaged depth or mainly the focus.</li>"
                 "<li>Leaf: whether the 91 images are a time or a depth series.</li></ul>")

    frame_rows = [[_e(r["file"]), r.get("frame_id", ""), "yes" if r.get("phase_available") else "no",
                   f"{r['signal_fraction']:.3f}" if "signal_fraction" in r else "",
                   f"{r['identity_pre_spatial_max_error']:.1e}" if "identity_pre_spatial_max_error" in r else "",
                   f"{r['stripe_index_raw']:.4f}" if "stripe_index_raw" in r else "",
                   f"{r['stripe_index_corrected']:.4f}" if "stripe_index_corrected" in r else "",
                   _e(", ".join(r["missing"]))] for r in report["frames"]]
    parts.append(f"<h2>Appendix: all {len(frame_rows)} files</h2><details><summary>Show per-file table</summary>"
                 + _table(["File", "Frame", "Phase", "Signal fraction", "Identity error", "Stripe raw",
                           "Stripe corrected", "Missing"], frame_rows, numeric=(1, 3, 4, 5, 6)) + "</details>")
    parts.append("</body></html>")
    path = out / filename
    path.write_text("\n".join(parts), encoding="utf-8")
    return path
