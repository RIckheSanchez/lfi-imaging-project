"""Thesis figures computed with part3_reconstruction from the Person 2 handoff and synthetic data.

Writes norm_strategies_veg130.png, phase_strategies_leaf45.png, stripe_index_frames.png,
slice_placement.png and sphere_sections.png, and prints the numbers quoted in the thesis.
Run the handoff command first, since the stripe and depth figures read its report:

    python -m part3_reconstruction handoff <handoff_v061> --output outputs/handoff_v061
    python scripts/make_thesis_figures.py <handoff_v061> --output outputs/thesis
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from part3_reconstruction.colormap_engine import fold_phase, normalise
from part3_reconstruction.data import ImagingData
from part3_reconstruction.handoff import load_frame
from part3_reconstruction.validation import dice_score
from part3_reconstruction.volume_reconstructor import VolumeReconstructor

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("root", help="Person 2 handoff folder containing data/")
parser.add_argument("--report", default=str(REPO / "outputs/handoff_v061/handoff_report.json"),
                    help="handoff_report.json written by the handoff command")
parser.add_argument("--output", default=str(REPO / "outputs/thesis"))
args = parser.parse_args()
DATA = Path(args.root)
REPORT = json.loads(Path(args.report).read_text(encoding="utf-8"))
OUT = Path(args.output)
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight"})


def panel(ax, label):
    ax.set_title(label, loc="left", fontsize=11, fontweight="bold")


# ---- normalisation strategies on Vegetables frame 130
veg = load_frame(DATA / "data/vegetable_frame130_spatial.npz")
amp = veg.amplitude
aspect = veg.metadata["pixel_aspect_yx"]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), constrained_layout=True)
for ax, (lab, s) in zip(axes.flat[:3], [("(a) linear", "linear"), ("(b) percentile 2-98 %", "percentile"),
                                         ("(c) logarithmic", "log")]):
    im = ax.imshow(normalise(amp, s), cmap="gray", vmin=0, vmax=1, aspect=aspect, interpolation="nearest", origin="lower")
    panel(ax, lab)
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
fig.colorbar(im, ax=axes[1, 0], shrink=0.9, label="normalised")
ax = axes[1, 1]
v = amp[np.isfinite(amp)]
ax.hist(v, bins=200, color="0.4", log=True)
lo, hi = np.percentile(v, [2, 98])
ax.axvline(v.min(), color="tab:blue", ls="--", label="linear limits (min, max)")
ax.axvline(v.max(), color="tab:blue", ls="--")
ax.axvline(lo, color="tab:red", label="percentile limits (2nd, 98th)")
ax.axvline(hi, color="tab:red")
ax.set_xlabel("amplitude_corrected"); ax.set_ylabel("pixel count")
ax.legend(fontsize=8, loc="upper right")
panel(ax, "(d) histogram")
fig.savefig(OUT / "norm_strategies_veg130.png")
plt.close(fig)
stats = {"veg130_min": float(v.min()), "veg130_max": float(v.max()), "veg130_p2": float(lo),
         "veg130_p98": float(hi), "veg130_frac_above_p98": float((v > hi).mean())}

# ---- phase strategies on Leaf image 45
leaf = load_frame(DATA / "data/leaf_frame045_spatial.npz")
ph = leaf.phase if leaf.phase.shape[0] < leaf.phase.shape[1] else leaf.phase.T
leaf_aspect = (ph.shape[1] / ph.shape[0]) / 1.5
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), constrained_layout=True)
im0 = axes[0].imshow(normalise(ph, "linear"), cmap="viridis", vmin=0, vmax=1, interpolation="nearest", aspect=leaf_aspect, origin="lower")
panel(axes[0], "(a) sequential, continuous phase")
fig.colorbar(im0, ax=axes[0], shrink=0.8, label="normalised phase")
im1 = axes[1].imshow(fold_phase(ph), cmap="twilight", vmin=-np.pi, vmax=np.pi, interpolation="nearest", aspect=leaf_aspect, origin="lower")
panel(axes[1], "(b) cyclic, folded to [-π, π)")
cb = fig.colorbar(im1, ax=axes[1], shrink=0.8, label="phase (rad)")
cb.set_ticks([-np.pi, 0, np.pi]); cb.set_ticklabels(["-π", "0", "π"])
for ax in axes:
    ax.set_xlabel("x (pixel)"); ax.set_ylabel("y (pixel)")
fig.savefig(OUT / "phase_strategies_leaf45.png")
plt.close(fig)
pv = ph[np.isfinite(ph)]
stats.update(leaf45_phase_min=float(pv.min()), leaf45_phase_max=float(pv.max()),
             leaf45_phase_range_cycles=float((pv.max() - pv.min()) / (2 * np.pi)))

# ---- stripe index per frame
fr = REPORT["frames"]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
veg_fr = sorted([f for f in fr if f["file"].startswith("vegetable")], key=lambda f: f["frame_id"])
x = [f["frame_id"] for f in veg_fr]
axes[0].plot(x, [f["stripe_index_raw"] for f in veg_fr], "o-", ms=4, label="raw_amplitude")
axes[0].plot(x, [f["stripe_index_corrected"] for f in veg_fr], "s-", ms=4, label="amplitude_corrected")
axes[0].set_xlabel("Vegetables frame"); axes[0].set_ylabel("stripe index"); axes[0].legend(fontsize=8)
panel(axes[0], "(a) Vegetables")
leaf_fr = sorted([f for f in fr if f["file"].startswith("leaf")], key=lambda f: f["frame_id"])
x = [f["frame_id"] for f in leaf_fr]
axes[1].plot(x, [f["stripe_index_raw"] for f in leaf_fr], "o-", ms=3, label="raw_amplitude")
axes[1].plot(x, [f["stripe_index_corrected"] for f in leaf_fr], "s-", ms=3, label="amplitude_corrected")
axes[1].set_xlabel("Leaf image"); axes[1].set_ylabel("stripe index"); axes[1].legend(fontsize=8)
panel(axes[1], "(b) Leaf")
fig.tight_layout()
fig.savefig(OUT / "stripe_index_frames.png")
plt.close(fig)
rv = np.array([f["stripe_index_corrected"] / f["stripe_index_raw"] for f in veg_fr])
rl = np.array([f["stripe_index_corrected"] / f["stripe_index_raw"] for f in leaf_fr])
stats.update(veg_ratio_min=float(rv.min()), veg_ratio_max=float(rv.max()),
             leaf_ratio_min=float(rl.min()), leaf_ratio_max=float(rl.max()),
             leaf_sig_min=float(min(f["signal_fraction"] for f in leaf_fr)))

# ---- slice placement: index-based vs physical
dr = REPORT["depth_resampling"]
z = np.array(dr["z_mm_input"]); ids = np.array(dr["frame_ids"])
uniform = np.linspace(z[0], z[-1], len(z))
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.plot(np.arange(len(z)), z, "o-", label="focus position from the experiment note")
ax.plot(np.arange(len(z)), uniform, "s--", ms=4, color="tab:red", label="position implied by a constant zoom factor")
for i in (5, 10, 15):
    ax.annotate(f"{z[i] - uniform[i]:+.1f} mm", (i, (z[i] + uniform[i]) / 2), xytext=(6, 0),
                textcoords="offset points", fontsize=8, va="center")
    ax.plot([i, i], [z[i], uniform[i]], color="0.5", lw=0.8)
ax.set_xlabel("slice index in the stack"); ax.set_ylabel("depth z (mm)")
ax.legend(fontsize=8, loc="lower right")
ax2 = ax.secondary_xaxis("top", functions=(lambda i: i, lambda i: i))
ax2.set_xticks(np.arange(0, len(z), 4)); ax2.set_xticklabels([str(ids[i]) for i in range(0, len(z), 4)])
ax2.set_xlabel("frame number")
fig.savefig(OUT / "slice_placement.png")
plt.close(fig)
err = np.abs(z - uniform)
stats.update(zoom_err_max=float(err.max()), zoom_err_mean=float(err.mean()),
             zoom_wrong_by_more_than_0_5mm=int((err > 0.5).sum()), zoom_slices=len(z))

# ---- sphere test sections
zc = np.arange(-10, 11, 2)
zz, yy, xx = np.meshgrid(zc, np.arange(21) - 10, np.arange(21) - 10, indexing="ij")
truth_z, truth_y, truth_x = np.meshgrid(np.arange(-10, 11), np.arange(21) - 10, np.arange(21) - 10, indexing="ij")
truth = truth_x ** 2 + truth_y ** 2 + truth_z ** 2 <= 49


def recon(values):
    d = ImagingData(values.astype(float), np.zeros(values.shape), {"phase_unit": "rad", "phase_state": "unwrapped"})
    return VolumeReconstructor(d, spacing_xy=(1, 1), z_coordinates=zc).resample_isotropic(1).data.amplitude


binary = recon((xx ** 2 + yy ** 2 + zz ** 2 <= 49).astype(float)) >= 0.5
cont = recon(49 - xx ** 2 - yy ** 2 - zz ** 2) >= 0
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
src = (xx ** 2 + yy ** 2 + zz ** 2 <= 49)[:, 10, :]
axes[0].imshow(src, cmap="Greys", extent=(-10.5, 10.5, 11, -11), aspect="auto", interpolation="nearest")
axes[0].set_aspect(1)
for ax, (lab, m) in zip(axes[1:], [(f"(b) from binary slices, Dice {dice_score(binary, truth):.3f}", binary),
                                   (f"(c) from continuous field, Dice {dice_score(cont, truth):.3f}", cont)]):
    ax.imshow(m[:, 10, :], cmap="Greys", extent=(-10.5, 10.5, 10.5, -10.5), interpolation="nearest")
    panel(ax, lab)
t = np.linspace(0, 2 * np.pi, 200)
for ax in axes:
    ax.plot(7 * np.cos(t), 7 * np.sin(t), color="tab:red", lw=1.2)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("z (mm)")
panel(axes[0], "(a) input: 11 slices, 2 mm apart")
fig.tight_layout()
fig.savefig(OUT / "sphere_sections.png")
plt.close(fig)
stats.update(dice_binary=dice_score(binary, truth), dice_cont=dice_score(cont, truth))

print(json.dumps(stats, indent=1))
