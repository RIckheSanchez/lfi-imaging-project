"""Minimal reader for the THz LFI pre-processing handoff v0.6.

Needs numpy; matplotlib only for the figures.  Run from this folder:  python load_example.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# Physical width / height of a full frame. Leaf arrays are stored long-axis first.
FRAME_ASPECT = {"leaf": 1.5, "vegetable": 1.9651163, "skin": 1.9651163}


def load_frame(path):
    """Return a dict of arrays from one handoff NPZ. Image arrays are (n, H, W)."""
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def usable_phase(frame):
    """Relative phase in radians (NaN where there is no signal), or None if withheld."""
    if not bool(frame["phase_correction_available"].all()):
        return None
    return frame["phase_corrected"]


def for_display(image, kind):
    """Return (image, pixel_aspect) so that imshow shows the true shape of the sample."""
    view = image.T if kind == "leaf" else image
    rows, columns = view.shape
    return view, (columns / rows) / FRAME_ASPECT[kind]


def main():
    leaf = load_frame(HERE / "data/leaf_frame045_spatial.npz")
    veg = load_frame(HERE / "data/vegetable_frame130_spatial.npz")
    skin = load_frame(HERE / "data/skin_frame40_spatial.npz")
    print("channels:", sorted(veg))
    for name, frame in [("Leaf", leaf), ("Vegetables", veg), ("Skin", skin)]:
        print("%-10s amplitude_corrected %s | phase usable: %s | signal fraction %.2f" % (
            name, frame["amplitude_corrected"].shape, usable_phase(frame) is not None,
            frame["signal_mask"].mean()))

    # The removed parts always add back to the input of the spatial step.
    rebuilt = veg["amplitude_corrected"] + veg["amplitude_stripe"] + veg["amplitude_notch_removed"]
    print("decomposition max error:", float(np.abs(rebuilt - veg["amplitude_pre_spatial"]).max()))

    with np.load(HERE / "vegetables_volume_21frames.npz") as volume:
        print("Volume:", volume["amplitude_corrected"].shape, "frames", volume["frame_ids"].tolist())
        print("z (mm):", np.round(volume["z_mm"], 2).tolist())
        strongest = int(volume["frame_ids"][volume["raw_amplitude"].mean(axis=(1, 2)).argmax()])
        print("strongest frame:", strongest)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping the figures")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.2), constrained_layout=True)
    for column, (kind, frame, label) in enumerate([("leaf", leaf, "Leaf 45"), ("vegetable", veg, "Vegetables 130"),
                                                  ("skin", skin, "Skin 40")]):
        image, aspect = for_display(frame["amplitude_corrected"][0], kind)
        low, high = np.percentile(image, [1, 99])
        axes[0, column].imshow(image, cmap="gray", vmin=low, vmax=high, aspect=aspect)
        axes[0, column].set_title(label + ": amplitude_corrected")
        phase = usable_phase(frame)
        if phase is None:
            image, aspect = for_display(frame["raw_phase"][0], kind)
            axes[1, column].imshow(image, cmap="twilight", vmin=-np.pi, vmax=np.pi, aspect=aspect)
            axes[1, column].set_title(label + ": raw_phase (kept wrapped)")
        else:
            image, aspect = for_display(phase[0], kind)
            low, high = np.nanpercentile(image, [2, 98])
            axes[1, column].imshow(image, cmap="gray", vmin=low, vmax=high, aspect=aspect)
            axes[1, column].set_title(label + ": phase_corrected (relative)")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(HERE / "example_output.png", dpi=130)
    print("wrote", HERE / "example_output.png")


if __name__ == "__main__":
    main()
