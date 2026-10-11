"""GUI-independent Matplotlib figures and reproducible display exports."""
import colorsys
import json
from pathlib import Path
import numpy as np
from matplotlib.figure import Figure
from .data import ImagingData
from .colormap_engine import normalise, fold_phase, resolve_cmap


class ImageGenerator:
    def __init__(self, amplitude, phase=None, metadata=None, *, scale_bar_length=None):
        self.data = amplitude if isinstance(amplitude, ImagingData) else ImagingData(amplitude, phase, metadata or {})
        if self.data.amplitude.ndim != 2:
            raise ValueError("ImageGenerator requires a single 2D slice")
        self.scale_bar_length = scale_bar_length

    def _scale_bar(self, ax):
        if self.scale_bar_length is None:
            return
        from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
        extent, unit = self._extent()
        length = float(self.scale_bar_length)
        if unit in {"pixel", "unknown"} or not np.isfinite(length) or not 0 < length <= (extent[1]-extent[0])*.8:
            raise ValueError("scale bar requires calibrated spacing and length within 80% of image width")
        bar = AnchoredSizeBar(ax.transData, length, f"{length:g} {unit}", "lower right", pad=.4, color="white", frameon=True)
        bar.patch.set_facecolor("black")
        bar.patch.set_alpha(.6)
        ax.add_artist(bar)

    def _extent(self):
        h, w = self.data.amplitude.shape
        spacing = self.data.metadata.get("spacing_xy")
        if spacing is None:
            return (-0.5, w - 0.5, -0.5, h - 0.5), "pixel"
        dx, dy = spacing
        if not np.isfinite([dx, dy]).all() or min(dx, dy) <= 0:
            raise ValueError("spacing_xy must contain positive finite dx, dy")
        ox, oy = self.data.metadata.get("origin_xy", (0, 0))
        return (ox-dx/2, ox+(w-.5)*dx, oy-dy/2, oy+(h-.5)*dy), self.data.metadata.get("spatial_unit", "unknown")

    def _aspect(self):
        """Pixel-unit axes keep the physical frame shape when pixel_aspect_yx (dy/dx) is known."""
        if self.data.metadata.get("spacing_xy") is not None:
            return "equal"
        ratio = self.data.metadata.get("pixel_aspect_yx")
        if ratio is None:
            return "equal"
        ratio = float(ratio)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("pixel_aspect_yx must be positive and finite")
        return ratio

    def _draw(self, fig, ax, channel, *, cmap=None, norm="percentile", phase_mode="cyclic", limits=None):
        arr = getattr(self.data, channel)
        if phase_mode not in {"cyclic", "sequential"}:
            raise ValueError("phase_mode must be cyclic or sequential")
        cyclic = channel == "phase" and phase_mode == "cyclic"
        if cyclic and self.data.metadata["phase_unit"] != "rad":
            raise ValueError("cyclic phase requires calibrated radians; use sequential for raw counts")
        values = fold_phase(arr) if cyclic else normalise(arr, norm, limits=limits)
        colour = resolve_cmap(cmap or ("phase" if cyclic else "inferno" if channel == "amplitude" else "viridis"))
        colour = colour.with_extremes(bad=(0.5, 0.5, 0.5, 0))
        extent, unit = self._extent()
        im = ax.imshow(values, origin="lower", extent=extent, cmap=colour,
                       vmin=-np.pi if cyclic else 0, vmax=np.pi if cyclic else 1, interpolation="nearest",
                       aspect=self._aspect())
        title = channel.capitalize()
        if channel == "phase" and self.data.metadata["phase_unit"] == "raw_counts":
            title += " (uncalibrated counts)"
        ax.set(xlabel=f"x ({unit})", ylabel=f"y ({unit})", title=title)
        self._scale_bar(ax)
        fig.colorbar(im, ax=ax, shrink=.65, label="Phase (rad, wrapped)" if cyclic else f"{channel.capitalize()} (normalised)")

    def _export(self, fig, export, settings):
        if export is not None:
            path = Path(export)
            if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
                raise ValueError("image export must be PNG or TIFF")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
            payload = {"metadata": self.data.metadata, "display": dict(settings, scale_bar_length=self.scale_bar_length),
                       "shape_yx": list(self.data.amplitude.shape),
                       "note": "Rendered display image, not raw quantitative data."}
            path.with_suffix(path.suffix + ".json").write_text(json.dumps(payload, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)), encoding="utf-8")
        return fig

    def generate_amplitude(self, cmap="inferno", norm="percentile", export=None, limits=None):
        fig = Figure(figsize=(7, 5), layout="constrained")
        self._draw(fig, fig.subplots(), "amplitude", cmap=cmap, norm=norm, limits=limits)
        return self._export(fig, export, dict(channel="amplitude", cmap=cmap, norm=norm, limits=limits))

    def generate_phase(self, phase_mode="cyclic", cmap=None, norm="linear", export=None, limits=None):
        fig = Figure(figsize=(7, 5), layout="constrained")
        self._draw(fig, fig.subplots(), "phase", phase_mode=phase_mode, cmap=cmap, norm=norm, limits=limits)
        return self._export(fig, export, dict(channel="phase", phase_mode=phase_mode, cmap=cmap, norm=norm, limits=limits))

    def generate_comparison(self, mode="side_by_side", phase_mode="cyclic", norm="percentile", export=None,
                            max_lightness=0.5):
        """Side-by-side panels or HSL fusion.

        In the fusion, lightness = max_lightness * normalised amplitude. With the
        default 0.5 the strongest pixels are fully saturated hues instead of white,
        so phase stays visible where the signal is strongest.
        """
        fig = Figure(figsize=(11, 5), layout="constrained")
        if mode == "side_by_side":
            axes = fig.subplots(1, 2, sharex=True, sharey=True)
            for ax, channel in zip(axes, ("amplitude", "phase")):
                self._draw(fig, ax, channel, norm=norm, phase_mode=phase_mode)
        elif mode == "hsl_fusion":
            if self.data.metadata["phase_unit"] != "rad":
                raise ValueError("HSL hue requires phase calibrated in radians")
            if not 0 < max_lightness <= 1:
                raise ValueError("max_lightness must be in (0, 1]")
            lightness = max_lightness * normalise(self.data.amplitude, norm)
            hue = (fold_phase(self.data.phase) + np.pi) / (2*np.pi)
            valid = np.isfinite(lightness) & np.isfinite(hue)
            rgb = np.zeros((*hue.shape, 4), dtype=np.float32)
            rgb[valid, :3] = np.array([colorsys.hls_to_rgb(float(h), float(l), 1.0) for h, l in zip(hue[valid], lightness[valid])])
            rgb[valid, 3] = 1
            extent, unit = self._extent()
            ax = fig.subplots()
            ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", aspect=self._aspect())
            ax.set(title=f"HSL fusion: hue = wrapped phase; lightness = amplitude (max {max_lightness:g})", xlabel=f"x ({unit})", ylabel=f"y ({unit})")
            self._scale_bar(ax)
        else:
            raise ValueError("mode must be side_by_side or hsl_fusion")
        return self._export(fig, export, dict(mode=mode, phase_mode="cyclic" if mode == "hsl_fusion" else phase_mode, norm=norm,
                                                max_lightness=max_lightness if mode == "hsl_fusion" else None))
