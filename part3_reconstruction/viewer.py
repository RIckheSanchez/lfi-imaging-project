"""Open the desktop viewer with the Part 3 depth reconstructor registered.

    python -m part3_reconstruction.viewer [path] [--vegetables-2026] [--spacing 0.1]

``path`` may be an HDF5 scan (opened as usual), a folder of frame-*.hdf5 files
(stacked by Part 1's ``load_volume``), or the Part 2 stacked volume NPZ
(``vegetables_volume_21frames.npz``). Only this process's loader is extended;
the viewer code is unchanged. Tick "Reconstruct" in the 3D panel to use the
uniform depth grid.
"""
import argparse
import sys
from pathlib import Path


def handoff_volume_as_raw(path):
    """Part 2 stacked volume NPZ -> the viewer's LFIRawData, with depth_mm from z_mm."""
    from part4_ui.lfi_core.data_types import LFIRawData
    from .handoff import load_volume
    data, z_mm, frame_ids, _ = load_volume(path)
    return LFIRawData(
        amplitude=data.amplitude, phase=data.phase, z_indices=[int(f) for f in frame_ids],
        target_type="Vegetables", file_path=str(path),
        metadata={"depth_mm": [float(z) for z in z_mm], "phase_converted": True,
                  "phase_encoding_used": "Part 2 handoff raw_phase (wrapped radians)",
                  "amplitude_channel": data.metadata["amplitude_channel"],
                  "source": "Part 2 handoff stacked volume"})


def load_for_viewer(path, fallback):
    """Folder -> Part 1 load_volume; .npz -> Part 2 volume; anything else -> the viewer's loader."""
    path = Path(path)
    if path.is_dir():
        from part1_data import load_volume
        from part4_ui.lfi_core.data_io import adapt_raw_data
        return adapt_raw_data(load_volume(str(path)))
    if path.suffix.lower() == ".npz":
        return handoff_volume_as_raw(path)
    return fallback(str(path))


def main(argv=None):
    parser = argparse.ArgumentParser(description="LFI Viewer with the Part 3 depth reconstructor")
    parser.add_argument("path", nargs="?", help="HDF5 scan, folder of frame-*.hdf5, or Part 2 volume .npz")
    parser.add_argument("--vegetables-2026", action="store_true",
                        help="frame 20 at 30.05 mm, 0.1 mm per frame (2026 Vegetables scan)")
    parser.add_argument("--spacing", type=float, default=0.1, help="output depth spacing in mm")
    args = parser.parse_args(argv)

    import pyqtgraph as pg
    from PySide6 import QtWidgets
    import part4_ui.app as viewer
    from part4_ui.lfi_core.backend import register_reconstructor
    from .gui_backend import DepthReconstructor, VEGETABLES_2026

    params = {"output_spacing_mm": args.spacing}
    if args.vegetables_2026:
        params.update(VEGETABLES_2026)
    register_reconstructor(DepthReconstructor(), params)
    original = viewer.smart_load
    viewer.smart_load = lambda p: load_for_viewer(p, original)

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    application.setApplicationName("LFI Viewer")
    window = viewer.MainWindow()
    window.show()
    if args.path:
        window.open_file(str(Path(args.path).resolve()))
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
