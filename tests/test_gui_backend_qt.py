"""The Part 3 depth reconstructor running inside the desktop viewer (needs PySide6 + pytest-qt)."""
import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
from PySide6 import QtCore, QtWidgets  # noqa: E402
import part4_ui.app as viewer  # noqa: E402
from part4_ui.lfi_core.backend import register_reconstructor  # noqa: E402
from part3_reconstruction.gui_backend import DepthReconstructor  # noqa: E402
from part3_reconstruction.viewer import load_for_viewer  # noqa: E402


class Recording(DepthReconstructor):
    def __init__(self):
        self.outputs = []

    def reconstruct(self, data, params):
        out = super().reconstruct(data, params)
        self.outputs.append(out)
        return out


def test_depth_reconstructor_in_viewer(qtbot, tmp_path, monkeypatch):
    z = np.array([30.15, 30.25, 30.55])
    rng = np.random.default_rng(0)
    amp = (1 + rng.random((3, 8, 10))).astype(np.float32)
    path = tmp_path / "vegetables_volume.npz"
    np.savez(path, amplitude_corrected=amp, raw_amplitude=amp, raw_phase=np.zeros_like(amp),
             z_mm=z, frame_ids=np.array([21, 22, 25]), signal_mask=amp > 0)
    original = viewer.smart_load
    monkeypatch.setattr(viewer, "smart_load", lambda p: load_for_viewer(p, original))
    backend = Recording()
    register_reconstructor(backend, {"output_spacing_mm": 0.1})
    try:
        window = viewer.MainWindow(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
        window.errors = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a: window.errors.append(a[-1]))
        qtbot.addWidget(window)
        window.show()
        assert not window.volume_panel.reconstruct.isHidden()          # control shown once registered
        window.open_file(str(path))
        qtbot.waitUntil(lambda: window.jobs == 0, timeout=60000)
        assert not window.errors and window.data.z_indices == [21, 22, 25]
        window.volume_panel.reconstruct.setChecked(True)
        window.request_volume()
        qtbot.waitUntil(lambda: window.jobs == 0, timeout=60000)
        assert not window.errors
        out = backend.outputs[-1]
        assert out.amplitude.shape == (5, 8, 10)                         # 30.15 .. 30.55 in 0.1 mm
        assert out.z_indices == [30.15, 30.25, 30.35, 30.45, 30.55]
        assert window.data.amplitude.shape == (3, 8, 10)                 # 2D views keep the measured slices
        window.close()
    finally:
        register_reconstructor(None)
