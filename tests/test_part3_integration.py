"""Part 3 integration tests use synthetic scan settings, never real calibration."""
import threading
import h5py
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
from part4_ui.app import MainWindow
from part4_ui.lfi_core.adapters import part3
from part4_ui.lfi_core.backend import (register_reconstructor, get_reconstructor_registration,
                                      IdentityReconstructor)
from part4_ui.lfi_core.data_types import LFIRawData


@pytest.fixture
def scan():
    z, y, x = np.mgrid[:3, :6, :7]
    return LFIRawData((z*100+y*10+x).astype(float), (z*10+y+x/10).astype(float),
                      [21, 23, 25], 'Synthetic', 'synthetic', {'phase_converted': False})


@pytest.fixture
def folder(tmp_path, scan):
    path = tmp_path/'frames'
    path.mkdir()
    for i, frame in enumerate(scan.z_indices):
        with h5py.File(path/f'frame-{frame}.hdf5', 'w') as f:
            f[f'amplitude/{frame}'] = scan.amplitude[i]
            f[f'phase/{frame}'] = scan.phase[i]
    return path


@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    assert part3.register_part3()
    w = MainWindow(QtCore.QSettings(str(tmp_path/'settings.ini'), QtCore.QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    w.errors = []
    monkeypatch.setattr(w, 'show_error', w.errors.append)
    w.show()
    yield w
    qtbot.waitUntil(lambda: w.jobs == 0, timeout=20000)
    w.volume_timer.stop()
    w.close()
    register_reconstructor(None)


def ready(qtbot, window):
    qtbot.waitUntil(lambda: window.jobs == 0, timeout=20000)
    assert not window.errors


def enter(panel, values=('10', '2', '21')):
    # Deliberately synthetic settings for these tiny fixture frames.
    for key, value in zip(part3.CALIBRATION_KEYS, values):
        panel.calibration_fields[key].setText(value)


def test_registration_success():
    try:
        assert part3.register_part3()
        r = get_reconstructor_registration()
        assert isinstance(r.implementation, part3.Part3Reconstructor)
        assert r.params == {}
    finally:
        register_reconstructor(None)


def test_import_failure_falls_back(qtbot, tmp_path, monkeypatch, caplog):
    def fail(name):
        raise ImportError('test: Part 3 dependency unavailable')
    monkeypatch.setattr(part3.importlib, 'import_module', fail)
    assert part3.register_part3() is False
    assert get_reconstructor_registration() is None
    assert 'retaining v1.3' in caplog.text
    assert 'dependency unavailable' in caplog.text
    w = MainWindow(QtCore.QSettings(str(tmp_path/'fallback.ini'), QtCore.QSettings.Format.IniFormat))
    qtbot.addWidget(w)
    assert isinstance(w.reconstructor, IdentityReconstructor)
    assert w.volume_panel.reconstruct.isHidden()
    assert w.volume_panel.calibration_toggle.isHidden()


def test_uncalibrated_chip_contract_keeps_v13(qtbot, window, folder):
    window.open_file(folder)
    ready(qtbot, window)
    window.data.target_type = 'Chip'  # Same missing-depth contract, no Realdata dependency.
    window.update_info()
    panel = window.volume_panel
    assert panel.reconstruct.isHidden() and panel.reconstruction_hint.isHidden()
    assert not panel.calibration_toggle.isHidden()
    assert not panel.calibration_toggle.isChecked() and panel.calibration_body.isHidden()
    assert all(field.text() == '' for field in panel.calibration_fields.values())
    assert panel.channel_notice.text() == '3D view uses the amplitude channel'
    assert window.csv_action.shortcut().toString() == 'Meta+Shift+E'
    assert window.fit_action.shortcut().toString() == 'Meta+0'
    assert window.show_waveform.text() == 'Show waveform'
    window.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: panel.result is not None, timeout=10000)
    assert 'ui_part3' not in panel.result
    assert panel.result['original_shape'] == (3, 6, 7)
    assert panel.channel_notice.text() == '3D view uses the amplitude channel'


def test_legal_calibration_shows_option(qtbot, window, folder):
    window.open_file(folder)
    ready(qtbot, window)
    panel = window.volume_panel
    panel.calibration_toggle.click()
    assert not panel.calibration_body.isHidden()
    enter(panel)
    assert window.reconstruction_available()
    assert not panel.reconstruct.isHidden()
    assert panel.reconstruct.text() == 'Use Part 3 depth reconstruction'
    assert not panel.calibration_error.text()
    window.set_language('CN')
    assert panel.reconstruct.text() == '使用 Part 3 深度重建'
    assert panel.channel_notice.text() == '三维视图使用振幅通道'


@pytest.mark.parametrize('values', [('', '2', '21'), ('10', '', '21'), ('10', '2', ''),
    ('-1', '2', '21'), ('0', '2', '21'), ('10', '0', '21'), ('10', '-2', '21'),
    ('nan', '2', '21'), ('10', 'inf', '21'), ('10', '2', 'nan'), ('no', '2', '21'),
    ('10', '2', '20'), ('10', '2', '26'), ('10', '2', '22'), ('10', '2', '21.5'),
    ('1', '2', '25')])
def test_invalid_calibration_rejected(window, scan, values):
    window.data = scan
    window.update_info()
    enter(window.volume_panel, values)
    assert not window.reconstruction_available()
    assert window.volume_panel.reconstruct.isHidden()
    assert window.volume_panel.calibration_error.text()


@pytest.mark.parametrize('depth', [[5, 9, 13], [13, 5, 9]])
def test_metadata_calibration_and_axis_phase_policy(scan, depth):
    scan.metadata['depth_mm'] = depth
    out = part3.Part3Reconstructor().reconstruct(scan, {})
    grid = np.array(out.metadata['depth_mm'])
    nearest = np.abs(grid[:, None] - np.array(depth)[None, :]).argmin(axis=1)
    assert out.amplitude.shape == (3, 6, 7)
    assert out.amplitude.dtype == out.phase.dtype == np.float32
    np.testing.assert_array_equal(out.phase, scan.phase.astype(np.float32)[nearest])
    assert out.metadata['part3_reconstruction']['output_spacing_mm'] == 4
    assert out.metadata['ui_part3']['axes'] == 'sample_indices'
    assert 'depth_mm' in scan.metadata and 'ui_part3' not in scan.metadata


def test_adapter_ignores_upstream_calibration_presets(scan, monkeypatch):
    import part3_reconstruction.gui_backend as backend
    monkeypatch.setitem(backend.DEFAULT_PARAMS, 'output_spacing_mm', 9876)
    monkeypatch.setitem(backend.VEGETABLES_2026, 'first_frame_id', -9999)
    adapter = part3.Part3Reconstructor()
    with pytest.raises(ValueError):
        adapter.reconstruct(scan, {})
    out = adapter.reconstruct(scan, {'z_start_mm': 10, 'z_step_mm': 2, 'first_frame_id': 21})
    assert out.amplitude.shape == (5, 6, 7)
    np.testing.assert_allclose(out.amplitude[1], (scan.amplitude[0] + scan.amplitude[1])/2)
    np.testing.assert_array_equal(out.phase, scan.phase.astype(np.float32)[[0, 0, 1, 1, 2]])
    assert out.metadata['phase_converted'] is False
    assert out.metadata['ui_part3']['output_spacing_mm'] == 2
    assert out.metadata['part3_reconstruction']['phase_source_slice_ids'] == [21, 21, 23, 23, 25]


@pytest.mark.parametrize('depth', [None, [1, 2], [1, 1, 2], [1, float('nan'), 3], [-1, 2, 3]])
def test_invalid_metadata_rejected(scan, depth):
    scan.metadata['depth_mm'] = depth
    with pytest.raises((TypeError, ValueError)):
        part3.Part3Reconstructor().reconstruct(scan, {})


def test_folder_end_to_end_and_reset(qtbot, window, folder, tmp_path, monkeypatch):
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', lambda *args: str(folder))
    window.open_folder_dialog()
    ready(qtbot, window)
    original = window.data.amplitude.copy()
    original_phase = window.data.phase.copy()
    enter(window.volume_panel)
    backend = window.reconstructor
    calls = []
    original_reconstruct = backend.reconstruct
    def record(data, params):
        out = original_reconstruct(data, params)
        calls.append((threading.get_ident(), out))
        return out
    monkeypatch.setattr(backend, 'reconstruct', record)
    window.tabs.setCurrentIndex(1)
    window.volume_panel.reconstruct.setChecked(True)
    window.request_volume()
    ready(qtbot, window)
    assert calls and calls[-1][0] != threading.get_ident()
    out = calls[-1][1]
    assert out.amplitude.shape == (5, 6, 7)
    assert window.volume_panel.result['original_shape'] == (5, 6, 7)
    assert 'phase from nearest measured slice' in window.volume_panel.channel_notice.text()
    assert 'axes in sample indices' in window.volume_panel.channel_notice.text()
    assert 'uniform depth grid' in window.volume_panel.channel_notice.text()
    assert 'Physical XYZ proportions are not represented' in window.volume_panel.hint.text()
    assert 'unknown' not in window.volume_panel.hint.text()
    np.testing.assert_array_equal(window.data.amplitude, original)
    np.testing.assert_array_equal(window.data.phase, original_phase)
    window.set_language('CN')
    assert '相位取最近实测切片' in window.volume_panel.channel_notice.text()
    window.set_language('EN')
    # Editing to an invalid calibration immediately revokes reconstruction.
    window.volume_panel.calibration_fields['z_step_mm'].setText('0')
    assert window.volume_panel.reconstruct.isHidden()
    assert not window.volume_panel.reconstruct.isChecked()
    qtbot.waitUntil(lambda: window.volume_panel.result is not None, timeout=10000)
    ready(qtbot, window)
    assert 'ui_part3' not in window.volume_panel.result
    assert window.volume_panel.channel_notice.text() == '3D view uses the amplitude channel'
    enter(window.volume_panel)
    window.open_file(folder/'frame-21.hdf5')
    assert all(field.text() == '' for field in window.volume_panel.calibration_fields.values())
    ready(qtbot, window)
    assert window.volume_panel.reconstruct.isHidden()
    assert not window.volume_panel.calibration_toggle.isChecked()


def test_metadata_enables_ui_without_manual(window, scan):
    window.data = scan
    scan.metadata['depth_mm'] = [5, 9, 13]
    window.update_info()
    assert window.reconstruction_available()
    assert not window.volume_panel.reconstruct.isHidden()
    assert all(field.text() == '' for field in window.volume_panel.calibration_fields.values())
    # Partial manual input must not silently reuse metadata.
    window.volume_panel.calibration_fields['z_start_mm'].setText('10')
    assert not window.reconstruction_available()
    assert window.volume_panel.reconstruct.isHidden()
