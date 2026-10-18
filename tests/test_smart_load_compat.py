"""UI-owned compatibility coverage; no Part 3 source or tests are patched."""
from pathlib import Path
import h5py
import numpy as np
import pytest
from PySide6 import QtCore
import part4_ui.app as viewer
from part4_ui.lfi_core import loaders
from part4_ui.lfi_core.data_types import LFIRawData


def write_scan(path, frame=21):
    values = np.arange(24, dtype=float).reshape(4, 6)
    with h5py.File(path, 'w') as f:
        f[f'amplitude/{frame}'] = values
        f[f'phase/{frame}'] = -values
    return values


def test_single_file_returns_raw_data(tmp_path):
    path = tmp_path/'scan.hdf5'
    expected = write_scan(path)
    data = viewer.smart_load(path)
    assert isinstance(data, LFIRawData)
    assert data.z_indices == [21] and data.amplitude.shape == (1, 4, 6)
    np.testing.assert_array_equal(data.amplitude[0], expected)
    np.testing.assert_array_equal(data.phase[0], -expected)
    assert Path(data.file_path) == path


def test_frame_folder_returns_raw_data(tmp_path):
    folder = tmp_path/'frames'
    folder.mkdir()
    for frame in [25, 21, 23]:
        write_scan(folder/f'frame-{frame}.hdf5', frame)
    data = viewer.smart_load(folder)
    assert isinstance(data, LFIRawData)
    assert data.z_indices == [21, 23, 25]
    assert data.amplitude.shape == data.phase.shape == (3, 4, 6)


@pytest.mark.parametrize('suffix', ['.txt', '.npz', '.mat'])
def test_unsupported_file_is_explicit(tmp_path, suffix):
    path = tmp_path/('unsupported'+suffix)
    path.write_bytes(b'not a supported scan')
    with pytest.raises(ValueError, match='No registered loader supports'):
        viewer.smart_load(path)


def test_missing_and_corrupt_files_propagate(tmp_path):
    with pytest.raises(FileNotFoundError):
        viewer.smart_load(tmp_path/'missing.hdf5')
    path = tmp_path/'corrupt.hdf5'
    path.write_bytes(b'not HDF5')
    with pytest.raises(OSError):
        viewer.smart_load(path)


def test_registered_loader_error_is_not_retried(tmp_path, monkeypatch):
    path = tmp_path/'scan.hdf5'
    write_scan(path)
    error = RuntimeError('backend failure must propagate unchanged')
    def fail(self, path):
        raise error
    monkeypatch.setattr(loaders.HDF5Loader, 'load', fail)
    with pytest.raises(RuntimeError) as result:
        viewer.smart_load(path)
    assert result.value is error


def test_registered_plugin_is_used(tmp_path, monkeypatch):
    path = tmp_path/'scan.compat'
    path.touch()
    data = LFIRawData(np.ones((1, 2, 3)), np.zeros((1, 2, 3)), [0], 'Synthetic', str(path))
    class Plugin:
        name = 'compat-test'
        path_types = ('file',)
        extensions = ('.compat',)
        def load(self, path):
            return data
        def make_data_key(self, path):
            return ('test-v1',)
    monkeypatch.setattr(loaders, '_loaders', dict(loaders._loaders))
    loaders.register_loader(Plugin())
    assert viewer.smart_load(path) is data


def test_explicit_legacy_override_is_window_scoped(qtbot, tmp_path, monkeypatch):
    class Backend:
        def reconstruct(self, data, params):
            return data
    settings = lambda name: QtCore.QSettings(str(tmp_path/name), QtCore.QSettings.Format.IniFormat)
    normal = viewer.MainWindow(settings('normal.ini'), reconstructor=Backend())
    qtbot.addWidget(normal)
    assert normal.volume_panel.reconstruct.isHidden()
    path = tmp_path/'extension.npz'
    path.touch()
    data = LFIRawData(np.ones((2, 3, 4)), np.zeros((2, 3, 4)), [21, 23], 'Synthetic', str(path), {'depth_mm': [5, 7]})
    calls = []
    def override(filename):
        calls.append(filename)
        return data
    monkeypatch.setattr(viewer, 'smart_load', override)
    legacy = viewer.MainWindow(settings('legacy.ini'), reconstructor=Backend())
    qtbot.addWidget(legacy)
    assert not legacy.volume_panel.reconstruct.isHidden()
    # Bootstrap visibility is not permission to compute without calibrated data.
    assert not legacy.reconstruction_available()
    legacy.show()
    errors = []
    monkeypatch.setattr(legacy, 'show_error', errors.append)
    legacy.open_file(path)
    qtbot.waitUntil(lambda: legacy.jobs == 0, timeout=10000)
    assert not errors and legacy.data is data
    assert calls == [str(path.resolve())]
    assert legacy.data_key[0] == str(path.resolve())
    assert legacy.data_key[1] == 'part3-smart-load-compat'
    assert legacy.reconstruction_available()
    assert normal._legacy_smart_loader is None
    legacy.close()
    normal.close()
