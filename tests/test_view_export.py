"""GUI export routing and framebuffer capture with no live OpenGL viewport."""
from pathlib import Path
from unittest.mock import Mock
import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from part4_ui.app import MainWindow
from part4_ui.lfi_core.data_types import LFIRawData
from part4_ui.lfi_core.export import export_csv


@pytest.fixture
def window(qtbot, tmp_path):
    widget = MainWindow(QtCore.QSettings(str(tmp_path/'export.ini'), QtCore.QSettings.Format.IniFormat))
    widget.errors = []
    widget.show_error = widget.errors.append
    qtbot.addWidget(widget)
    values = np.arange(40, dtype=float).reshape(2, 4, 5)
    widget.data = LFIRawData(values, -values, [530, 545], 'Chip', str(tmp_path/'chip.sample.hdf5'),
        {'phase_encoding_used': 'test,encoding', 'phase_converted': False})
    widget.raw_data = widget.data
    with QtCore.QSignalBlocker(widget.z_slider):
        widget.z_slider.setRange(0, 1)
        widget.z_slider.setValue(1)
    widget.show()
    yield widget
    widget.volume_timer.stop()
    qtbot.waitUntil(lambda: widget.jobs == 0, timeout=10000)
    widget.close()


def ready(qtbot, window):
    qtbot.waitUntil(lambda: window.jobs == 0, timeout=10000)
    assert not window.errors


@pytest.mark.parametrize('direct', [False, True])
def test_both_csv_entrypoints_include_header(qtbot, window, tmp_path, monkeypatch, direct):
    window.channel.setCurrentIndex(1)
    ready(qtbot, window)
    output = tmp_path/'phase.csv'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args: (str(output), 'CSV'))
    if direct:
        window.export_csv_dialog()
    else:
        # Choose CSV in the actual shared format dialog without displaying a modal loop.
        monkeypatch.setattr(QtWidgets.QDialog, 'exec', lambda self: QtWidgets.QDialog.DialogCode.Accepted)
        window.export_dialog()
    ready(qtbot, window)
    content = output.read_text(encoding='utf-8')
    assert '# Channel: phase' in content and '# Slice index (zero-based): 1' in content
    assert '# Z value: 545' in content and '# Phase encoding: test;encoding' in content
    assert '# Values: raw' in content and '# Shape: 4 rows (Y) x 5 columns (X)' in content
    assert np.array_equal(np.loadtxt(output, delimiter=','), window.data.phase[1])
    # Compare actual numeric bytes using the same newline rules as the old write_text path.
    previous = tmp_path/'previous-matrix.csv'
    previous.write_text(export_csv(window.data.phase[1]))
    matrix = b''.join(line for line in output.read_bytes().splitlines(keepends=True) if not line.startswith(b'# '))
    assert matrix == previous.read_bytes()


def test_offscreen_button_remains_disabled_and_does_not_capture(qtbot, window):
    panel = window.volume_panel
    assert not panel.export_button.isEnabled()
    window.request_volume()
    ready(qtbot, window)
    assert panel.result is not None and panel.scene is None
    assert not panel.export_button.isEnabled()
    window.export_3d_view_dialog()
    assert not window.errors and not window.exported_path
    panel.clear_result()
    assert not panel.export_button.isEnabled()


@pytest.fixture
def framebuffer(qtbot, window, monkeypatch):
    window.request_volume()
    ready(qtbot, window)
    image = QtGui.QImage(30, 20, QtGui.QImage.Format.Format_RGBA8888)
    image.fill(QtGui.QColor('#345678'))
    scene = Mock()
    scene.grabFramebuffer.return_value = image
    window.volume_panel.scene = scene
    # Only the platform guard is simulated; no OpenGL viewport is created.
    monkeypatch.setattr(QtWidgets.QApplication, 'platformName', lambda: 'test-onscreen')
    window.volume_panel.show_result(window.volume_panel.result)
    return scene, image


def test_export_button_state_translation_and_invalidation(window, framebuffer):
    panel = window.volume_panel
    assert panel.export_button.isEnabled()
    assert panel.export_button.text() == 'Export 3D view…'
    panel.set_language('CN')
    assert panel.export_button.text() == '导出三维视图…'
    panel.invalidate()
    assert not panel.export_button.isEnabled()
    assert not panel.can_export_view()


@pytest.mark.parametrize('mode', ['volume', 'isosurface'])
def test_framebuffer_png_and_matching_sidecar(qtbot, window, framebuffer, tmp_path, monkeypatch, mode):
    scene, image = framebuffer
    window.channel.setCurrentIndex(1)  # 3D remains amplitude, not the selected 2D phase.
    ready(qtbot, window)
    if mode == 'isosurface':
        window.volume_panel.mode.setCurrentIndex(1)
        window.request_volume()
        ready(qtbot, window)
    directory = tmp_path/'exports'
    directory.mkdir()
    window.settings.setValue('export_directory', str(directory))
    output = directory/'current-view.png'
    dialogs = []
    def choose(*args):
        assert scene.grabFramebuffer.call_count == 1  # Capture precedes the modal dialog.
        dialogs.append(args)
        return str(output), 'PNG (*.png)'
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', choose)
    window.volume_panel.export_button.click()
    ready(qtbot, window)
    scene.grabFramebuffer.assert_called_once_with()
    assert Path(dialogs[0][2]) == directory/f'lfi_3d_{mode}_chip.sample.png'
    saved = QtGui.QImage(str(output))
    assert saved.size() == image.size() and saved.pixelColor(0, 0) == image.pixelColor(0, 0)
    text = output.with_suffix('.txt').read_text(encoding='utf-8')
    assert f'# 3D mode: {mode}' in text
    assert '# Channel: amplitude' in text
    assert '# Framebuffer shape: 20 rows (Y) x 30 columns (X)' in text
    assert '# Source file: chip.sample.hdf5' in text
    assert '# Downsampling step: 1' in text
    expected = '(90; 99.5)' if mode == 'isosurface' else '(1; 99.5)'
    assert '# Percentile thresholds: ' + expected in text
    assert '# Threshold values:' in text
    assert window.exported_path == str(output)
    assert window.finder_button.isVisible()
    assert window.settings.value('export_directory') == str(directory)


def test_cancel_3d_export_writes_nothing(window, framebuffer, monkeypatch, tmp_path):
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args: ('', ''))
    window.export_3d_view_dialog()
    assert window.jobs == 0 and not window.exported_path
    assert not list(tmp_path.glob('*.png')) and not list(tmp_path.glob('*.txt'))


def test_empty_framebuffer_reports_error(window, framebuffer, monkeypatch):
    scene, _ = framebuffer
    scene.grabFramebuffer.return_value = QtGui.QImage()
    def unexpected(*args):
        pytest.fail('Save dialog must not open without a framebuffer')
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', unexpected)
    window.export_3d_view_dialog()
    assert any('framebuffer is unavailable' in message for message in window.errors)
    assert window.jobs == 0
