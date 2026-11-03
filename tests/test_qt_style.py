"""Style selection is pure; Fusion checks do not claim Windows glyph validation."""
import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from part4_ui.qt_style import select_qt_style
from part4_ui.app import MainWindow
from part4_ui.lfi_core.data_types import LFIRawData


@pytest.mark.parametrize('platform,override,expected', [
    ('win32', None, 'Fusion'), ('darwin', None, None), ('linux', None, None),
    ('win32', 'windows11', 'windows11'), ('darwin', 'Fusion', 'Fusion'),
    ('win32', '  Fusion  ', 'Fusion'), ('win32', '', 'Fusion'), ('darwin', ' ', None),
])
def test_select_qt_style(platform, override, expected):
    assert select_qt_style(platform, override) == expected


def test_fusion_spinboxes_palette_and_dark_plot(qapp, qtbot, tmp_path):
    previous = qapp.style().objectName()
    palette = QtGui.QPalette(qapp.palette())
    try:
        qapp.setStyle('Fusion')
        window = MainWindow(QtCore.QSettings(str(tmp_path/'fusion.ini'), QtCore.QSettings.Format.IniFormat))
        qtbot.addWidget(window)
        values = np.arange(2*400*400, dtype=float).reshape(2, 400, 400)
        window.data = LFIRawData(values, -values, [530, 531], 'Synthetic', 'preview.hdf5')
        window.show()
        window.request_slice()
        qtbot.waitUntil(lambda: window.jobs == 0, timeout=10000)
        window.pixel_x.setValue(303)
        window.pixel_y.setValue(303)
        window.pixel_size.setValue(303.125)
        assert window.pixel_x.text() == window.pixel_y.text() == '303'
        assert window.pixel_size.value() == 303.125
        assert qapp.style().objectName() == 'fusion'
        assert window.image_plot.backgroundBrush().color().name() == '#151d2c'
        assert window.histogram.backgroundBrush().color() == window.palette().color(QtGui.QPalette.ColorRole.Window)
        for axis in ('left', 'bottom'):
            assert window.histogram.getAxis(axis).textPen().color() == window.palette().color(QtGui.QPalette.ColorRole.WindowText)
        assert window.grab().save(str(tmp_path/'fusion-preview.png'))
        print('FUSION_PREVIEW', tmp_path/'fusion-preview.png')
        window.close()
    finally:
        qapp.setStyle(previous)
        qapp.setPalette(palette)
