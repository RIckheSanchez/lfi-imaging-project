"""Dynamic slice text must not resize the surrounding panels or image host."""
import json
from types import SimpleNamespace
import numpy as np
from PySide6 import QtCore
import part4_ui.app as module
from part4_ui.lfi_core.data_types import LFIRawData


def test_slice_geometry_stays_fixed_across_slices_languages_and_manual_dock_resize(qtbot, tmp_path, monkeypatch):
    grid = np.arange(24*32, dtype=float).reshape(24, 32) + .123456
    magnitudes = [0, -100, 100, -5, 5, -50, 50]
    amplitude = np.stack([grid * (-1 if i % 2 else 1) * 10.**magnitudes[i % 7] for i in range(70)])
    scan = LFIRawData(amplitude, -amplitude, list(range(530, 600)), 'Synthetic',
        str(tmp_path/'synthetic-wide-values.hdf5'), {'ui_preprocessing': {
            'normalisation': 'zscore', 'amplitude_units': 'dimensionless',
            'channels': {'amplitude': {'processed': True}, 'phase': {'processed': False}}}})
    monkeypatch.setattr(module, 'load_path', lambda path: (scan, (str(path), 'synthetic-layout')))
    window = module.MainWindow(
        QtCore.QSettings(str(tmp_path/'layout.ini'), QtCore.QSettings.Format.IniFormat),
        preprocessor=SimpleNamespace(preprocess=lambda data, params: data))
    qtbot.addWidget(window)
    errors = []
    window.show_error = errors.append
    window.setFixedSize(1440, 900)
    window.show()
    window.apply_preprocessing.setChecked(True)
    window.open_file(scan.file_path)
    def settle():
        qtbot.waitUntil(lambda: window.jobs == 0, timeout=10000)
        qtbot.wait(25)  # Deliver the layout requests posted by dynamic QLabel text.
        assert not errors
    settle()
    window.select_pixel(17, 13)
    right_panel = window.scan_group.parentWidget()
    records = []
    try:
        for language in ('EN', 'CN'):
            window.set_language(language)
            settle()
            for index in range(scan.n_slices):
                window.z_slider.setValue(index)
                settle()
                assert window.selected_pixel == (17, 13)
                label = window.z_label
                assert not label.wordWrap()
                assert label.fontMetrics().horizontalAdvance(label.text()) <= label.contentsRect().width()
                assert label.height() == max(label.fontMetrics().boundingRect(
                    module.text('slice_position', lang)).height() for lang in ('EN', 'CN'))
                for wrapped in (window.pixel_label, window.waveform_pixel,
                                window.scan_label, window.preprocessing_status):
                    assert wrapped.heightForWidth(wrapped.width()) <= wrapped.height(), (
                        language, index, wrapped.text(), wrapped.heightForWidth(wrapped.width()), wrapped.height())
                records.append({'language': language, 'index': index,
                    'plot_host': window.plot_host.geometry().getRect(),
                    'dock': window.dock.geometry().getRect(),
                    'right_panel': right_panel.geometry().getRect(),
                    'bar_heights': (window.menuBar().height(), window.statusBar().height(), window.tabs.tabBar().height()),
                    'image_plot': window.image_plot.geometry().getRect(),
                    'pixel_text': window.pixel_label.text(), 'slice_text': window.z_label.text()})
        (tmp_path/'slice-geometries.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
        for language in ('EN', 'CN'):
            for key in ('plot_host', 'dock', 'right_panel', 'image_plot'):
                shapes = sorted({tuple(row[key]) for row in records if row['language'] == language})
                print('GEOMETRY', language, key, shapes)
        print('GEOMETRY_RECORDS', tmp_path/'slice-geometries.json')
        assert len({tuple(row['plot_host']) for row in records}) == 1
        assert len({tuple(row['dock']) for row in records}) == 1
        assert len({tuple(row['right_panel']) for row in records}) == 1
        # Text-driven resizing is forbidden; deliberate user dock resizing is allowed.
        before = window.dock.width()
        window.resizeDocks([window.dock], [before + 60], QtCore.Qt.Orientation.Horizontal)
        settle()
        assert window.dock.width() >= before + 40
        resized = window.plot_host.geometry()
        for language, index in [('EN', 0), ('CN', 69), ('EN', 9)]:
            window.set_language(language)
            window.z_slider.setValue(index)
            settle()
            assert window.plot_host.geometry() == resized
    finally:
        window.volume_timer.stop()
        settle()
        window.close()
