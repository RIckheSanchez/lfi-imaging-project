"""Dynamic slice text must not resize the surrounding panels or image host."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6 import QtCore
import part4_ui.app as module
from part4_ui.lfi_core.data_types import LFIRawData


@pytest.mark.parametrize('dock_minimum', [None, 536], ids=['platform-default', 'windows-sized-dock'])
def test_slice_geometry_stays_fixed_across_slices_languages_and_manual_dock_resize(qtbot, tmp_path, monkeypatch, dock_minimum):
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
    if dock_minimum is not None:
        # Reproduce the minimum-width constraint reported by Windows CI locally.
        update_width = window.update_slice_label_width
        def enforce_platform_minimum():
            update_width()
            content = window.dock.widget()
            content.setMinimumWidth(max(content.minimumWidth(), dock_minimum))
        monkeypatch.setattr(window, 'update_slice_label_width', enforce_platform_minimum)
        enforce_platform_minimum()
        settle()
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
        metrics = window.z_label.fontMetrics()
        slice_widths = {
            language: max(metrics.horizontalAdvance(module.text('slice_position', language).format(
                number=scan.n_slices, total=scan.n_slices, z=z)) for z in scan.z_indices)
            for language in ('EN', 'CN')}
        form_widths = {
            language: max(label.fontMetrics().horizontalAdvance(module.text(key, language))
                          for key, label in window.labels.items())
            for language in ('EN', 'CN')}
        margins = window.dock.widget().layout().contentsMargins()
        print('DOCK_WIDTH_DIAGNOSTICS', json.dumps({
            'slice_text_widths': slice_widths, 'slice_margin': 12,
            'slice_reserved_width': window.z_label.minimumWidth(),
            'form_label_column_widths': form_widths,
            'form_horizontal_spacing': window.form.horizontalSpacing(),
            'dock_layout_margins': [margins.left(), margins.top(), margins.right(), margins.bottom()],
            'control_spacing_allowance': 32,
            'dock_content_minimum_width': window.dock.widget().minimumWidth(),
            'dock_minimum_size_hint': window.dock.minimumSizeHint().width(),
            'central_minimum_size_hint': window.centralWidget().minimumSizeHint().width(),
            'font': window.z_label.font().toString(),
            'resolved_font_family': window.z_label.fontInfo().family(),
            'logical_dpi': window.z_label.logicalDpiX(),
        }, ensure_ascii=False))
        # Keep the fixed-size sweep above intact. Give this separate manual resize
        # check enough space even with larger Windows fonts and panel minima.
        central = window.centralWidget()
        minimum_central_width = max(central.minimumWidth(), central.minimumSizeHint().width())
        window.setFixedSize(max(window.width() + 600,
            window.dock.width() + 60 + minimum_central_width + 64), window.height())
        settle()
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
        grown = window.dock.width()
        minimum = max(window.dock.minimumWidth(), window.dock.minimumSizeHint().width())
        window.resizeDocks([window.dock], [minimum], QtCore.Qt.Orientation.Horizontal)
        settle()
        assert window.dock.width() <= grown - 40
        assert minimum <= window.dock.width() <= max(before, minimum) + 4
    finally:
        window.volume_timer.stop()
        settle()
        window.close()
