"""Spin boxes use ASCII numeric text independently of application startup order."""
import pytest
from PySide6 import QtCore, QtWidgets
from part4_ui.app import MainWindow


@pytest.mark.parametrize('locale_name', ['ar_EG', 'de_DE'])
def test_spinbox_line_edits_ignore_native_digits_and_decimal_comma(qtbot, tmp_path, locale_name):
    previous = QtCore.QLocale()
    native = QtCore.QLocale(locale_name)
    try:
        QtCore.QLocale.setDefault(native)
        # Establish that this locale really differs from C numeric formatting.
        if locale_name == 'ar_EG':
            assert native.zeroDigit() != '0'
            assert any(character not in '0123456789' for character in native.toString(303))
        else:
            assert native.decimalPoint() == ','
            assert ',' in native.toString(303.125, 'f', 3)
        # Do not call main(): per-widget locales must also work for direct creation.
        window = MainWindow(QtCore.QSettings(str(tmp_path/'locale.ini'), QtCore.QSettings.Format.IniFormat))
        qtbot.addWidget(window)
        assert QtCore.QLocale().name() == native.name()
        window.z_slider.setRange(0, 9999)
        window.z_spin.setRange(0, 9999)
        window.z_spin.setValue(533)
        window.pixel_x.setRange(0, 9999)
        window.pixel_y.setRange(0, 9999)
        window.pixel_x.setValue(303)
        window.pixel_y.setValue(533)
        window.pixel_size.setValue(303.125)
        window.volume_panel.percentile_value.setValue(77)
        boxes = window.findChildren(QtWidgets.QSpinBox) + window.findChildren(QtWidgets.QDoubleSpinBox)
        expected = {window.z_spin: '533', window.pixel_x: '303', window.pixel_y: '533',
                    window.pixel_size: '303.125000', window.volume_panel.percentile_value: '77'}
        assert set(boxes) == set(expected)  # Cover every numeric spin box in the UI.
        for language in ('EN', 'CN'):
            window.set_language(language)
            for box in boxes:
                assert box.locale().name() == 'C'
                text = box.lineEdit().text()
                assert text == expected[box]
                assert text and all(character in '0123456789.' for character in text)
        window.volume_timer.stop()
        window.close()
    finally:
        QtCore.QLocale.setDefault(previous)
    assert QtCore.QLocale().name() == previous.name()
