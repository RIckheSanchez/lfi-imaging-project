"""Count suffix and Scan Info heading have different Chinese grammar."""
import numpy as np
import pytest
from PySide6 import QtCore
from part4_ui.app import MainWindow
from part4_ui.lfi_core.data_types import LFIRawData


@pytest.mark.parametrize('language,heading,suffix', [('CN', '层数', '层'), ('EN', 'Slices', 'Slices')])
def test_slice_heading_and_suffix(qtbot, tmp_path, language, heading, suffix):
    window = MainWindow(QtCore.QSettings(str(tmp_path/'settings.ini'), QtCore.QSettings.Format.IniFormat))
    qtbot.addWidget(window)
    # Shape-only view avoids allocating a real scan; this test verifies text.
    values = np.broadcast_to(np.array(1.0), (70, 1400, 400))
    window.data = LFIRawData(values, values, list(range(530, 600)), 'Chip', 'chip_borad1.hdf5')
    window.set_language(language)
    assert window.loaded_status() == f'chip_borad1.hdf5 · 400 × 1400 · 70 {suffix}'
    assert f'{heading}: 70' in window.scan_label.text().splitlines()
    expected_position = '第 1 / 70 张切片 · Z = 530' if language == 'CN' else 'Slice 1 of 70 · Z = 530'
    assert window.z_label.text() == expected_position
