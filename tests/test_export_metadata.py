"""CSV provenance is optional and never changes the existing numeric matrix."""
from copy import deepcopy
from datetime import datetime
import io
import numpy as np
import pytest
from part4_ui import APP_NAME, __version__
from part4_ui.lfi_core.data_types import LFIRawData
from part4_ui.lfi_core.export import (export_csv, scan_export_metadata,
    view_export_metadata, view_png_filename, format_export_metadata)


@pytest.fixture
def scan(tmp_path):
    values = np.arange(40, dtype=float).reshape(2, 4, 5)
    return LFIRawData(values, -values, [530, 545], 'Chip, sample', str(tmp_path/'chip,board.hdf5'), {
        'phase_encoding_used': 'test,encoding', 'phase_converted': False,
        'ui_preprocessing': {'normalisation': 'zscore', 'amplitude_units': 'dimensionless',
            'channels': {'amplitude': {'processed': True},
                         'phase': {'processed': False, 'unwrapping_status': 'withheld_unconfirmed_encoding'}}}})


def test_csv_optional_header_preserves_numeric_bytes_and_roundtrips(tmp_path, scan):
    values = np.array([[1.234567891, -2.345678912, np.nan], [0., 1e20, 1e-20]])
    old = io.StringIO()
    np.savetxt(old, values, delimiter=',', fmt='%.6g')
    assert export_csv(values) == export_csv(values, None) == old.getvalue()
    metadata = scan_export_metadata(scan, 1, 'phase')
    content = export_csv(values, metadata)
    header = [line for line in content.splitlines(keepends=True) if line.startswith('# ')]
    matrix = ''.join(line for line in content.splitlines(keepends=True) if not line.startswith('# '))
    assert matrix.encode('utf-8') == old.getvalue().encode('utf-8')
    assert header and all(',' not in line for line in header)
    path = tmp_path/'phase.csv'
    path.write_text(content, encoding='utf-8')
    expected = np.loadtxt(io.StringIO(old.getvalue()), delimiter=',')
    assert np.array_equal(np.loadtxt(path, delimiter=','), expected, equal_nan=True)
    assert '# Source file: chip;board.hdf5' in content
    assert '# Target type: Chip; sample' in content
    assert '# Channel: phase' in content
    assert '# Values: raw' in content
    assert '# Slice index (zero-based): 1' in content
    assert '# Z value: 545' in content
    assert '# Shape: 4 rows (Y) x 5 columns (X)' in content
    assert '# Phase encoding: test;encoding' in content
    assert '# Unwrap status: withheld_unconfirmed_encoding' in content
    assert '# Numeric format: %.6g' in content
    assert '# App name: ' + APP_NAME in content
    assert '# App version: ' + __version__ in content
    stamp = datetime.fromisoformat(metadata['Export timestamp'])
    assert stamp.utcoffset() is not None


def test_csv_pandas_reader_when_installed(tmp_path, scan):
    pandas = pytest.importorskip('pandas')
    path = tmp_path/'amplitude.csv'
    path.write_text(export_csv(scan.amplitude[1], scan_export_metadata(scan, 1, 'amplitude')), encoding='utf-8')
    frame = pandas.read_csv(path, comment='#', header=None)
    assert frame.shape == (4, 5)
    assert np.array_equal(frame.to_numpy(), scan.amplitude[1])


def test_channel_specific_units_and_processing_status(scan):
    original = deepcopy(scan.metadata)
    amplitude = scan_export_metadata(scan, 0, 'amplitude')
    phase = scan_export_metadata(scan, 0, 'phase')
    assert amplitude['Values'] == 'pre-processed'
    assert amplitude['Pre-processing status'] == 'applied'
    assert amplitude['Normalisation'] == 'zscore'
    assert amplitude['Units'] == 'dimensionless'
    assert phase['Values'] == 'raw'
    assert phase['Pre-processing status'] == 'unchanged (pipeline enabled)'
    assert phase['Normalisation'].startswith('none')
    assert phase['Units'] == 'unconfirmed (raw input units)'
    assert scan.metadata == original
    scan.metadata['phase_converted'] = True
    scan.metadata['ui_preprocessing']['channels']['phase'].update(processed=True, unwrapping_status='unwrapped')
    phase = scan_export_metadata(scan, 0, 'phase')
    assert phase['Values'] == 'pre-processed' and phase['Units'] == 'radians'
    assert phase['Unwrap status'] == 'unwrapped'
    scan.metadata.pop('ui_preprocessing')
    raw = scan_export_metadata(scan, 0, 'amplitude')
    assert raw['Values'] == 'raw' and raw['Normalisation'] == 'none'
    assert raw['Pre-processing status'] == 'not applied'


def test_header_values_cannot_add_rows_or_columns():
    content = format_export_metadata({'Source,file': 'scan,one\r\n# forged: data', 'Units': 'mV, RMS'})
    assert content.splitlines() == ['# Source;file: scan;one # forged: data', '# Units: mV; RMS']


@pytest.mark.parametrize('mode', ['volume', 'isosurface'])
def test_3d_filename_and_sidecar(scan, mode):
    result = {'mode': mode, 'step': 5, 'original_shape': (70, 1400, 400),
              'sampled_points': 313600, 'percentile_range': (90, 99.5), 'thresholds': (12.25, 99.75)}
    assert view_png_filename(scan.file_path, mode) == f'lfi_3d_{mode}_chip,board.png'
    metadata = view_export_metadata(scan, 1, result, timestamp='2026-11-04T10:00:00+10:00')
    assert metadata['Channel'] == 'amplitude'  # Never the selected 2D phase channel.
    assert metadata['3D mode'] == mode
    assert metadata['Downsampling step'] == 5
    assert metadata['Percentile thresholds'] == (90, 99.5)
    assert metadata['Threshold values'] == (12.25, 99.75)
    assert metadata['Rendered volume shape (Z Y X)'] == '70 x 1400 x 400'
    content = format_export_metadata(metadata)
    assert '# App version: ' + __version__ in content
    assert '# Export timestamp: 2026-11-04T10:00:00+10:00' in content
    assert '# Percentile thresholds: (90; 99.5)' in content
    assert '# Threshold values: (12.25; 99.75)' in content
    assert all(line.startswith('# ') and ',' not in line for line in content.splitlines())
