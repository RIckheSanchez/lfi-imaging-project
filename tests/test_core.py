import io
from pathlib import Path
import hashlib
import os
import numpy as np
import h5py
from PIL import Image
import pytest
from part4_ui.lfi_core.data_io import smart_load, local_data_key, upload_data_key, load_uploaded_bytes
from part4_ui.lfi_core.processing import normalized_slice, adjust_slice, downsampled_volume, isosurface_thresholds
from part4_ui.lfi_core.export import export_csv, pixel_image, publication_png, scale_bar
from part4_ui.lfi_core.backend import IdentityPreprocessor, IdentityReconstructor

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def mushroom():
    return smart_load(str(ROOT/'Realdata/mushroom.hdf5'))

@pytest.mark.parametrize('channel', ['amplitude', 'phase'])
def test_csv_roundtrip(mushroom, channel):
    raw = mushroom.get_slice(40)[channel]
    recovered = np.loadtxt(io.StringIO(export_csv(raw)), delimiter=',')
    assert recovered.shape == raw.shape
    assert np.allclose(raw, recovered)

@pytest.mark.parametrize('image_format', ['PNG', 'TIFF'])
def test_pixel_dpi(mushroom, image_format):
    normalized, _, _ = normalized_slice(mushroom, 40)
    image = Image.open(io.BytesIO(pixel_image(adjust_slice(normalized), 'Cool', image_format)))
    assert image.size == (1300, 400)
    assert np.allclose(image.info['dpi'], (300, 300), atol=.01)

@pytest.mark.parametrize('colormap', ['Viridis', 'Cool', 'ice', 'missing'])
def test_publication_dpi(mushroom, colormap):
    normalized, _, _ = normalized_slice(mushroom, 40)
    image = Image.open(io.BytesIO(publication_png(normalized, colormap, 10)))
    assert np.allclose(image.info['dpi'], (300, 300), atol=.01)


def test_volume_and_thresholds(mushroom):
    x, y, z, values, step = downsampled_volume(mushroom)
    assert x.shape == y.shape == z.shape == values.shape
    assert np.array_equal(values, mushroom.amplitude[::step, ::step, ::step])
    assert isosurface_thresholds(values) == tuple(np.percentile(values, [90, 99.5]))


def test_identity_backends(mushroom):
    assert IdentityPreprocessor().preprocess(mushroom, {}) is mushroom
    assert IdentityReconstructor().reconstruct(mushroom, {}) is mushroom


def test_source_switch_and_keys(tmp_path, mushroom):
    path = tmp_path/'different.hdf5'
    grid = np.arange(400*1300, dtype=float).reshape(400, 1300)
    with h5py.File(path, 'w') as handle:
        for index in range(5):
            handle.create_dataset(f'amplitude/{40+index}', data=grid + index + 1e6)
            handle.create_dataset(f'phase/{40+index}', data=-grid)
    other = smart_load(str(path))
    assert local_data_key(path) != local_data_key(mushroom.file_path)
    assert not np.allclose(normalized_slice(mushroom, 40)[0], normalized_slice(other, 40)[0])
    assert downsampled_volume(mushroom)[3].sum() != downsampled_volume(other)[3].sum()
    old_key = local_data_key(path)
    os.utime(path, (old_key[1] + 10, old_key[1] + 10))
    assert local_data_key(path) != old_key
    assert upload_data_key('a.hdf5', b'abc') != upload_data_key('a.hdf5', b'abd')
    assert upload_data_key('a.hdf5', b'abc') == ('upload', 'a.hdf5', 3, hashlib.sha1(b'abc').hexdigest())


def test_upload_cleanup_and_identity(tmp_path, monkeypatch):
    monkeypatch.setattr('tempfile.tempdir', str(tmp_path))
    content = (ROOT/'Realdata/mushroom.hdf5').read_bytes()
    data, key = load_uploaded_bytes('mushroom.hdf5', content)
    assert key == upload_data_key('mushroom.hdf5', content)
    assert data.target_type == 'Mushroom'
    assert not Path(data.file_path).exists()
    assert not list(tmp_path.glob('*.hdf5'))
    with pytest.raises(OSError):
        load_uploaded_bytes('invalid.hdf5', b'invalid')
    assert not list(tmp_path.glob('*.hdf5'))


def test_scale_and_adjustments():
    assert scale_bar(400, 10) == (100, '1 mm')
    assert np.allclose(adjust_slice(np.array([0., .5, 1.]), 50, 50), [0, .5, 1])


@pytest.fixture
def synthetic_volume():
    from part4_ui.lfi_core.data_types import LFIRawData
    z, y, x = np.mgrid[:12, :18, :20]
    values = (z*7 + y*2 + x).astype(float)
    return LFIRawData(values, -values, list(range(12)), 'Synthetic', 'synthetic.hdf5')


def test_volume_axis_order_and_rgba(synthetic_volume):
    from part4_ui.lfi_core.processing import prepare_3d
    result = prepare_3d(synthetic_volume, 'volume', brightness=50, contrast=50)
    assert result['rgba'].shape == (20, 18, 12, 4)
    assert result['rgba'].dtype == np.uint8
    assert result['values_sum'] == synthetic_volume.amplitude.sum()
    assert result['rgba'][0,0,0,3] == 0
    assert result['rgba'][-1,-1,-1,3] > 0


def test_three_uncapped_isosurfaces(synthetic_volume):
    from part4_ui.lfi_core.processing import prepare_3d
    result = prepare_3d(synthetic_volume, 'isosurface')
    assert result['thresholds'] == tuple(np.percentile(synthetic_volume.amplitude, [90,99.5]))
    assert len(result['meshes']) == 3
    assert [color[3] for _,_,color in result['meshes']] == [.12,.30,.60]
    for vertices, faces, color in result['meshes']:
        assert vertices.shape[1] == faces.shape[1] == 3
        assert faces.max() < len(vertices)
        assert np.all(vertices >= 0)
        assert np.all(vertices <= [19,17,11])
        assert color[3] in (.12,.30,.60)


def test_3d_degenerate_and_single_layer(mushroom, synthetic_volume):
    from part4_ui.lfi_core.processing import prepare_3d
    with pytest.raises(ValueError, match='at least two'):
        prepare_3d(mushroom)
    synthetic_volume.amplitude[:] = 1
    with pytest.raises(ValueError, match='distinct'):
        prepare_3d(synthetic_volume, 'isosurface')



def test_volume_robust_clipping_and_nonlinear_opacity(synthetic_volume):
    from part4_ui.lfi_core.processing import prepare_3d
    values = synthetic_volume.amplitude
    values[0,0,0], values[-1,-1,-1] = -1e9, 1e9
    result = prepare_3d(synthetic_volume, brightness=50, contrast=50)
    low, high = np.percentile(values, [1,99.5])
    assert result['normalization_range'] == (low,high)
    alpha = result['rgba'][...,3].transpose(2,1,0)
    assert np.all(alpha[values <= low] == 0)
    assert np.all(alpha[values >= high] == int(.35*255))
    middle = (values-low)/(high-low)
    mask = (middle > .4) & (middle < .6)
    assert np.all(alpha[mask] < (middle[mask]*.35*255).astype(np.uint8))
    assert values[-1,-1,-1] == 1e9



@pytest.mark.parametrize('percentile', [-1,99.5,100])
def test_isosurface_invalid_percentile(synthetic_volume, percentile):
    with pytest.raises(ValueError,match='below 99.5'):
        isosurface_thresholds(synthetic_volume.amplitude, percentile)



def test_phase_semantics_unknown_and_confirmed():
    from part4_ui.lfi_core.phase_semantics import phase_label_key, phase_encoding_status
    assert phase_label_key({}) == 'phase_raw'
    assert phase_encoding_status({}) == 'unknown'
    assert phase_label_key({'phase_converted': False}) == 'phase_raw'
    assert phase_label_key({'phase_converted': True}) == 'phase_rad'
    assert phase_label_key({'phase_converted': 'False'}) == 'phase_raw'


@pytest.mark.parametrize('name', ['chip_borad1.hdf5', 'mushroom.hdf5', 'pure marble_29.hdf5'])
def test_pre_migration_numeric_baseline(name):
    import json
    baseline = json.loads((ROOT/'tests/v11-loader-baseline.json').read_text())[name]
    data = smart_load(str(ROOT/'Realdata'/name))
    assert list(data.amplitude.shape) == baseline['shape']
    assert data.z_indices == baseline['z_indices']
    for channel in ('amplitude', 'phase'):
        canonical = np.ascontiguousarray(getattr(data, channel), dtype='<f8')
        assert hashlib.sha256(canonical.tobytes()).hexdigest() == baseline[channel+'_sha256']
    assert data.metadata['phase_converted'] is False


def test_uint16_display_and_raw_export(mushroom):
    from part4_ui.lfi_core.data_types import LFIRawData
    from part4_ui.lfi_core.processing import prepare_3d
    assert mushroom.phase.dtype == np.uint16
    raw = mushroom.phase[0]
    normalized, low, high = normalized_slice(mushroom, 40, 'phase')
    expected = (raw.astype(np.float64)-float(raw.min()))/(float(raw.max())-float(raw.min())+1e-10)
    assert normalized.dtype == np.float64
    assert np.array_equal(normalized, expected)
    assert np.isfinite(adjust_slice(normalized)).all()
    assert np.array_equal(np.loadtxt(io.StringIO(export_csv(raw)), delimiter=','), raw)
    cube = np.arange(125, dtype=np.uint16).reshape(5,5,5)*500
    data = LFIRawData(cube, cube, list(range(5)), 'Unknown', 'synthetic', {})
    assert downsampled_volume(data)[3].dtype == np.float64
    assert prepare_3d(data)['values_sum'] == float(cube.astype(np.float64).sum())


def test_public_loader_adapter_and_error(monkeypatch):
    import part4_ui.lfi_core.data_io as module
    from types import SimpleNamespace
    source = SimpleNamespace(amplitude=np.zeros((1,2,2)), phase=np.ones((1,2,2), dtype=np.uint16),
                             z_indices=[7], target_type='Mushroom', file_path='scan.hdf5',
                             metadata={'nested': {'anything': [1,2]}, 'phase_converted': False})
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return source
    monkeypatch.setattr(module, 'load_single_scan', load)
    result = module.smart_load('scan.hdf5')
    assert calls == [('scan.hdf5', {'include_waveform': False})]
    assert result.metadata == source.metadata
    assert result.phase is source.phase
    def fail(*args, **kwargs):
        raise ValueError('Part 1 original failure')
    monkeypatch.setattr(module, 'load_single_scan', fail)
    with pytest.raises(ValueError, match='Part 1 original failure'):
        module.smart_load('invalid.hdf5')
