import ast, io, sys, time, gc, json, platform, subprocess, tempfile
import h5py
import hashlib
from unittest.mock import patch
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'files'))
source=root/'files/lfi_viewer_final.py'
tree=ast.parse(source.read_text())
# Execute the real application helpers without running its Streamlit UI.
nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef,ast.Try))]
namespace={}
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace)
data=namespace['smart_load'](str(root/'Realdata/mushroom.hdf5'))
for channel in ('amplitude','phase'):
 raw=data.get_slice(data.z_indices[0])[channel]
 recovered=np.loadtxt(io.StringIO(namespace['export_csv'](raw)),delimiter=',')
 assert raw.shape==recovered.shape
 assert np.allclose(raw,recovered)
 print('CSV PASS',channel,raw.shape)
X,Y,Z,values,step=namespace['get_downsampled_volume'].__wrapped__(data, (data.file_path, Path(data.file_path).stat().st_mtime, Path(data.file_path).stat().st_size))
trace=namespace['create_3d_trace'](X,Y,Z,values,'Viridis','isosurface')
assert trace.type=='isosurface' and trace.surface.count==3
assert trace.isomin==np.percentile(values,60) and trace.isomax==np.percentile(values,95)
assert not trace.caps.x.show and not trace.caps.y.show and not trace.caps.z.show
print('ISOSURFACE PASS (single plane mushroom; object validation only)')
raw=data.amplitude[0]; norm=(raw-raw.min())/(raw.max()-raw.min()+1e-10)
for cmap in ('Viridis','Cool','ice','missing'):
 png=namespace['publication_png'](norm,cmap,10)
 dpi=Image.open(io.BytesIO(png)).info['dpi']
 assert np.allclose(dpi,(300,300),atol=.01)
print('PUBLICATION DPI PASS',dpi)
from streamlit.testing.v1 import AppTest
app=AppTest.from_file(str(source),default_timeout=30).run()
app.checkbox[0].check().run()
app.text_input[0].set_value(str(root/'Realdata/mushroom.hdf5')).run()
next(b for b in app.button if b.label=='📥 Load from Path').click().run()
assert not app.exception, app.exception
assert any(n.label=='Pixel X' for n in app.number_input)
next(n for n in app.number_input if n.label=='Pixel size (μm)').set_value(10.).run()
assert not app.exception,app.exception
print('APPTEST PASS')
# Capture results from the actual cached helpers inside an AppTest runtime.
capture_source = source.read_text() + """
if st.session_state.data is not None:
    _, _, _, cached_values, _ = get_downsampled_volume(st.session_state.data, data_key)
    st.session_state['test_volume_sum'] = float(cached_values.sum())
    st.session_state['test_slice'] = get_normalized_slice(
        st.session_state.data, data_key, current_z, channel_key)[0]
    st.session_state['test_z'] = current_z
"""
# Exercise the real uploader/button path. Isolate its system temp directory,
# and track writes so reruns cannot silently create and delete extra files.
with tempfile.TemporaryDirectory(prefix='lfi-upload-test-') as upload_directory:
    with patch.object(tempfile, 'tempdir', upload_directory), patch.object(
            tempfile, 'NamedTemporaryFile', wraps=tempfile.NamedTemporaryFile) as temporary_file:
        upload_app = AppTest.from_string(capture_source, default_timeout=60).run()
        mushroom_bytes = (root/'Realdata/mushroom.hdf5').read_bytes()
        upload_app.file_uploader[0].set_value(
            ('mushroom.hdf5', mushroom_bytes, 'application/x-hdf5')).run()
        assert not upload_app.exception, upload_app.exception
        assert temporary_file.call_count == 0
        next(b for b in upload_app.button if b.label=='📥 Load Uploaded File').click().run()
        assert not upload_app.exception, upload_app.exception
        expected_key = ('upload', 'mushroom.hdf5', len(mushroom_bytes),
                        hashlib.sha1(mushroom_bytes).hexdigest())
        assert upload_app.session_state['data_key'] == expected_key
        assert temporary_file.call_count == 1
        assert not Path(upload_app.session_state['data'].file_path).exists()
        assert not list(Path(tempfile.gettempdir()).glob('*.hdf5'))
        upload_sum = upload_app.session_state['test_volume_sum']
        upload_slice = upload_app.session_state['test_slice'].copy()
        for brightness in (61, 62):
            next(slider for slider in upload_app.slider if slider.label=='Brightness').set_value(brightness).run()
            assert not upload_app.exception, upload_app.exception
            assert upload_app.session_state['data_key'] == expected_key
            assert upload_app.session_state['test_volume_sum'] == upload_sum
            assert np.array_equal(upload_app.session_state['test_slice'], upload_slice)
            assert not list(Path(tempfile.gettempdir()).glob('*.hdf5'))
        assert temporary_file.call_count == 1
        print('UPLOAD RERUN PASS: 2 brightness reruns, stable key, 1 temporary write')
        upload_app.checkbox[0].check().run()
        chip_path = (root/'Realdata/chip_borad1.hdf5').resolve()
        upload_app.text_input[0].set_value(str(chip_path)).run()
        next(b for b in upload_app.button if b.label=='📥 Load from Path').click().run()
        assert not upload_app.exception, upload_app.exception
        assert upload_app.session_state['data_key'] == (
            str(chip_path), chip_path.stat().st_mtime, chip_path.stat().st_size)
        switched_data = upload_app.session_state['data']
        _, _, _, expected_volume, _ = namespace['get_downsampled_volume'].__wrapped__(
            switched_data, upload_app.session_state['data_key'])
        expected_normalized = namespace['get_normalized_slice'].__wrapped__(
            switched_data, upload_app.session_state['data_key'], switched_data.z_indices[0])[0]
        assert upload_app.session_state['test_volume_sum'] == float(expected_volume.sum())
        assert upload_sum != upload_app.session_state['test_volume_sum']
        assert np.allclose(upload_app.session_state['test_slice'], expected_normalized)
        assert upload_slice.shape != upload_app.session_state['test_slice'].shape
        assert temporary_file.call_count == 1
        assert not list(Path(tempfile.gettempdir()).glob('*.hdf5'))
        print('UPLOAD TO PATH CACHE PASS 3D', upload_sum, '->', upload_app.session_state['test_volume_sum'])
        print('UPLOAD TO PATH CACHE PASS 2D', upload_slice.shape, '->', expected_normalized.shape)
        print('UPLOAD CLEANUP PASS: no .hdf5 files remain in isolated system temp directory')
        # A failed load must also remove its temporary file.
        try:
            namespace['load_uploaded_bytes']('invalid.hdf5', b'invalid HDF5')
        except OSError:
            pass
        else:
            raise AssertionError('Invalid upload unexpectedly loaded')
        assert not list(Path(tempfile.gettempdir()).glob('*.hdf5'))
        print('UPLOAD FAILURE CLEANUP PASS')

cache_app = AppTest.from_string(capture_source, default_timeout=60).run()
cache_app.checkbox[0].check().run()
cache_app.text_input[0].set_value(str(root/'Realdata/chip_borad1.hdf5')).run()
next(b for b in cache_app.button if b.label=='📥 Load from Path').click().run()
assert not cache_app.exception, cache_app.exception
# Chip uses z=530..599. Relabel only its in-memory test copy to 0..69
# so the two files collide at exactly the same z cache argument (0).
cache_app.session_state['data'].z_indices = list(range(70))
cache_app.run()
assert not cache_app.exception, cache_app.exception
chip_sum = cache_app.session_state['test_volume_sum']
chip_slice = cache_app.session_state['test_slice'].copy()
assert cache_app.session_state['test_z'] == 0
with tempfile.TemporaryDirectory() as directory:
    temporary_path = Path(directory)/'cache-regression.hdf5'
    height, width = chip_slice.shape
    test_slice = np.arange(height * width, dtype=float).reshape(height, width)
    with h5py.File(temporary_path, 'w') as handle:
        for index in range(5):
            handle.create_dataset(f'amplitude/{index}', data=test_slice + 1000000 + index)
            handle.create_dataset(f'phase/{index}', data=-test_slice - index)
    cache_app.text_input[0].set_value(str(temporary_path)).run()
    next(b for b in cache_app.button if b.label=='📥 Load from Path').click().run()
    assert not cache_app.exception, cache_app.exception
    new_sum = cache_app.session_state['test_volume_sum']
    new_slice = cache_app.session_state['test_slice']
    assert cache_app.session_state['test_z'] == 0
    assert chip_sum != new_sum
    assert chip_slice.shape == new_slice.shape
    assert not np.allclose(chip_slice, new_slice)
    expected_slice = (test_slice - test_slice.min()) / (test_slice.max() - test_slice.min() + 1e-10)
    assert np.allclose(new_slice, expected_slice)
    print('CACHE REGRESSION PASS 3D', chip_sum, '->', new_sum)
    print('CACHE REGRESSION PASS 2D same z=0, shape=', new_slice.shape)

times=[]
for i in range(5):
 gc.collect()
 start=time.perf_counter()
 loaded=namespace['smart_load'](str(root/'Realdata/chip_borad1.hdf5'))
 elapsed=time.perf_counter()-start
 assert loaded.amplitude.shape==(70,1400,400)
 times.append(elapsed)
 del loaded
 print('TIMING',i+1,elapsed)
print('RESULT',json.dumps({'seconds':times,'mean':float(np.mean(times)),'population_std':float(np.std(times)),'sample_std':float(np.std(times,ddof=1)),'python':sys.version,'platform':platform.platform()}))
print(subprocess.check_output(['system_profiler','SPHardwareDataType'],text=True).split('Serial Number')[0])
