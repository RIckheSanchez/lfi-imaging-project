import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from PIL import Image
from PySide6 import QtCore, QtWidgets
from part4_ui.app import MainWindow
from part4_ui.lfi_core.adapters.part2 import Part2Preprocessor
from part4_ui.lfi_core.data_types import LFIRawData
from part4_ui.lfi_core.processing import normalized_slice, finite_values, prepare_3d
from part4_ui.lfi_core.export import export_csv, pixel_image, publication_png

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('converted,units,expected', [(False,'radians',False),(True,'raw_codes',False),(True,'radians',True)])
def test_part2_adapter_preserves_identity_and_metadata(monkeypatch, converted, units, expected):
    import part2_preprocess.pipeline as pipeline
    raw = LFIRawData(np.ones((1,3,4)),np.zeros((1,3,4)),[21],'Vegetables','original.hdf5',
                     {'phase_converted':converted,'other':{'complete':1}})
    corrected = np.full((1,3,4),3.)
    meta = {'phase_units':units,'spatial_products':{'amplitude_corrected':corrected},
            'config_resolved':{'normalisation':{'method':'zscore'}},'extra':'keep'}
    def preprocess(scan, config_path, config_overrides):
        assert scan is raw
        assert Path(config_path).is_absolute() and Path(config_path).samefile(ROOT/'part2_preprocess/config/default.toml')
        assert config_overrides == {'noise_reduction':{'method':'none'}}
        return SimpleNamespace(amplitude=np.zeros_like(corrected),phase=raw.phase,metadata=meta)
    monkeypatch.setattr(pipeline,'preprocess_part1',preprocess)
    result = Part2Preprocessor().preprocess(raw,{'noise_reduction':{'method':'none'}})
    assert result.amplitude is corrected
    assert (result.file_path,result.target_type,result.z_indices)==('original.hdf5','Vegetables',[21])
    assert result.metadata['other'] == raw.metadata['other']
    assert result.metadata['part2_preprocessing'] is meta
    assert result.metadata['phase_converted'] is expected
    assert result.metadata['ui_preprocessing']['amplitude_product']=='amplitude_corrected'
    assert result.metadata['ui_preprocessing']['amplitude_units']=='dimensionless'
    assert raw.metadata['phase_converted'] is converted


def test_nan_processing_and_exports():
    z,y,x=np.mgrid[:12,:18,:20]
    cube=(z*7+y*2+x).astype(float)
    cube[0,0,0]=np.nan
    scan=LFIRawData(cube,cube.copy(),list(range(12)),'Synthetic','synthetic',{})
    normalized,low,high=normalized_slice(scan,0)
    assert np.isnan(normalized[0,0]) and low==1 and high==53
    assert finite_values(cube).size==cube.size-1
    recovered=np.loadtxt(io.StringIO(export_csv(cube[0])),delimiter=',')
    assert np.array_equal(recovered,cube[0],equal_nan=True)
    image=Image.open(io.BytesIO(pixel_image(normalized,'Viridis')))
    assert image.getpixel((0,0))[3]==0
    assert publication_png(normalized,'Viridis').startswith(b'\x89PNG')
    volume=prepare_3d(scan)
    assert volume['rgba'][0,0,0,3]==0
    assert volume['values_sum']==np.nansum(cube)
    iso=prepare_3d(scan,'isosurface')
    assert all(np.isfinite(vertices).all() for vertices,_,_ in iso['meshes'])
    scan.amplitude[:]=np.nan
    for operation in (lambda:normalized_slice(scan,0),lambda:prepare_3d(scan),
                      lambda:export_csv(scan.amplitude[0]),lambda:pixel_image(scan.amplitude[0],'Viridis')):
        with pytest.raises(ValueError,match='no finite values'):
            operation()


@pytest.fixture
def adapter_window(qtbot,tmp_path,monkeypatch):
    window=MainWindow(QtCore.QSettings(str(tmp_path/'ui.ini'),QtCore.QSettings.Format.IniFormat))
    window.errors=[]
    monkeypatch.setattr(QtWidgets.QMessageBox,'warning',lambda parent,title,message:window.errors.append(message))
    qtbot.addWidget(window)
    window.show()
    yield window
    qtbot.waitUntil(lambda:window.jobs==0,timeout=60000)
    window.close()


def ready(qtbot,window):
    qtbot.waitUntil(lambda:window.jobs==0,timeout=60000)
    assert not window.errors
    assert window.raw_slice is not None


@pytest.mark.parametrize('name',['chip_borad1.hdf5','mushroom.hdf5','pure marble_29.hdf5'])
def test_real_part2_toggle_worker_and_labels(qtbot,adapter_window,monkeypatch,name):
    window=adapter_window
    backend=Part2Preprocessor()
    original=backend.preprocess
    calls=[]
    def observed(data,params):
        start=time.perf_counter()
        result=original(data,params)
        calls.append({'thread':threading.get_ident(),'seconds':time.perf_counter()-start})
        return result
    monkeypatch.setattr(backend,'preprocess',observed)
    window.preprocessor=backend
    window.pre_group.show()
    start=time.perf_counter();window.open_file(str(ROOT/'Realdata'/name));ready(qtbot,window)
    load_elapsed=time.perf_counter()-start
    output=ROOT/'docs/screenshots/v12-labels';output.mkdir(parents=True,exist_ok=True)
    slug=Path(name).stem.replace(' ','_')
    assert window.amplitude_label()=='Amplitude'
    window.grab().save(str(output/(slug+'-off.png')))
    start=time.perf_counter();window.apply_preprocessing.setChecked(True);ready(qtbot,window)
    enable_elapsed=time.perf_counter()-start
    assert calls and calls[0]['thread']!=threading.get_ident()
    assert window.data.metadata['phase_converted'] is False
    assert window.data.metadata['ui_preprocessing']['amplitude_product']=='output.amplitude'
    assert window.data.file_path==window.raw_data.file_path
    expected='Amplitude (pre-processed, dimensionless)' if name.startswith('chip') else 'Amplitude (pre-processed)'
    assert window.colorbar.axis.labelText==window.amplitude_label()==expected
    assert window.histogram.getAxis('bottom').labelText==expected
    window.grab().save(str(output/(slug+'-on.png')))
    start=time.perf_counter();window.apply_preprocessing.setChecked(False);ready(qtbot,window)
    disable_elapsed=time.perf_counter()-start
    assert window.data is window.raw_data and window.amplitude_label()=='Amplitude'
    (output/(slug+'-timing.json')).write_text(json.dumps({'file':name,'load_unprocessed_seconds':load_elapsed,
        'enable_preprocessing_seconds':enable_elapsed,'disable_preprocessing_seconds':disable_elapsed,
        'worker_preprocess_seconds':calls[0]['seconds'],'off_label':'Amplitude','on_label':expected},indent=2))


def test_nan_gui_histogram_colorbar_and_pixel(qtbot,adapter_window,tmp_path):
    window=adapter_window
    path=tmp_path/'nan.hdf5'
    values=np.arange(60,dtype=float).reshape(3,4,5);values[0,0,0]=np.nan
    with h5py.File(path,'w') as f:
        for z in range(3):
            f[f'amplitude/{z}']=values[z];f[f'phase/{z}']=values[z]
    window.open_file(path);ready(qtbot,window)
    assert window.colorbar.levels()==(1.,19.)
    window.select_pixel(0,0,switch_tab=False)
    assert 'Missing (NaN)' in window.pixel_label.text()
    assert window.histogram_curve.getData()[1].sum()==19
    window.data.phase[:]=np.nan
    window.channel.setCurrentIndex(1)
    qtbot.waitUntil(lambda:window.jobs==0,timeout=60000)
    assert any('no finite values' in error for error in window.errors)


@pytest.fixture
def frame_folder(tmp_path):
    folder=tmp_path/'frames';folder.mkdir()
    for frame in (21,22,30):
        with h5py.File(folder/f'frame-{frame}.hdf5','w') as file:
            file[f'amplitude/{frame}']=np.full((4,5),frame,dtype=float)
            file[f'phase/{frame}']=np.full((4,5),-frame,dtype=float)
    return folder


def test_folder_loader_public_api_and_identity(frame_folder):
    from part4_ui.lfi_core.loaders import load_path, get_loaders
    from part4_ui.lfi_core.data_io import smart_load
    assert {loader.name for loader in get_loaders()} >= {'hdf5','frame-folder'}
    scan,key=load_path(frame_folder)
    assert scan.amplitude.shape==(3,4,5)
    assert scan.z_indices==[21,22,30]
    assert key[0]==str(frame_folder.resolve())
    assert all(len(entry)==3 and isinstance(entry[1],int) for entry in key[2])
    file=frame_folder/'frame-21.hdf5'
    direct=smart_load(str(file));dispatched,_=load_path(file)
    assert np.array_equal(direct.amplitude,dispatched.amplitude)
    old_key=key
    import os
    stat=file.stat();os.utime(file,ns=(stat.st_atime_ns,stat.st_mtime_ns+1000000))
    assert load_path(frame_folder)[1]!=old_key
    (frame_folder/'experiment_details.txt').write_text('z_start_mm: 1\nz_step_mm: 0.1\n')
    assert len(load_path(frame_folder)[1][2])==4


def test_open_folder_dialog_recent_and_dispatch(qtbot,adapter_window,frame_folder,monkeypatch):
    window=adapter_window
    monkeypatch.setattr(QtWidgets.QFileDialog,'getExistingDirectory',lambda *args:str(frame_folder))
    window.folder_action.trigger();ready(qtbot,window)
    assert window.data.z_indices==[21,22,30]
    assert window.settings.value('recent_files',[],type=list)[0]==str(frame_folder)
    assert window.accepts_path(str(frame_folder))
    window.open_file(str(ROOT/'Realdata/mushroom.hdf5'));ready(qtbot,window)
    assert window.data.width==1300
    next(action for action in window.recent_menu.actions() if action.text()==str(frame_folder)).trigger()
    ready(qtbot,window)
    assert window.data.z_indices==[21,22,30]
    window.set_language('CN')
    assert window.folder_action.text()=='打开文件夹…'


def test_registered_extra_loader_dispatch(tmp_path,monkeypatch):
    import part4_ui.lfi_core.loaders as module
    monkeypatch.setattr(module,'_loaders',dict(module._loaders))
    class Extra:
        name='extra';path_types=('file',);extensions=('.extra',)
        def load(self,path):return LFIRawData(np.ones((1,2,3)),np.zeros((1,2,3)),[0],'Synthetic',str(path),{})
        def make_data_key(self,path):return ('version-1',Path(path).stat().st_size)
    module.register_loader(Extra())
    path=tmp_path/'sample.extra';path.write_text('fixture')
    data,key=module.load_path(path)
    assert data.amplitude.shape==(1,2,3) and key[1]=='extra'
    with pytest.raises(ValueError,match='already registered'):module.register_loader(Extra())


@pytest.mark.parametrize('registered,depth,visible',[(False,[1,2,3],False),(True,None,False),(True,[1,2],False),(True,[1,2,3],True)])
def test_reconstruction_requires_backend_and_matching_depth(qtbot,tmp_path,monkeypatch,registered,depth,visible):
    from part4_ui.lfi_core.backend import register_reconstructor
    class Fake:
        def reconstruct(self,data,params):return data
    register_reconstructor(Fake() if registered else None)
    try:
        window=MainWindow(QtCore.QSettings(str(tmp_path/'depth.ini'),QtCore.QSettings.Format.IniFormat))
        qtbot.addWidget(window);window.show()
        window.data=LFIRawData(np.ones((3,4,5)),np.zeros((3,4,5)),[0,1,2],'Synthetic','fixture',{})
        if depth is not None:window.data.metadata['depth_mm']=depth
        window.update_info()
        assert window.volume_panel.reconstruct.isHidden() is not visible
        assert window.volume_panel.reconstruction_hint.isHidden() is visible
        assert window.volume_panel.reconstruction_hint.text()=='Reconstruction needs slice depth calibration (depth_mm).'
        if visible:
            window.volume_panel.reconstruct.setChecked(True)
            window.data.metadata.pop('depth_mm')
            window.update_info()
            assert window.volume_panel.reconstruct.isHidden()
            assert not window.volume_panel.reconstruct.isChecked()
        window.set_language('CN')
        assert window.volume_panel.reconstruction_hint.text()=='重建需要层深度标定（depth_mm）。'
        window.close()
    finally:register_reconstructor(None)
