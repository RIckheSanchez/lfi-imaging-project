from pathlib import Path
import threading
import time
import io
import numpy as np
import h5py
import pytest
from PIL import Image
from PySide6 import QtCore, QtWidgets
from part4_ui.app import MainWindow
from part4_ui.lfi_core.processing import normalized_slice, adjust_slice

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def window(qtbot, tmp_path, monkeypatch):
    settings = QtCore.QSettings(str(tmp_path/'settings.ini'), QtCore.QSettings.Format.IniFormat)
    widget = MainWindow(settings)
    widget.errors = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning', lambda parent, title, message: widget.errors.append(message))
    qtbot.addWidget(widget)
    widget.show()
    yield widget
    qtbot.waitUntil(lambda: widget.jobs == 0, timeout=60000)
    widget.close()


def wait_ready(qtbot, window):
    qtbot.waitUntil(lambda: window.jobs == 0, timeout=60000)
    assert not window.errors
    assert window.raw_slice is not None


def load_mushroom(qtbot, window):
    window.open_file(str(ROOT/'Realdata/mushroom.hdf5'))
    wait_ready(qtbot, window)


def test_chip_load_responsive_and_screenshot(qtbot, window, monkeypatch):
    import part4_ui.app as module
    original = module.smart_load
    load_times = []
    worker_threads = []
    ticks = []
    main_thread = threading.get_ident()
    def timed_load(path):
        worker_threads.append(threading.get_ident())
        start = time.perf_counter()
        data = original(path)
        load_times.append((start, time.perf_counter()))
        return data
    monkeypatch.setattr(module, 'smart_load', timed_load)
    timer = QtCore.QTimer(window)
    timer.setInterval(10)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    timer.start()
    window.open_file(str(ROOT/'Realdata/chip_borad1.hdf5'))
    assert window.progress.isVisible()
    wait_ready(qtbot, window)
    timer.stop()
    assert worker_threads[0] != main_thread
    start, end = load_times[0]
    load_ticks = [tick for tick in ticks if start <= tick <= end]
    assert len(load_ticks) >= 3
    assert window.data.amplitude.shape == (70, 1400, 400)
    assert window.image.axisOrder == 'row-major'
    assert window.view.state['yInverted']
    assert window.view.state['aspectLocked']
    assert window.z_spin.maximum() == 69
    assert np.array_equal(window.raw_slice, window.data.amplitude[0])
    screenshots = ROOT/'docs/screenshots'
    screenshots.mkdir(parents=True, exist_ok=True)
    qtbot.wait(100)
    assert window.grab().save(str(screenshots/'chip_desktop.png'))
    print(f'CHIP RESPONSIVE PASS: worker load={end-start:.6f}s; GUI timer ticks during load={len(load_ticks)}; worker thread differs from GUI thread')
    print('SCREENSHOT:', screenshots/'chip_desktop.png')
    window.tabs.setCurrentIndex(1)
    window.request_volume()
    wait_ready(qtbot, window)
    assert window.volume_panel.result['mode'] == 'volume'
    assert len(window.volume_panel.scene_items) == 1
    window.volume_panel.mode.setCurrentIndex(1)
    window.request_volume()
    wait_ready(qtbot, window)
    assert len(window.volume_panel.scene_items) == 3
    assert window.volume_panel.result['values_sum'] == 394085247.0
    print('CHIP 3D PASS: real chip volume buffer and 3 marching-cubes meshes created offscreen')



def test_hover_click_waveform_and_language(qtbot, window):
    load_mushroom(qtbot, window)
    # Zoom so one sensor pixel spans multiple screen pixels for exact QTest clicks.
    window.view.setRange(QtCore.QRectF(0, 0, 100, 100), padding=0)
    qtbot.wait(50)
    x, y = 10, 12
    scene_point = window.image.mapToScene(QtCore.QPointF(x + .5, y + .5))
    window.hover_pixel(scene_point)
    assert f'x={x}' in window.statusBar().currentMessage()
    assert f'{window.raw_slice[y,x]:.6g}' in window.statusBar().currentMessage()
    position = window.image_plot.mapFromScene(scene_point)
    qtbot.mouseClick(window.image_plot.viewport(), QtCore.Qt.MouseButton.LeftButton, pos=position)
    assert window.selected_pixel == (x, y)
    assert window.tabs.currentIndex() == 2
    assert window.cross_x.value() == x + .5
    assert window.cross_y.value() == y + .5
    curve = window.amplitude_plot.listDataItems()[0]
    assert np.array_equal(curve.yData, window.data.amplitude[:,y,x])
    window.set_language('CN')
    assert window.tabs.tabText(0) == '二维切片'
    assert window.labels['brightness'].text() == '亮度'
    assert '原始值' in window.pixel_label.text()
    window.set_language('EN')
    assert window.tabs.tabText(0) == '2D Slice'


def test_adjustments_phase_and_scale(qtbot, window):
    load_mushroom(qtbot, window)
    window.view.setRange(QtCore.QRectF(10, 20, 100, 100), padding=0)
    original_range = np.array(window.view.viewRange())
    window.channel.setCurrentIndex(1)
    window.brightness.setValue(35)
    window.contrast.setValue(90)
    wait_ready(qtbot, window)
    assert np.array_equal(window.raw_slice, window.data.phase[0])
    normalized = normalized_slice(window.data, 40, 'phase')[0]
    assert np.allclose(window.adjusted, adjust_slice(normalized, 35, 90))
    assert np.allclose(window.view.viewRange(), original_range)
    assert window.scale_item is None
    window.pixel_size.setValue(10)
    assert window.scale_item is not None
    assert window.scale_item.text.toPlainText() == '2 mm'
    window.pixel_size.setValue(0)
    assert window.scale_item is None
    assert window.scale_hint.isVisible()


def test_native_dialog_and_drop(qtbot, window, monkeypatch):
    path = str(ROOT/'Realdata/mushroom.hdf5')
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getOpenFileName', lambda *args: (path, ''))
    window.open_dialog()
    wait_ready(qtbot, window)
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(path)])
    class Drop:
        accepted = False
        def mimeData(self): return mime
        def acceptProposedAction(self): self.accepted = True
    event = Drop()
    window.dragEnterEvent(event)
    assert event.accepted
    window.dropEvent(event)
    wait_ready(qtbot, window)


def test_recent_five_and_settings(qtbot, window, tmp_path):
    paths = []
    for index in range(6):
        path = tmp_path/f'scan{index}.hdf5'
        with h5py.File(path, 'w') as handle:
            handle.create_dataset('amplitude/0', data=np.arange(20).reshape(4,5))
            handle.create_dataset('phase/0', data=np.zeros((4,5)))
        paths.append(str(path))
        window.open_file(str(path))
        wait_ready(qtbot, window)
    assert window.settings.value('recent_files', [], type=list) == paths[::-1][:5]
    assert len(window.recent_menu.actions()) == 5
    window.set_language('CN')
    assert window.settings.value('language') == 'CN'


def test_loading_error_message(qtbot, window, tmp_path):
    window.open_file(str(tmp_path/'missing.hdf5'))
    qtbot.waitUntil(lambda: window.jobs == 0, timeout=10000)
    assert len(window.errors) == 1
    assert 'Could not load' in window.errors[0]
    assert window.data is None
    assert not window.progress.isVisible()


@pytest.mark.parametrize('image_format', ['CSV', 'PNG', 'TIFF', 'publication'])
def test_worker_exports(qtbot, window, tmp_path, image_format):
    load_mushroom(qtbot, window)
    path = tmp_path/'export'
    window.export_to(path, image_format)
    assert window.progress.isVisible()
    wait_ready(qtbot, window)
    if image_format == 'CSV':
        recovered = np.loadtxt(path, delimiter=',')
        assert np.allclose(recovered, window.data.amplitude[0])
    else:
        image = Image.open(io.BytesIO(path.read_bytes()))
        assert np.allclose(image.info['dpi'], (300,300), atol=.01)


def test_identity_preprocessing_off_main_thread(qtbot, window):
    threads = []
    class Backend:
        def preprocess(self, data, params):
            threads.append(threading.get_ident())
            return data
    window.preprocessor = Backend()
    window.apply_preprocessing.setChecked(True)
    load_mushroom(qtbot, window)
    assert threads and threads[0] != threading.get_ident()
    assert window.data is window.raw_data


def test_volume_mesh_worker_and_invalidation(qtbot, window, tmp_path):
    from part4_ui.lfi_core.processing import prepare_3d
    path = tmp_path/'volume.hdf5'
    z, y, x = np.mgrid[:12,:18,:20]
    values = (z*7+y*2+x).astype(float)
    with h5py.File(path, 'w') as handle:
        for index in range(12):
            handle.create_dataset(f'amplitude/{index}', data=values[index])
            handle.create_dataset(f'phase/{index}', data=-values[index])
    window.open_file(path)
    wait_ready(qtbot, window)
    threads = []
    class Backend:
        def reconstruct(self, data, params):
            threads.append(threading.get_ident())
            return data
    window.reconstructor = Backend()
    panel = window.volume_panel
    panel.reconstruct.setChecked(True)
    qtbot.mouseClick(panel.render_button, QtCore.Qt.MouseButton.LeftButton)
    wait_ready(qtbot, window)
    assert threads[0] != threading.get_ident()
    assert panel.result['mode'] == 'volume'
    assert panel.result['values_sum'] == values.sum()
    assert len(panel.scene_items) == 1
    assert panel.result['rgba'].shape == (20,18,12,4)
    panel.mode.setCurrentIndex(1)
    assert panel.result is None
    panel.percentile.setValue(65)
    qtbot.mouseClick(panel.render_button, QtCore.Qt.MouseButton.LeftButton)
    wait_ready(qtbot, window)
    assert len(panel.scene_items) == 3
    assert panel.result['thresholds'] == tuple(np.percentile(values,[65,95]))
    window.set_language('CN')
    assert panel.render_button.text() == '渲染 / 刷新三维'
    window.brightness.setValue(55)
    wait_ready(qtbot, window)
    assert panel.result is None
    assert not panel.scene_items
    expired_token = window.volume_token
    expired_result = prepare_3d(window.data)
    load_mushroom(qtbot, window)
    window._result(('volume', expired_token, expired_result))
    assert panel.result is None
    assert not panel.render_button.isEnabled()
    print('3D WORKER PASS: volume XYZ buffer, 3 real GLMeshItems, percentile, Identity worker, stale-result invalidation, bilingual UI')


def test_3d_failure_is_readable(qtbot, window, tmp_path):
    path = tmp_path/'constant.hdf5'
    with h5py.File(path,'w') as handle:
        for index in range(3):
            handle.create_dataset(f'amplitude/{index}',data=np.ones((4,5)))
            handle.create_dataset(f'phase/{index}',data=np.zeros((4,5)))
    window.open_file(path)
    wait_ready(qtbot, window)
    window.volume_panel.mode.setCurrentIndex(1)
    window.request_volume()
    qtbot.waitUntil(lambda: window.jobs == 0, timeout=10000)
    assert len(window.errors) == 1
    assert 'Could not render 3D' in window.errors[0]
    assert 'distinct' in window.errors[0]
    assert window.volume_panel.result is None



def test_colorbar_bounds_fit_and_loaded_status(qtbot, window):
    load_mushroom(qtbot, window)
    assert window.colorbar.levels() == tuple(map(float, (window.raw_slice.min(), window.raw_slice.max())))
    assert window.colorbar.axis.labelText == 'Amplitude'
    assert 'mushroom.hdf5' in window.statusBar().currentMessage()
    assert '1300 × 400' in window.statusBar().currentMessage()
    window.channel.setCurrentIndex(1)
    wait_ready(qtbot, window)
    assert window.colorbar.axis.labelText == 'Phase (rad)'
    window.view.setRange(QtCore.QRectF(10,20,100,100), padding=0)
    window.fit_action.trigger()
    x_range, y_range = window.view.viewRange()
    assert x_range[0] >= -26.01 and x_range[1] <= 1326.01
    assert y_range[0] >= -8.01 and y_range[1] <= 408.01
    window.colormap.setCurrentText('Plasma')
    wait_ready(qtbot, window)
    assert window.colorbar.levels() == tuple(map(float, (window.raw_slice.min(), window.raw_slice.max())))


def test_backend_registration_controls_and_worker(qtbot, tmp_path):
    from part4_ui.lfi_core.backend import register_preprocessor, register_reconstructor
    calls = []
    class Backend:
        def preprocess(self, data, params):
            calls.append(('preprocess', threading.get_ident(), params))
            return data
        def reconstruct(self, data, params):
            calls.append(('reconstruct', threading.get_ident(), params))
            return data
    register_preprocessor(Backend(), {'sigma': 1.5})
    register_reconstructor(Backend(), {'iterations': 2})
    try:
        widget = MainWindow(QtCore.QSettings(str(tmp_path/'registry.ini'), QtCore.QSettings.Format.IniFormat))
        qtbot.addWidget(widget)
        widget.errors = []
        widget.show()
        assert widget.pre_group.isVisible()
        assert not widget.volume_panel.reconstruct.isHidden()
        widget.apply_preprocessing.setChecked(True)
        load_mushroom(qtbot, widget)
        assert calls[0][0] == 'preprocess'
        assert calls[0][1] != threading.get_ident()
        assert calls[0][2] == {'sigma': 1.5}
        widget.volume_panel.reconstruct.setChecked(True)
        # Use a small multilayer scan to execute the registered reconstructor.
        path = tmp_path/'registered.hdf5'
        with h5py.File(path,'w') as handle:
            for index in range(3):
                handle.create_dataset(f'amplitude/{index}', data=np.arange(20).reshape(4,5)+index)
                handle.create_dataset(f'phase/{index}', data=np.zeros((4,5)))
        widget.open_file(path)
        wait_ready(qtbot, widget)
        widget.request_volume()
        wait_ready(qtbot, widget)
        reconstruct_call = next(call for call in calls if call[0] == 'reconstruct')
        assert reconstruct_call[1] != threading.get_ident()
        assert reconstruct_call[2] == {'iterations': 2}
        widget.close()
    finally:
        register_preprocessor(None)
        register_reconstructor(None)


def test_default_backend_controls_hidden(window):
    assert window.pre_group.isHidden()
    assert window.volume_panel.reconstruct.isHidden()
    assert 'Identity' not in window.volume_panel.hint.text()



def test_reprocess_preserves_z_pixel_and_view(qtbot, window, tmp_path):
    from dataclasses import replace
    calls = []
    class Backend:
        def preprocess(self, data, params):
            calls.append(threading.get_ident())
            return replace(data, amplitude=data.amplitude+1000)
    window.preprocessor = Backend()
    path = tmp_path/'preserve.hdf5'
    with h5py.File(path,'w') as handle:
        for index in range(5):
            handle.create_dataset(f'amplitude/{index}', data=np.arange(100).reshape(10,10)+index)
            handle.create_dataset(f'phase/{index}', data=np.zeros((10,10)))
    window.open_file(path)
    wait_ready(qtbot, window)
    window.z_spin.setValue(3)
    wait_ready(qtbot, window)
    window.select_pixel(2,4, switch_tab=False)
    window.view.setRange(QtCore.QRectF(1,1,5,5),padding=0)
    view_range = np.array(window.view.viewRange())
    recent = window.settings.value('recent_files',[],type=list)
    for enabled in (True, False):
        window.apply_preprocessing.setChecked(enabled)
        wait_ready(qtbot, window)
        assert window.z_slider.value() == window.z_spin.value() == 3
        assert window.selected_pixel == (2,4)
        current_range = np.array(window.view.viewRange())
        # Aspect-lock may expand one axis slightly while tick labels settle.
        assert np.allclose(current_range.mean(axis=1), view_range.mean(axis=1))
        assert np.isclose(np.ptp(current_range,axis=1).min(), np.ptp(view_range,axis=1).min())
        assert window.raw_slice[4,2] == 45 + (1000 if enabled else 0)
        assert window.settings.value('recent_files',[],type=list) == recent
    assert calls and calls[0] != threading.get_ident()
