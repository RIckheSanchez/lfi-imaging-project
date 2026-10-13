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
