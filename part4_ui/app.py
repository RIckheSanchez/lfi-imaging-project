"""Launch with python -m part4_ui.app from the repository root."""
import json
import os
from copy import deepcopy
import sys
import traceback
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from .i18n import text
from .lfi_core.phase_semantics import phase_label_key, phase_encoding_status
from .lfi_core.loaders import load_path, loader_for, get_loaders
from .lfi_core.processing import normalized_slice, adjust_slice, prepare_3d, finite_values, finite_range
from .lfi_core.export import scale_bar, export_csv, pixel_image, publication_png
from .lfi_core.backend import (IdentityPreprocessor, IdentityReconstructor,
                               get_preprocessor_registration, get_reconstructor_registration, register_preprocessor)
from .volume_panel import VolumePanel

COLORMAPS = ['Viridis', 'Plasma', 'Inferno', 'Magma', 'Turbo', 'Hot', 'Cool']


class ImagePlotHost(QtWidgets.QWidget):
    """Size the plot to the sensor aspect ratio instead of stretching its bounds."""
    def __init__(self, plot):
        super().__init__()
        self.plot = plot
        plot.setParent(self)
        self.shape = (400, 400)
        self.setMinimumSize(240, 240)

    def set_data_shape(self, width, height):
        self.shape = (width, height)
        self.resize_plot()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_plot()

    def resize_plot(self):
        width, height = self.shape
        available_width, available_height = self.width(), self.height()
        scale = max(.01, min((available_width-160)/width, (available_height-65)/height))
        plot_width = min(available_width, int(width*scale+160))
        plot_height = min(available_height, int(height*scale+65))
        self.plot.setGeometry((available_width-plot_width)//2, (available_height-plot_height)//2,
                              plot_width, plot_height)


class WorkerSignals(QtCore.QObject):
    result = QtCore.Signal(object)
    error = QtCore.Signal(object)
    finished = QtCore.Signal()


class Worker(QtCore.QRunnable):
    """Run non-Qt computation on the pool, delivering results to GUI slots."""
    def __init__(self, kind, token, operation):
        super().__init__()
        self.kind, self.token, self.operation = kind, token, operation
        self.signals = WorkerSignals()

    def run(self):
        try:
            self.signals.result.emit((self.kind, self.token, self.operation()))
        except Exception as error:
            self.signals.error.emit((self.kind, self.token, str(error)))
        finally:
            self.signals.finished.emit()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings=None, preprocessor=None, reconstructor=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings('LFI', 'Viewer')
        preprocessing = get_preprocessor_registration()
        reconstruction = get_reconstructor_registration()
        self.preprocessor = preprocessor or (preprocessing.implementation if preprocessing else IdentityPreprocessor())
        self.reconstructor = reconstructor or (reconstruction.implementation if reconstruction else IdentityReconstructor())
        self.dev_params = os.environ.get('LFI_DEV_PARAMS') == '1'
        self.preprocess_params = deepcopy(preprocessing.params) if preprocessing and self.dev_params else {}
        self.reconstruction_params = deepcopy(reconstruction.params) if reconstruction else {}
        self.language = self.settings.value('language', 'EN')
        self.data = self.raw_data = None
        self.data_key = None
        self.selected_pixel = (0, 0)
        self.raw_slice = self.adjusted = None
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.jobs = 0
        self.load_token = self.slice_token = self.volume_token = 0
        self.workers = set()
        self.scale_item = None
        self.fit_next_slice = True
        self.resize(1320, 860)
        self.setAcceptDrops(True)
        self._build_ui()
        self.set_language(self.language)
        self.statusBar().showMessage(self.tr_text('ready'))

    def tr_text(self, key):
        return text(key, self.language)

    def value_label(self):
        return self.tr_text('processed_value' if self.data is not None and self.data.metadata.get('ui_preprocessing') else 'raw')

    def pixel_value_text(self, value):
        return f'{value:.6g}' if np.isfinite(value) else self.tr_text('missing_value')

    def amplitude_label(self):
        state = self.data.metadata.get('ui_preprocessing') if self.data is not None else None
        if state:
            return self.tr_text('amplitude_preprocessed_dimensionless' if state['amplitude_units'] == 'dimensionless' else 'amplitude_preprocessed')
        return self.tr_text('amplitude')

    def phase_label(self):
        metadata = self.data.metadata if self.data is not None else {}
        return self.tr_text(phase_label_key(metadata))

    def _build_ui(self):
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(150)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.busy_label = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.busy_label)
        self.dock = QtWidgets.QDockWidget(self)
        self.dock.setObjectName('display_controls')
        dock_widget = QtWidgets.QWidget()
        dock_layout = QtWidgets.QVBoxLayout(dock_widget)
        self.form = QtWidgets.QFormLayout()
        self.labels = {}
        self.channel = QtWidgets.QComboBox()
        self.channel.addItems(['Amplitude', 'Phase'])
        self.brightness = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.brightness.setRange(0, 100)
        self.brightness.setValue(60)
        self.contrast = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.contrast.setRange(0, 100)
        self.contrast.setValue(75)
        self.colormap = QtWidgets.QComboBox()
        self.colormap.addItems(COLORMAPS)
        self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.z_spin = QtWidgets.QSpinBox()
        self.z_slider.setRange(0, 0)
        self.z_spin.setRange(0, 0)
        z_widget = QtWidgets.QWidget()
        z_layout = QtWidgets.QVBoxLayout(z_widget)
        z_layout.setContentsMargins(0, 0, 0, 0)
        z_layout.addWidget(self.z_slider)
        z_layout.addWidget(self.z_spin)
        self.z_label = QtWidgets.QLabel()
        z_layout.addWidget(self.z_label)
        self.pixel_size = QtWidgets.QDoubleSpinBox()
        self.pixel_size.setDecimals(6)
        self.pixel_size.setRange(0, 1e9)
        for key, widget in [('channel', self.channel), ('brightness', self.brightness),
                            ('contrast', self.contrast), ('colormap', self.colormap),
                            ('z', z_widget), ('pixel_size', self.pixel_size)]:
            label = QtWidgets.QLabel()
            self.labels[key] = label
            self.form.addRow(label, widget)
        dock_layout.addLayout(self.form)
        self.scale_hint = QtWidgets.QLabel()
        self.scale_hint.setWordWrap(True)
        dock_layout.addWidget(self.scale_hint)
        self.pre_group = QtWidgets.QGroupBox()
        pre_layout = QtWidgets.QVBoxLayout(self.pre_group)
        self.apply_preprocessing = QtWidgets.QCheckBox()
        self.parameter = QtWidgets.QLineEdit()
        self.parameter.setText(json.dumps(self.preprocess_params))
        self.parameter.editingFinished.connect(self.update_preprocess_params)
        self.parameter_label = QtWidgets.QLabel()
        pre_layout.addWidget(self.apply_preprocessing)
        pre_layout.addWidget(self.parameter_label)
        pre_layout.addWidget(self.parameter)
        self.parameter_label.setVisible(self.dev_params)
        self.parameter.setVisible(self.dev_params)
        self.pre_group.setVisible(not isinstance(self.preprocessor, IdentityPreprocessor))
        dock_layout.addWidget(self.pre_group)
        dock_layout.addStretch()
        self.dock.setWidget(dock_widget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dock)
        self.tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(self.tabs)
        self.slice_tab = QtWidgets.QWidget()
        slice_layout = QtWidgets.QHBoxLayout(self.slice_tab)
        self.image_plot = pg.PlotWidget(background='#151d2c')
        self.image_plot.setMinimumSize(160, 160)
        self.view = self.image_plot.getViewBox()
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.image = pg.ImageItem(axisOrder='row-major')
        self.image_plot.addItem(self.image)
        self.cross_x = pg.InfiniteLine(angle=90, pen=pg.mkPen('#ffcf57', width=1))
        self.cross_y = pg.InfiniteLine(angle=0, pen=pg.mkPen('#ffcf57', width=1))
        self.image_plot.addItem(self.cross_x)
        self.image_plot.addItem(self.cross_y)
        self.cross_x.hide()
        self.cross_y.hide()
        self.image_plot.scene().sigMouseMoved.connect(self.hover_pixel)
        self.image_plot.scene().sigMouseClicked.connect(self.click_pixel)
        self.colorbar = pg.ColorBarItem(values=(0,1), interactive=False, colorMapMenu=False)
        self.colorbar.axis.setWidth(85)
        self.image_plot.getPlotItem().layout.addItem(self.colorbar, 2, 5)
        self.plot_host = ImagePlotHost(self.image_plot)
        slice_layout.addWidget(self.plot_host, 3)
        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_widget)
        self.scan_group = QtWidgets.QGroupBox()
        self.scan_label = QtWidgets.QLabel()
        self.scan_label.setWordWrap(True)
        self.scan_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        scan_layout = QtWidgets.QVBoxLayout(self.scan_group)
        scan_layout.addWidget(self.scan_label)
        self.pixel_group = QtWidgets.QGroupBox()
        self.pixel_label = QtWidgets.QLabel()
        pixel_layout = QtWidgets.QVBoxLayout(self.pixel_group)
        pixel_layout.addWidget(self.pixel_label)
        self.histogram = pg.PlotWidget()
        self.histogram.setMinimumHeight(200)
        self.histogram_curve = self.histogram.plot(pen=pg.mkPen('#5bb8eb', width=2), fillLevel=0, brush=(91,184,235,60))
        info_layout.addWidget(self.scan_group)
        info_layout.addWidget(self.pixel_group)
        info_layout.addWidget(self.histogram, 1)
        info_widget.setMinimumWidth(390)
        info_widget.setMaximumWidth(480)
        slice_layout.addWidget(info_widget, 1)
        self.volume_panel = VolumePanel()
        self.volume_panel.set_reconstruction_available(False)
        self.volume_panel.render_requested.connect(self.request_volume)
        self.volume_panel.invalidated.connect(self.invalidate_volume)
        self.brightness.valueChanged.connect(self.invalidate_volume)
        self.contrast.valueChanged.connect(self.invalidate_volume)
        self.colormap.currentIndexChanged.connect(self.invalidate_volume)
        self.waveform_tab = QtWidgets.QWidget()
        wave_layout = QtWidgets.QVBoxLayout(self.waveform_tab)
        self.amplitude_plot = pg.PlotWidget()
        self.phase_plot = pg.PlotWidget()
        wave_layout.addWidget(self.amplitude_plot)
        wave_layout.addWidget(self.phase_plot)
        self.tabs.addTab(self.slice_tab, '')
        self.tabs.addTab(self.volume_panel, '')
        self.tabs.addTab(self.waveform_tab, '')
        self.channel.currentIndexChanged.connect(self.request_slice)
        self.brightness.valueChanged.connect(self.request_slice)
        self.contrast.valueChanged.connect(self.request_slice)
        self.colormap.currentIndexChanged.connect(self.request_slice)
        self.z_slider.valueChanged.connect(self._z_changed)
        self.z_spin.valueChanged.connect(self.z_slider.setValue)
        self.pixel_size.valueChanged.connect(self.update_scale)
        self.apply_preprocessing.toggled.connect(self.reprocess)
        self._build_menus()

    def _build_menus(self):
        self.menuBar().clear()
        self.file_menu = self.menuBar().addMenu(self.tr_text('file'))
        self.open_action = self.file_menu.addAction(self.tr_text('open'), self.open_dialog)
        self.open_action.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        self.folder_action = self.file_menu.addAction(self.tr_text('open_folder'), self.open_folder_dialog)
        self.recent_menu = self.file_menu.addMenu(self.tr_text('recent'))
        self.export_action = self.file_menu.addAction(self.tr_text('export'), self.export_dialog)
        self.export_action.setEnabled(self.data is not None)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.tr_text('quit'), self.close).setShortcut(QtGui.QKeySequence.StandardKey.Quit)
        view_menu = self.menuBar().addMenu(self.tr_text('view'))
        view_menu.addAction(self.dock.toggleViewAction())
        self.fit_action = view_menu.addAction(self.tr_text('fit'), self.fit_view)
        self.fit_action.setShortcut(QtGui.QKeySequence('Meta+0'))
        language_menu = self.menuBar().addMenu(self.tr_text('language'))
        language_menu.addAction('English', lambda: self.set_language('EN'))
        language_menu.addAction('中文', lambda: self.set_language('CN'))
        help_menu = self.menuBar().addMenu(self.tr_text('help'))
        help_menu.addAction(self.tr_text('about'), lambda: QtWidgets.QMessageBox.information(
            self, self.tr_text('about'), self.tr_text('about_text')))
        self.update_recent()

    def set_language(self, language):
        self.language = language
        self.settings.setValue('language', language)
        self.setWindowTitle(self.tr_text('title'))
        self.dock.setWindowTitle(self.tr_text('controls'))
        for key, label in self.labels.items():
            label.setText(self.tr_text(key))
        self.channel.setItemText(0, self.tr_text('amplitude'))
        self.channel.setItemText(1, self.tr_text('phase'))
        self.scale_hint.setText(self.tr_text('scale_hint'))
        self.pre_group.setTitle(self.tr_text('preprocessing'))
        self.apply_preprocessing.setText(self.tr_text('apply'))
        self.parameter_label.setText(self.tr_text('parameters'))
        self.scan_group.setTitle(self.tr_text('scan_info'))
        self.pixel_group.setTitle(self.tr_text('pixel_info'))
        for index, key in enumerate(('slice', 'volume', 'waveform')):
            self.tabs.setTabText(index, self.tr_text(key))
        self.volume_panel.set_language(language)
        self.image_plot.setLabel('bottom', self.tr_text('x'))
        self.image_plot.setLabel('left', self.tr_text('y'))
        self.histogram.setTitle(self.tr_text('histogram'))
        self.update_histogram_label()
        self.histogram.setLabel('left', self.tr_text('count'))
        self.amplitude_plot.setLabel('bottom', self.tr_text('z_index'))
        self.amplitude_plot.setLabel('left', self.amplitude_label())
        self.phase_plot.setLabel('bottom', self.tr_text('z_index'))
        self.phase_plot.setLabel('left', self.phase_label())
        self._build_menus()
        self.update_info()
        self.update_histogram_theme()
        self.update_colorbar()
        if self.jobs:
            self.busy_label.setText(self.tr_text('busy'))

    def update_histogram_label(self):
        self.histogram.setLabel('bottom', self.phase_label() if self.channel.currentIndex() else self.amplitude_label(), **{'font-size': '9pt'})
        # AxisItem updates text without repositioning when its geometry is fixed.
        # Re-run the public resize handler to center a changed, longer label.
        self.histogram.getAxis('bottom').resizeEvent()

    def update_histogram_theme(self):
        palette = self.palette()
        background = palette.color(QtGui.QPalette.ColorRole.Window)
        foreground = palette.color(QtGui.QPalette.ColorRole.WindowText)
        accent = palette.color(QtGui.QPalette.ColorRole.Highlight)
        self.histogram.setBackground(background)
        for name in ('left', 'bottom'):
            axis = self.histogram.getAxis(name)
            axis.setPen(pg.mkPen(foreground))
            axis.setTextPen(pg.mkPen(foreground))
        self.histogram.setTitle(self.tr_text('histogram'), color=foreground.name())
        self.histogram_curve.setPen(pg.mkPen(accent, width=2))
        fill = QtGui.QColor(accent)
        fill.setAlpha(55)
        self.histogram_curve.setBrush(pg.mkBrush(fill))

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.Type.PaletteChange, QtCore.QEvent.Type.ApplicationPaletteChange,
                            QtCore.QEvent.Type.StyleChange) and hasattr(self, 'histogram'):
            self.update_histogram_theme()

    def update_recent(self):
        self.recent_menu.clear()
        for path in self.settings.value('recent_files', [], type=list)[:5]:
            self.recent_menu.addAction(path, lambda checked=False, filename=path: self.open_file(filename))

    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, self.tr_text('open'), '', self.tr_text('filter'))
        if path:
            self.open_file(path)

    def open_folder_dialog(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr_text('open_folder'))
        if path:
            self.open_file(path)

    def accepts_path(self, path):
        try:
            loader_for(path)
            return True
        except (OSError, ValueError):
            return False

    def open_file(self, path):
        self.load_token += 1
        self.slice_token += 1
        self.invalidate_volume()
        apply = self.apply_preprocessing.isChecked()
        backend = self.preprocessor
        params = deepcopy(self.preprocess_params) if self.dev_params else {}
        def load():
            raw, key = load_path(path)
            processed = backend.preprocess(raw, params) if apply else raw
            return raw, processed, key
        self._submit('load', self.load_token, load)

    def update_preprocess_params(self):
        if not self.dev_params:
            self.preprocess_params = {}
            return
        try:
            params = json.loads(self.parameter.text())
            if not isinstance(params, dict):
                raise ValueError(self.tr_text('parameter_error'))
            self.preprocess_params = params
            if self.apply_preprocessing.isChecked():
                self.reprocess()
        except (ValueError, TypeError) as error:
            self.show_error(f"{self.tr_text('parameter_error')}: {error}")

    def reprocess(self):
        if self.raw_data is None:
            return
        self.load_token += 1
        self.slice_token += 1
        self.invalidate_volume()
        raw, key = self.raw_data, self.data_key
        apply, backend = self.apply_preprocessing.isChecked(), self.preprocessor
        params = deepcopy(self.preprocess_params) if self.dev_params else {}
        self._submit('reprocess', self.load_token,
                     lambda: backend.preprocess(raw, params) if apply else raw)

    def _submit(self, kind, token, operation):
        worker = Worker(kind, token, operation)
        self.workers.add(worker)
        self.jobs += 1
        self.progress.show()
        self.busy_label.setText(self.tr_text('busy'))
        worker.signals.result.connect(self._result)
        worker.signals.error.connect(self._error)
        worker.signals.finished.connect(self._finished)
        self.pool.start(worker)

    @QtCore.Slot(object)
    def _result(self, message):
        kind, token, value = message
        try:
            if kind == 'load' and token == self.load_token:
                self.invalidate_volume()
                self.raw_data, self.data, self.data_key = value
                self.selected_pixel = (self.data.width // 2, self.data.height // 2)
                self.raw_slice = self.adjusted = None
                self.fit_next_slice = True
                self.plot_host.set_data_shape(self.data.width, self.data.height)
                self.view.setLimits(xMin=-self.data.width*.02, xMax=self.data.width*1.02,
                                    yMin=-self.data.height*.02, yMax=self.data.height*1.02,
                                    minXRange=min(2,self.data.width), minYRange=min(2,self.data.height),
                                    maxXRange=self.data.width*1.04, maxYRange=self.data.height*1.04)
                self.slice_token += 1
                with QtCore.QSignalBlocker(self.z_slider), QtCore.QSignalBlocker(self.z_spin):
                    self.z_slider.setRange(0, self.data.n_slices - 1)
                    self.z_spin.setRange(0, self.data.n_slices - 1)
                    self.z_slider.setValue(0)
                    self.z_spin.setValue(0)
                recent = self.settings.value('recent_files', [], type=list)
                self.settings.setValue('recent_files', [self.data_key[0]] + [p for p in recent if p != self.data_key[0]][:4])
                self.update_recent()
                self.export_action.setEnabled(True)
                self.volume_panel.set_data_available(self.data.n_slices > 1)
                self.update_info()
                self.statusBar().showMessage(self.loaded_status())
                self.request_slice()
            elif kind == 'reprocess' and token == self.load_token:
                old_shape = self.data.amplitude.shape
                index, pixel = self.z_slider.value(), self.selected_pixel
                self.data = value
                self.raw_slice = self.adjusted = None
                with QtCore.QSignalBlocker(self.z_slider), QtCore.QSignalBlocker(self.z_spin):
                    self.z_slider.setRange(0, self.data.n_slices - 1)
                    self.z_spin.setRange(0, self.data.n_slices - 1)
                    self.z_slider.setValue(min(index, self.data.n_slices - 1))
                    self.z_spin.setValue(self.z_slider.value())
                if old_shape[1:] == self.data.amplitude.shape[1:]:
                    self.selected_pixel = pixel
                else:
                    self.selected_pixel = (self.data.width // 2, self.data.height // 2)
                    self.plot_host.set_data_shape(self.data.width, self.data.height)
                    self.view.setLimits(xMin=-self.data.width*.02, xMax=self.data.width*1.02,
                                        yMin=-self.data.height*.02, yMax=self.data.height*1.02,
                                        minXRange=min(2,self.data.width), minYRange=min(2,self.data.height),
                                        maxXRange=self.data.width*1.04, maxYRange=self.data.height*1.04)
                    self.fit_next_slice = True
                self.volume_panel.set_data_available(self.data.n_slices > 1)
                self.update_info()
                self.request_slice()
            elif kind == 'slice' and token == self.slice_token:
                self.colorbar.show()
                self.raw_slice, self.adjusted, counts, edges = value
                cmap = pg.colormap.get(self.colormap.currentText().lower(), source='matplotlib')
                self.image.setLookupTable(cmap.getLookupTable(0., 1., 256))
                self.image.setImage(self.adjusted, levels=(0, 1), autoLevels=False)
                if self.fit_next_slice:
                    self.fit_view()
                    self.fit_next_slice = False
                self.update_histogram_label()
                self.histogram_curve.setData((edges[:-1] + edges[1:]) / 2, counts)
                self.update_colorbar()
                self.update_scale()
                self.select_pixel(*self.selected_pixel, switch_tab=False)
                self.update_info()
            elif kind == 'volume' and token == self.volume_token:
                self.volume_panel.show_result(value)
            elif kind == 'export':
                self.statusBar().showMessage(f"{self.tr_text('export_done')}: {value}", 10000)
        except Exception as error:
            self.show_error(str(error))

    @QtCore.Slot(object)
    def _error(self, message):
        kind, token, error = message
        if kind in ('load', 'reprocess') and token != self.load_token:
            return
        if kind == 'slice' and token != self.slice_token:
            return
        if kind == 'volume' and token != self.volume_token:
            return
        if kind == 'slice':
            self.raw_slice = self.adjusted = None
            self.image.clear()
            self.histogram_curve.clear()
            self.colorbar.hide()
            self.pixel_label.setText(self.tr_text('no_finite_values'))
        key = {'reprocess': 'render_error', 'load': 'load_error', 'slice': 'render_error', 'export': 'export_error', 'volume': '3d_error'}[kind]
        self.show_error(f'{self.tr_text(key)}:\n{error}')

    def show_error(self, message):
        QtWidgets.QMessageBox.warning(self, self.tr_text('error'), message)

    @QtCore.Slot()
    def _finished(self):
        self.jobs -= 1
        if not self.jobs:
            self.progress.hide()
            self.busy_label.clear()
            self.workers.clear()
            if self.data is not None:
                self.statusBar().showMessage(self.loaded_status())

    def _z_changed(self, index):
        with QtCore.QSignalBlocker(self.z_spin):
            self.z_spin.setValue(index)
        self.request_slice()

    def request_slice(self, *args):
        if self.data is None:
            return
        self.slice_token += 1
        data, z = self.data, self.data.z_indices[self.z_slider.value()]
        channel = 'phase' if self.channel.currentIndex() else 'amplitude'
        brightness, contrast = self.brightness.value(), self.contrast.value()
        def process():
            normalized, _, _ = normalized_slice(data, z, channel)
            raw = data.get_slice(z)[channel]
            counts, edges = np.histogram(finite_values(raw), bins=50)
            return raw, adjust_slice(normalized, brightness, contrast), counts, edges
        self._submit('slice', self.slice_token, process)

    def invalidate_volume(self, *args):
        self.volume_token += 1
        self.volume_panel.clear_result()

    def request_volume(self):
        if self.data is None or self.data.n_slices < 2:
            return
        self.volume_token += 1
        token = self.volume_token
        data, backend = self.data, self.reconstructor
        apply = self.volume_panel.reconstruct.isChecked() and self.reconstruction_available()
        params = deepcopy(self.reconstruction_params)
        mode = 'isosurface' if self.volume_panel.mode.currentIndex() else 'volume'
        percentile = self.volume_panel.percentile.value()
        colormap = self.colormap.currentText()
        brightness, contrast = self.brightness.value(), self.contrast.value()
        def compute():
            reconstructed = backend.reconstruct(data, params) if apply else data
            return prepare_3d(reconstructed, mode, percentile, colormap, brightness, contrast)
        self._submit('volume', token, compute)

    def fit_view(self):
        if self.data is not None:
            self.view.setRange(QtCore.QRectF(0, 0, self.data.width, self.data.height), padding=.01)

    def update_colorbar(self):
        if self.raw_slice is None:
            return
        cmap = pg.colormap.get(self.colormap.currentText().lower(), source='matplotlib')
        positions = np.linspace(0, 1, 256)
        colors = cmap.map(adjust_slice(positions, self.brightness.value(), self.contrast.value()), mode='byte')
        self.colorbar.setColorMap(pg.ColorMap(positions, colors))
        minimum, maximum = finite_range(self.raw_slice)
        self.colorbar.setLevels((minimum, maximum if maximum > minimum else minimum+1))
        self.colorbar.axis.setLabel(self.phase_label() if self.channel.currentIndex() else self.amplitude_label())

    def loaded_status(self):
        data = self.data
        return f"{Path(data.file_path).name} · {data.width} × {data.height} · {data.n_slices} {self.tr_text('slices')}"

    def reconstruction_available(self):
        if self.data is None or isinstance(self.reconstructor, IdentityReconstructor):
            return False
        depth = self.data.metadata.get('depth_mm')
        try:
            return depth is not None and len(depth) == self.data.n_slices
        except TypeError:
            return False

    def update_info(self):
        self.volume_panel.set_reconstruction_available(self.reconstruction_available())
        if self.data is None:
            return
        data = self.data
        rows = [('target', data.target_type), ('size', f'{data.width} × {data.height}'),
                ('slices', data.n_slices), ('z_range', f'{data.z_indices[0]} → {data.z_indices[-1]}'),
                ('filename', Path(data.file_path).name),
                ('phase_status', phase_encoding_status(data.metadata))]
        self.scan_label.setText('\n'.join(f'{self.tr_text(key)}: {value}' for key, value in rows))
        self.phase_plot.setLabel('left', self.phase_label())
        self.amplitude_plot.setLabel('left', self.amplitude_label())
        self.channel.setItemText(0, self.tr_text('amplitude'))
        self.channel.setToolTip(self.amplitude_label())
        self.z_label.setText(f"{self.tr_text('z')}: {data.z_indices[self.z_slider.value()]}")
        if self.raw_slice is not None:
            x, y = self.selected_pixel
            label = self.phase_label() if self.channel.currentIndex() else self.amplitude_label()
            self.pixel_label.setText(f"X: {x}   Y: {y}\n{label}\n{self.value_label()}: {self.pixel_value_text(self.raw_slice[y, x])}")

    def _pixel_at(self, position):
        if self.raw_slice is None or not self.image.sceneBoundingRect().contains(position):
            return None
        point = self.image.mapFromScene(position)
        x, y = int(np.floor(point.x())), int(np.floor(point.y()))
        if 0 <= x < self.raw_slice.shape[1] and 0 <= y < self.raw_slice.shape[0]:
            return x, y
        return None

    def hover_pixel(self, position):
        pixel = self._pixel_at(position)
        if pixel is not None:
            x, y = pixel
            self.statusBar().showMessage(f"x={x}  y={y}  {self.value_label()}={self.pixel_value_text(self.raw_slice[y, x])}")

    def click_pixel(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pixel = self._pixel_at(event.scenePos())
        if pixel is not None:
            event.accept()
            self.select_pixel(*pixel)

    def select_pixel(self, x, y, switch_tab=True):
        if self.raw_slice is None:
            return
        x = max(0, min(int(x), self.data.width - 1))
        y = max(0, min(int(y), self.data.height - 1))
        self.selected_pixel = (x, y)
        self.cross_x.setPos(x + .5)
        self.cross_y.setPos(y + .5)
        self.cross_x.show()
        self.cross_y.show()
        self.update_info()
        self.amplitude_plot.clear()
        self.phase_plot.clear()
        self.amplitude_plot.plot(self.data.z_indices, self.data.amplitude[:, y, x], pen='#55b3e5', symbol='o')
        self.phase_plot.plot(self.data.z_indices, self.data.phase[:, y, x], pen='#c590ed', symbol='o')
        if switch_tab:
            self.tabs.setCurrentIndex(2)

    def update_scale(self, *args):
        if self.scale_item is not None:
            # pyqtgraph ScaleBar reanchors even when detached unless offset is None.
            self.scale_item.offset = None
            self.view.removeItem(self.scale_item)
            self.scale_item.deleteLater()
            self.scale_item = None
        self.scale_hint.setVisible(self.pixel_size.value() == 0)
        if self.data is None or self.pixel_size.value() == 0:
            return
        length, label = scale_bar(self.data.width, self.pixel_size.value())
        self.scale_item = pg.ScaleBar(length, suffix='', offset=(-20, -35), brush='white', pen='white')
        self.scale_item.setParentItem(self.view)
        self.scale_item.text.setText(label)

    def dragEnterEvent(self, event):
        if any(url.isLocalFile() and self.accepts_path(url.toLocalFile())
               for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile() and self.accepts_path(url.toLocalFile()):
                self.open_file(url.toLocalFile())
                event.acceptProposedAction()
                break

    def export_dialog(self):
        if self.data is None:
            return
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr_text('export'))
        layout = QtWidgets.QFormLayout(dialog)
        formats = QtWidgets.QComboBox()
        formats.addItems(['CSV', 'PNG', 'JPEG', 'TIFF', self.tr_text('publication')])
        layout.addRow(self.tr_text('format'), formats)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Save).setText(self.tr_text('save'))
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Cancel).setText(self.tr_text('cancel'))
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        image_format = ['CSV', 'PNG', 'JPEG', 'TIFF', 'publication'][formats.currentIndex()]
        channel = 'phase' if self.channel.currentIndex() else 'amplitude'
        z = self.data.z_indices[self.z_slider.value()]
        extension = 'png' if image_format == 'publication' else image_format.lower()
        filename = f'lfi_{channel}_z{z}.{extension}'
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, self.tr_text('export'), filename, f'{image_format} (*.{extension})')
        if path:
            self.export_to(path, image_format)

    def export_to(self, path, image_format):
        data = self.data
        z = data.z_indices[self.z_slider.value()]
        channel = 'phase' if self.channel.currentIndex() else 'amplitude'
        brightness, contrast = self.brightness.value(), self.contrast.value()
        colormap, pixel_size = self.colormap.currentText(), self.pixel_size.value()
        color_label = self.phase_label() if channel == 'phase' else self.amplitude_label()
        x_label, y_label = self.tr_text('x'), self.tr_text('y')
        def save():
            if image_format == 'CSV':
                Path(path).write_text(export_csv(data.get_slice(z)[channel]))
            else:
                norm, _, _ = normalized_slice(data, z, channel)
                adjusted = adjust_slice(norm, brightness, contrast)
                content = publication_png(adjusted, colormap, pixel_size, color_label, x_label, y_label) if image_format == 'publication' else pixel_image(adjusted, colormap, image_format)
                Path(path).write_bytes(content)
            return str(path)
        self._submit('export', 0, save)

    def closeEvent(self, event):
        if self.jobs:
            QtWidgets.QMessageBox.information(self, self.tr_text('busy'), self.tr_text('wait_close'))
            event.ignore()
        else:
            event.accept()


def main():
    from .lfi_core.adapters.part2 import Part2Preprocessor
    register_preprocessor(Part2Preprocessor())
    pg.setConfigOptions(imageAxisOrder='row-major', antialias=True)
    application = QtWidgets.QApplication(sys.argv)
    application.setApplicationName('LFI Viewer')
    window = MainWindow()
    def handle_exception(error_type, value, trace):
        traceback.print_exception(error_type, value, trace)
        window.show_error(str(value))
    sys.excepthook = handle_exception
    window.show()
    return application.exec()


if __name__ == '__main__':
    sys.exit(main())
