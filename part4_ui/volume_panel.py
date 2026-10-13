"""Interactive OpenGL volume/mesh panel, with a safe offscreen test surface."""
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
from .i18n import text


class VolumePanel(QtWidgets.QWidget):
    render_requested = QtCore.Signal()
    invalidated = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.language = 'EN'
        self.scene = None
        self.scene_items = []
        self.scene_annotations = []
        self.result = None
        self.center = QtGui.QVector3D()
        self.distance = 1000
        layout = QtWidgets.QVBoxLayout(self)
        toolbar = QtWidgets.QHBoxLayout()
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(['Volume', 'Isosurface'])
        self.percentile_label = QtWidgets.QLabel()
        self.percentile = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.percentile.setMaximumWidth(140)
        self.percentile_value = QtWidgets.QSpinBox()
        self.percentile_value.setRange(0, 99)
        self.percentile_value.setValue(90)
        self.percentile.valueChanged.connect(self.percentile_value.setValue)
        self.percentile_value.valueChanged.connect(self.percentile.setValue)
        self.percentile.setRange(0, 99)
        self.percentile.setValue(90)
        self.reconstruct = QtWidgets.QCheckBox()
        self.render_button = QtWidgets.QPushButton()
        self.reset_button = QtWidgets.QPushButton()
        toolbar.addWidget(self.mode)
        toolbar.addWidget(self.percentile_label)
        toolbar.addWidget(self.percentile)
        toolbar.addWidget(self.percentile_value)
        toolbar.addWidget(self.reconstruct)
        toolbar.addStretch()
        toolbar.addWidget(self.render_button)
        toolbar.addWidget(self.reset_button)
        layout.addLayout(toolbar)
        self.hint = QtWidgets.QLabel()
        self.hint.setWordWrap(True)
        layout.addWidget(self.hint)
        self.viewport_layout = QtWidgets.QVBoxLayout()
        layout.addLayout(self.viewport_layout, 1)
        self.placeholder = QtWidgets.QLabel(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        self.viewport_layout.addWidget(self.placeholder)
        self.info = QtWidgets.QLabel()
        layout.addWidget(self.info)
        self.render_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.render_button.clicked.connect(self.render_requested)
        self.reset_button.clicked.connect(self.reset_camera)
        self.mode.currentIndexChanged.connect(self.update_mode_controls)
        self.mode.currentIndexChanged.connect(self.invalidate)
        self.percentile.valueChanged.connect(self.invalidate)
        self.reconstruct.toggled.connect(self.invalidate)
        self.set_language('EN')
        self.update_mode_controls()

    def set_language(self, language):
        self.language = language
        for widget, key in [(self.percentile_label, 'iso_percentile'),
                            (self.reconstruct, 'apply_reconstruction'),
                            (self.render_button, 'render_3d'), (self.reset_button, 'reset_camera')]:
            widget.setText(text(key, language))
        self.mode.setItemText(0, text('volume_mode', language))
        self.mode.setItemText(1, text('isosurface_mode', language))
        self.hint.setText(text('3d_hint', language))
        self.placeholder.setText(text('3d_offscreen' if QtWidgets.QApplication.platformName() == 'offscreen' else '3d_start', language))
        if self.result is not None:
            self._update_info()

    def update_mode_controls(self, *args):
        isosurface = self.mode.currentIndex() == 1
        for widget in (self.percentile_label, self.percentile, self.percentile_value):
            widget.setVisible(isosurface)

    def invalidate(self, *args):
        self.invalidated.emit()
        self.clear_result()

    def clear_result(self):
        if self.scene is not None:
            for item in self.scene_items + self.scene_annotations:
                self.scene.removeItem(item)
        self.scene_items.clear()
        self.scene_annotations.clear()
        self.result = None
        self.info.clear()
        self.reset_button.setEnabled(False)

    def set_data_available(self, available):
        self.clear_result()
        self.render_button.setEnabled(available)
        if not available:
            self.info.setText(text('3d_single_layer', self.language))

    def show_result(self, result):
        self.clear_result()
        self.result = result
        # QOpenGLWidget is unsupported by Qt's offscreen platform. Construct real
        # GL items but do not create an OpenGL viewport there; avoid native crashes.
        if self.scene is None and QtWidgets.QApplication.platformName() != 'offscreen':
            self.scene = gl.GLViewWidget()
            self.scene.setBackgroundColor('#151d2c')
            self.viewport_layout.addWidget(self.scene)
            self.placeholder.hide()
        if result['mode'] == 'volume':
            item = gl.GLVolumeItem(result['rgba'], smooth=True, glOptions='translucent')
            item.scale(result['step'], result['step'], result['step'])
            self.scene_items = [item]
        else:
            self.scene_items = [gl.GLMeshItem(vertexes=vertices, faces=faces, color=color,
                                            smooth=False, computeNormals=False,
                                            glOptions='translucent')
                                for vertices, faces, color in result['meshes']]
        # Draw inner shells first so transparent outer shells blend over them.
        if result['mode'] == 'isosurface':
            for index, item in enumerate(self.scene_items):
                item.setDepthValue(-index)
        for item in self.scene_items:
            if self.scene is not None:
                self.scene.addItem(item)
        z, y, x = result['original_shape']
        if self.scene is not None:
            axes = gl.GLAxisItem()
            axes.setSize(x=x-1, y=y-1, z=z-1)
            labels = [gl.GLTextItem(pos=position, text=axis, color='white')
                      for position, axis in [((x-1,0,0), 'X'), ((0,y-1,0), 'Y'), ((0,0,z-1), 'Z')]]
            self.scene_annotations = [axes] + labels
            for annotation in self.scene_annotations:
                self.scene.addItem(annotation)
        self.center = QtGui.QVector3D((x-1)/2, (y-1)/2, (z-1)/2)
        self.distance = max(x, y, z) * 1.5
        self.reset_camera()
        self.reset_button.setEnabled(True)
        self._update_info()

    def reset_camera(self):
        if self.scene is not None:
            self.scene.setCameraPosition(pos=self.center, distance=self.distance, elevation=25, azimuth=-45)

    def _update_info(self):
        result = self.result
        self.info.setText(f"{text('volume_dimensions', self.language)}: {result['original_shape']}   "
                          f"{text('sampled_points', self.language)}: {result['sampled_points']:,}   "
                          f"{text('downsample', self.language)}: 1/{result['step']}")
        if result['mode'] == 'isosurface':
            self.info.setText(self.info.text() + f"   {text('surface_count', self.language)}: {len(result['meshes'])}")
