"""Camera input regressions; hidden GL widgets need no offscreen GL context."""
import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from part4_ui.volume_panel import VolumePanel, ZoomableGLViewWidget
from part4_ui.lfi_core.data_types import LFIRawData
from part4_ui.lfi_core.processing import prepare_3d


@pytest.fixture
def view(qtbot):
    widget = ZoomableGLViewWidget()
    qtbot.addWidget(widget)
    widget.set_volume_size((70, 1400, 400))
    widget.setCameraPosition(distance=2100)
    return widget


def pinch(view, value, kind=QtCore.Qt.NativeGestureType.ZoomNativeGesture):
    point = QtCore.QPointF(10, 10)
    event = QtGui.QNativeGestureEvent(kind, QtGui.QPointingDevice.primaryPointingDevice(),
                                     2, point, point, point, value, QtCore.QPointF())
    QtWidgets.QApplication.sendEvent(view, event)
    return event


def wheel(view, angle, pixels=(0, 0), ctrl=False):
    event = QtGui.QWheelEvent(
        QtCore.QPointF(10, 10), QtCore.QPointF(10, 10),
        QtCore.QPoint(*pixels), QtCore.QPoint(*angle),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.KeyboardModifier.ControlModifier if ctrl else QtCore.Qt.KeyboardModifier.NoModifier,
        QtCore.Qt.ScrollPhase.ScrollUpdate, False)
    QtWidgets.QApplication.sendEvent(view, event)
    return event


def test_native_pinch_zoom_and_volume_relative_limits(view):
    assert pinch(view, .2).isAccepted()
    assert view.opts['distance'] == pytest.approx(1680)
    pinch(view, -.25)
    assert view.opts['distance'] == pytest.approx(2100)
    pinch(view, 2)
    assert view.opts['distance'] == pytest.approx(1400 * .05)
    pinch(view, -1e6)
    assert view.opts['distance'] == pytest.approx(1400 * 50)
    # Other native gestures keep the base widget behavior.
    before = view.opts['distance']
    pinch(view, .2, QtCore.Qt.NativeGestureType.RotateNativeGesture)
    assert view.opts['distance'] == before


@pytest.mark.parametrize('angle,pixels,amount,base', [
    ((0, 120), (0, 0), 120, .999),
    ((5, -120), (0, 0), -120, .999),
    ((120, -5), (0, 0), 120, .999),
    ((0, -120), (1, 30), 30, .99),
    ((120, 0), (-30, 1), -30, .99),
])
def test_wheel_uses_dominant_axis_and_prefers_trackpad_pixels(view, angle, pixels, amount, base):
    event = wheel(view, angle, pixels)
    assert event.isAccepted()
    assert view.opts['distance'] == pytest.approx(2100 * base**amount)
    assert (view.opts['distance'] < 2100) == (amount > 0)


@pytest.mark.parametrize('pixels', [(0, 0), (0, 30)])
def test_ctrl_wheel_changes_fov_not_distance(view, pixels):
    before = view.opts['fov']
    wheel(view, (0, 120), pixels, ctrl=True)
    assert view.opts['distance'] == 2100
    assert view.opts['fov'] < before


def test_wheel_distance_limits_follow_new_volume_size(view):
    view.set_volume_size((3, 4, 5))
    view.setCameraPosition(distance=5 * 50)
    wheel(view, (0, -120))
    assert view.opts['distance'] == 5 * 50
    view.setCameraPosition(distance=5 * .05)
    wheel(view, (0, 120))
    assert view.opts['distance'] == 5 * .05


def test_zoom_buttons_shortcuts_translation_and_reset(qtbot, view):
    panel = VolumePanel()
    qtbot.addWidget(panel)
    panel.scene = view  # Remains hidden: offscreen Qt has no OpenGL viewport.
    assert not panel.zoom_in_button.isEnabled()
    assert not panel.zoom_out_button.isEnabled()
    values = np.arange(60, dtype=float).reshape(3, 4, 5)
    data = LFIRawData(values, values.copy(), [0, 1, 2], 'test', 'test')
    panel.show_result(prepare_3d(data))
    assert view.volume_extent == 5
    assert panel.zoom_in_button.isEnabled() and panel.zoom_out_button.isEnabled()
    panel.show()
    panel.activateWindow()
    qtbot.waitUntil(panel.isActiveWindow)
    distance = view.opts['distance']
    qtbot.mouseClick(panel.zoom_in_button, QtCore.Qt.MouseButton.LeftButton)
    assert view.opts['distance'] == pytest.approx(distance * .8)
    qtbot.mouseClick(panel.zoom_out_button, QtCore.Qt.MouseButton.LeftButton)
    assert view.opts['distance'] == pytest.approx(distance)
    qtbot.keyClick(panel, QtCore.Qt.Key.Key_Equal, QtCore.Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: abs(view.opts['distance'] - distance * .8) < 1e-6)
    qtbot.keyClick(panel, QtCore.Qt.Key.Key_Minus, QtCore.Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: abs(view.opts['distance'] - distance) < 1e-6)
    panel.set_language('CN')
    assert panel.zoom_in_button.shortcut().toString() == 'Ctrl+='
    assert panel.zoom_out_button.shortcut().toString() == 'Ctrl+-'
    assert panel.zoom_in_button.text() == '放大'
    assert panel.zoom_out_button.text() == '缩小'
    panel.set_language('EN')
    assert panel.zoom_in_button.text() == 'Zoom in'
    assert panel.zoom_out_button.text() == 'Zoom out'
    view.zoom(.5)
    qtbot.mouseClick(panel.reset_button, QtCore.Qt.MouseButton.LeftButton)
    assert view.opts['distance'] == pytest.approx(distance)
    panel.clear_result()
    assert not panel.zoom_in_button.isEnabled()
    assert not panel.zoom_out_button.isEnabled()
