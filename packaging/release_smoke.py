"""Opt-in packaged-app validation; never active during normal use.

Set LFI_VALIDATE_BUNDLE to an output directory and LFI_VALIDATE_FILE to a scan.
This launches the real Cocoa window, saves QWidget/GL framebuffer captures, and
records failures instead of hiding OpenGL errors. No data or graphics are mocked.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path
from PySide6 import QtCore, QtWidgets
import part4_ui.app as desktop


def install():
    original = desktop.MainWindow
    class ValidationWindow(original):
        def __init__(self):
            super().__init__()
            self.output = Path(os.environ['LFI_VALIDATE_BUNDLE'])
            self.output.mkdir(parents=True, exist_ok=True)
            self.report = {'architecture': platform.machine(), 'frozen': bool(getattr(sys,'frozen',False)),
                           'executable': sys.executable, 'qt_platform': QtWidgets.QApplication.platformName(),
                           'errors': [], 'qt_messages': []}
            self.set_language('EN')
            self.started = time.monotonic()
            self.stage = 'load'
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.check)
            self.timer.start()
            self.old_handler = QtCore.qInstallMessageHandler(self.qt_message)
            QtCore.QTimer.singleShot(0, lambda: self.open_file(os.environ['LFI_VALIDATE_FILE']))

        def qt_message(self, kind, context, message):
            self.report['qt_messages'].append(message)
            print(message, flush=True)

        def show_error(self, message):
            self.report['errors'].append(str(message))
            print('VALIDATION ERROR:', message, flush=True)

        def finish(self):
            self.timer.stop()
            self.report['elapsed_seconds'] = time.monotonic()-self.started
            (self.output/'bundle-report.json').write_text(json.dumps(self.report, indent=2))
            QtCore.qInstallMessageHandler(self.old_handler)
            print(json.dumps(self.report, indent=2), flush=True)
            QtWidgets.QApplication.quit()

        def capture(self, name):
            scene = self.volume_panel.scene
            record = {'context_valid': bool(scene and scene.isValid())}
            if scene and scene.isValid():
                frame = scene.grabFramebuffer()
                record['framebuffer_saved'] = frame.save(str(self.output/(name+'-framebuffer.png')))
                colors = {frame.pixelColor(x,y).rgba() for x in range(0,frame.width(),max(1,frame.width()//50))
                          for y in range(0,frame.height(),max(1,frame.height()//50))}
                record['sampled_framebuffer_colors'] = len(colors)
            record['window_saved'] = self.grab().save(str(self.output/(name+'.png')))
            self.report[name] = record

        def check(self):
            try:
                if time.monotonic()-self.started > 90:
                    self.report['errors'].append('Packaged application validation timed out.')
                    self.finish()
                    return
                if self.jobs:
                    return
                if self.report['errors']:
                    self.finish()
                    return
                if self.stage == 'load' and self.raw_slice is not None:
                    self.report['preprocessing-off'] = {
                        'shape': list(self.data.amplitude.shape), 'amplitude_label': self.amplitude_label(),
                        'phase_converted': self.data.metadata.get('phase_converted'),
                        'elapsed_seconds': time.monotonic()-self.started,
                        'saved': self.grab().save(str(self.output/'bundle-preprocessing-off.png'))}
                    self.z_values.setCurrentIndex(min(30, self.data.n_slices - 1))
                    self.pixel_x.setValue(min(700, self.data.width - 1))
                    self.pixel_y.setValue(min(200, self.data.height - 1))
                    self.grab().save(str(self.output/'bundle-2d.png'))
                    self.report['preprocessing_control_visible'] = self.pre_group.isVisible()
                    self.report['developer_parameters_hidden'] = self.parameter.isHidden() and self.parameter_label.isHidden()
                    self.report['preprocess_params'] = self.preprocess_params
                    self.preprocess_started = time.monotonic()
                    self.apply_preprocessing.setChecked(True)
                    self.stage = 'preprocess'
                elif self.stage == 'preprocess' and self.raw_slice is not None:
                    self.report['preprocessing-on'] = {
                        'shape': list(self.data.amplitude.shape), 'amplitude_label': self.amplitude_label(),
                        'colorbar_label': self.colorbar.axis.labelText,
                        'phase_converted': self.data.metadata.get('phase_converted'),
                        'product': self.data.metadata['ui_preprocessing'],
                        'elapsed_seconds': time.monotonic()-self.preprocess_started,
                        'saved': self.grab().save(str(self.output/'bundle-preprocessing-on.png'))}
                    self.grab().save(str(self.output/'bundle-2d.png'))
                    self.set_language('CN')
                    QtWidgets.QApplication.processEvents()
                    self.grab().save(str(self.output/'bundle-2d-cn.png'))
                    self.set_language('EN')
                    self.tabs.setCurrentIndex(1)
                    self.stage = 'volume-compute'
                elif self.stage.endswith('-compute') and self.volume_panel.result is not None:
                    self.paint_after = time.monotonic()+2
                    self.stage = self.stage.replace('-compute', '-paint')
                elif self.stage == 'volume-paint' and time.monotonic() >= self.paint_after:
                    self.capture('bundle-volume')
                    self.report['reconstruction_control_hidden'] = self.volume_panel.reconstruct.isHidden()
                    self.report['reconstruction_hint_visible'] = self.volume_panel.reconstruction_hint.isVisible()
                    self.report['reconstruction_hint'] = self.volume_panel.reconstruction_hint.text()
                    self.volume_panel.mode.setCurrentIndex(1)
                    self.stage = 'isosurface-compute'
                elif self.stage == 'isosurface-paint' and time.monotonic() >= self.paint_after:
                    self.capture('bundle-isosurface')
                    self.volume_panel.percentile.setValue(80)
                    self.report['automatic-percentile-busy'] = bool(self.volume_panel.busy.text())
                    self.stage = 'percentile-compute'
                elif self.stage == 'percentile-paint' and time.monotonic() >= self.paint_after:
                    self.capture('bundle-percentile')
                    self.stage = 'export'
                    self.tabs.setCurrentIndex(0)
                    self.channel.setCurrentIndex(1)
                elif self.stage == 'export' and self.raw_slice is not None:
                    # Exercise the real direct-export entry point with a Qt save
                    # dialog so its complete title and content can be captured.
                    self.timer.stop()
                    original_save = QtWidgets.QFileDialog.getSaveFileName
                    def capture_save(parent, title, filename, filters):
                        dialog = QtWidgets.QFileDialog(parent, title, filename, filters)
                        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)
                        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
                        dialog.resize(1100, 600)
                        def capture_dialog():
                            self.report['export-summary'] = self.export_summary()
                            self.report['export-dialog-title'] = title
                            self.report['export-dialog-qt'] = dialog.grab().save(str(self.output/'bundle-export-dialog.png'))
                            dialog.reject()
                        QtCore.QTimer.singleShot(800, capture_dialog)
                        dialog.exec()
                        return '', ''
                    QtWidgets.QFileDialog.getSaveFileName = capture_save
                    try:
                        self.export_csv_dialog()
                    finally:
                        QtWidgets.QFileDialog.getSaveFileName = original_save
                    self.finish()
            except Exception as error:
                self.report['errors'].append(str(error))
                self.finish()
    desktop.MainWindow = ValidationWindow
