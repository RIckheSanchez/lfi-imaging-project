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
                    if self.channel.currentIndex() != 1:
                        self.channel.setCurrentIndex(1)
                        return
                    self.report['2d'] = {'channel': 'phase', 'phase_label': self.phase_label(),
                                         'phase_metadata': {key: self.data.metadata.get(key) for key in ('phase_encoding_used', 'phase_converted')},
                                         'shape': list(self.data.amplitude.shape),
                                         'raw_sum': float(self.raw_slice.sum()),
                                         'saved': self.grab().save(str(self.output/'bundle-2d.png'))}
                    self.tabs.setCurrentIndex(1)
                    self.request_volume()
                    self.stage = 'volume-compute'
                elif self.stage.endswith('-compute') and self.volume_panel.result is not None:
                    self.paint_after = time.monotonic()+2
                    self.stage = self.stage.replace('-compute', '-paint')
                elif self.stage == 'volume-paint' and time.monotonic() >= self.paint_after:
                    self.capture('bundle-volume')
                    self.volume_panel.mode.setCurrentIndex(1)
                    self.request_volume()
                    self.stage = 'isosurface-compute'
                elif self.stage == 'isosurface-paint' and time.monotonic() >= self.paint_after:
                    self.capture('bundle-isosurface')
                    self.finish()
            except Exception as error:
                self.report['errors'].append(str(error))
                self.finish()
    desktop.MainWindow = ValidationWindow
