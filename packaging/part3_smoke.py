"""Opt-in native packaged validation, using real data and explicit environment inputs.

This is automated Cocoa validation, not a substitute for manual OS-dialog checks.
No calibration values are embedded here. Normal launches never import this module.
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
            self.output = Path(os.environ['LFI_VALIDATE_PART3'])
            self.output.mkdir(parents=True, exist_ok=True)
            self.report = {'frozen': bool(getattr(sys, 'frozen', False)),
                           'executable': sys.executable, 'architecture': platform.machine(),
                           'platform': QtWidgets.QApplication.platformName(), 'errors': [],
                           'validation_mode': 'automated Cocoa widget/worker validation'}
            self.stage = 'chip'
            self.started = time.monotonic()
            self.set_language('EN')
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.check)
            self.timer.start()
            QtCore.QTimer.singleShot(0, lambda: self.open_file(os.environ['LFI_PART3_CHIP']))

        def show_error(self, message):
            self.report['errors'].append(str(message))

        def capture(self, name):
            record = {'saved': self.grab().save(str(self.output/(name+'.png')))}
            scene = self.volume_panel.scene
            if scene is not None:
                record['context_valid'] = scene.isValid()
                image = scene.grabFramebuffer()
                record['framebuffer_saved'] = image.save(str(self.output/(name+'-framebuffer.png')))
                record['framebuffer_colors'] = len({image.pixelColor(x, y).rgba()
                    for x in range(0, image.width(), max(1, image.width()//50))
                    for y in range(0, image.height(), max(1, image.height()//50))})
            self.report[name] = record

        def finish(self):
            self.timer.stop()
            self.report['elapsed_seconds'] = time.monotonic()-self.started
            (self.output/'native-report.json').write_text(json.dumps(self.report, indent=2))
            print(json.dumps(self.report, indent=2), flush=True)
            QtWidgets.QApplication.quit()

        def check(self):
            try:
                if time.monotonic()-self.started > 120:
                    raise RuntimeError('Native validation timed out')
                if self.report['errors']:
                    self.finish()
                    return
                if self.jobs:
                    return
                panel = self.volume_panel
                if self.stage == 'chip' and self.raw_slice is not None:
                    assert panel.reconstruct.isHidden()
                    assert panel.reconstruction_hint.isHidden()
                    assert all(f.text() == '' for f in panel.calibration_fields.values())
                    assert panel.channel_notice.text() == '3D view uses the amplitude channel'
                    self.report['chip'] = {'shape': self.data.amplitude.shape,
                        'depth_mm': self.data.metadata.get('depth_mm'), 'reconstruct_hidden': True,
                        'hint_hidden': True, 'status': panel.channel_notice.text(),
                        'csv_shortcut': self.csv_action.shortcut().toString()}
                    self.capture('chip-2d')
                    self.tabs.setCurrentIndex(1)
                    self.stage = 'chip-compute'
                elif self.stage.endswith('-compute') and panel.result is not None:
                    self.paint_after = time.monotonic()+1
                    self.stage = self.stage.replace('-compute', '-paint')
                elif self.stage == 'chip-paint' and time.monotonic() >= self.paint_after:
                    self.capture('chip-3d-no-calibration')
                    # Invoke the actual public Open Folder path; only dialog selection
                    # is supplied by the harness. Folder loading itself is untouched.
                    old = QtWidgets.QFileDialog.getExistingDirectory
                    QtWidgets.QFileDialog.getExistingDirectory = lambda *args: os.environ['LFI_PART3_FOLDER']
                    try:
                        self.open_folder_dialog()
                    finally:
                        QtWidgets.QFileDialog.getExistingDirectory = old
                    self.stage = 'vegetables'
                elif self.stage == 'vegetables' and self.raw_slice is not None:
                    self.report['folder_path'] = os.environ['LFI_PART3_FOLDER']
                    self.report['folder_entry'] = 'MainWindow.open_folder_dialog -> registered FrameFolderLoader'
                    self.report['loaded_frame_ids'] = list(self.data.z_indices)
                    self.report['measured_shape'] = self.data.amplitude.shape
                    self.report['phase_converted'] = self.data.metadata.get('phase_converted')
                    panel.calibration_toggle.setChecked(True)
                    values = {'z_start_mm': os.environ['LFI_PART3_START'],
                              'z_step_mm': os.environ['LFI_PART3_STEP'],
                              'first_frame_id': os.environ['LFI_PART3_FIRST']}
                    self.report['user_inputs_for_validation'] = values
                    self.report['assumption'] = os.environ['LFI_PART3_ASSUMPTION']
                    self.annotation = QtWidgets.QLabel(self.report['assumption'])
                    self.annotation.setWordWrap(True)
                    panel.layout().insertWidget(1, self.annotation)
                    for key, value in values.items():
                        panel.calibration_fields[key].setText(value)
                    assert self.reconstruction_available()
                    panel.reconstruct.setChecked(True)
                    self.request_volume()
                    self.stage = 'reconstructed-compute'
                elif self.stage == 'reconstructed-paint' and time.monotonic() >= self.paint_after:
                    assert panel.result['ui_part3']['phase_policy'] == 'nearest_measured_slice'
                    assert 'axes in sample indices' in panel.channel_notice.text()
                    self.report['reconstructed_shape'] = panel.result['original_shape']
                    self.report['status'] = panel.channel_notice.text()
                    self.report['ui_part3'] = panel.result['ui_part3']
                    self.capture('vegetables-part3-assumption')
                    panel.mode.setCurrentIndex(1)
                    self.stage = 'isosurface-compute'
                elif self.stage == 'isosurface-paint' and time.monotonic() >= self.paint_after:
                    self.capture('vegetables-isosurface-assumption')
                    self.set_language('CN')
                    self.stage = 'cn-paint'
                    self.paint_after = time.monotonic()+1
                elif self.stage == 'cn-paint' and time.monotonic() >= self.paint_after:
                    self.capture('vegetables-part3-cn-assumption')
                    self.finish()
            except Exception as error:
                self.report['errors'].append(str(error))
                self.finish()
    desktop.MainWindow = ValidationWindow
