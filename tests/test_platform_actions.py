"""Platform reveal and shortcut behavior without invoking the OS file manager."""
import ntpath
from pathlib import Path
import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets
import part4_ui.app as module
from part4_ui.lfi_core.data_types import LFIRawData


@pytest.fixture
def window(qtbot, tmp_path):
    widget = module.MainWindow(QtCore.QSettings(str(tmp_path/'platform.ini'), QtCore.QSettings.Format.IniFormat))
    qtbot.addWidget(widget)
    return widget


@pytest.mark.parametrize('platform,en,cn', [
    ('darwin', 'Show in Finder', '在 Finder 中显示'),
    ('win32', 'Show in Explorer', '在资源管理器中显示'),
    ('linux', 'Show in folder', '在文件夹中显示'),
])
def test_reveal_dispatch_and_bilingual_label(window, monkeypatch, tmp_path, platform, en, cn):
    monkeypatch.setattr(module.sys, 'platform', platform)
    commands, urls = [], []
    monkeypatch.setattr(module.subprocess, 'Popen', lambda args: commands.append(args))
    monkeypatch.setattr(module.QtGui.QDesktopServices, 'openUrl', lambda url: urls.append(url) or True)
    path = str(tmp_path/'export folder'/'phase image.csv')
    if platform == 'win32':
        path = 'C:/Users/User Name/Desktop/export folder/phase image.csv'
    window.exported_path = path
    for language, label in [('EN', en), ('CN', cn)]:
        window.set_language(language)
        assert window.finder_button.text() == label
        assert window.fit_action.shortcut().toString() == 'Ctrl+0'
        assert window.csv_action.shortcut().toString() == 'Ctrl+Shift+E'
    window.show_in_finder()
    if platform == 'darwin':
        assert commands == [['open', '-R', path]]
        assert not urls
    elif platform == 'win32':
        assert commands == [['explorer', '/select,' + ntpath.normpath(path)]]
        assert commands[0][1] == r'/select,C:\Users\User Name\Desktop\export folder\phase image.csv'
        assert not urls
    else:
        assert not commands
        assert len(urls) == 1 and urls[0].isLocalFile()
        assert Path(urls[0].toLocalFile()) == Path(path).parent.resolve()


@pytest.mark.parametrize('platform', ['darwin', 'win32', 'linux'])
def test_reveal_launch_errors_are_status_messages(window, monkeypatch, tmp_path, platform):
    monkeypatch.setattr(module.sys, 'platform', platform)
    def failed(*args):
        raise OSError('file manager unavailable')
    monkeypatch.setattr(module.subprocess, 'Popen', failed)
    monkeypatch.setattr(module.QtGui.QDesktopServices, 'openUrl', failed)
    window.exported_path = str(tmp_path/'phase.csv')
    window.show_in_finder()
    assert 'Could not show the exported file' in window.statusBar().currentMessage()
    assert 'file manager unavailable' in window.statusBar().currentMessage()


def test_folder_reveal_rejection_is_reported_in_chinese(window, monkeypatch, tmp_path):
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    monkeypatch.setattr(module.QtGui.QDesktopServices, 'openUrl', lambda url: False)
    window.set_language('CN')
    window.exported_path = str(tmp_path/'phase.csv')
    window.show_in_finder()
    assert window.statusBar().currentMessage() == '无法显示导出的文件'


def test_reveal_without_export_does_nothing(window, monkeypatch):
    def unexpected(*args):
        pytest.fail('File manager called without an exported path')
    monkeypatch.setattr(module.subprocess, 'Popen', unexpected)
    monkeypatch.setattr(module.QtGui.QDesktopServices, 'openUrl', unexpected)
    window.show_in_finder()


@pytest.mark.parametrize('desktop_exists', [True, False])
def test_export_directory_fallback(window, monkeypatch, tmp_path, desktop_exists):
    data = np.ones((1, 2, 3))
    window.data = LFIRawData(data, data, [10], 'test', 'test.hdf5')
    desktop = tmp_path/'Desktop'
    if desktop_exists:
        desktop.mkdir()
    window.settings.setValue('export_directory', str(tmp_path/'missing-saved-directory'))
    monkeypatch.setattr(QtCore.QStandardPaths, 'writableLocation', lambda location: str(desktop) if desktop_exists else '')
    calls = []
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getSaveFileName', lambda *args: calls.append(args) or ('', ''))
    window.export_csv_dialog()
    expected = desktop if desktop_exists else Path.home()
    assert Path(calls[0][2]).parent == expected
