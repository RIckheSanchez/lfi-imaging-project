# Native arm64 macOS bundle. Build with an arm64 Python environment.
from pathlib import Path
root = Path(SPECPATH)
a = Analysis([str(root / 'packaging' / 'entrypoint.py')],
             pathex=[str(root), str(root / 'packaging')],
             binaries=[], datas=[], hiddenimports=['part4_ui', 'release_smoke'],
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['streamlit', 'plotly', 'pandas', 'pytest', 'pytestqt',
                       'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
                       'PySide6.QtWebEngineQuick', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LFI Viewer',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False, target_arch='arm64', codesign_identity=None, entitlements_file=None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='LFI Viewer')
app = BUNDLE(coll, name='LFI Viewer.app', icon=None,
             bundle_identifier='au.edu.uq.lfi.viewer',
             info_plist={'CFBundleShortVersionString': '1.0',
                         'NSHighResolutionCapable': True})
