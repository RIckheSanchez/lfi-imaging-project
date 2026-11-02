# Windows onedir/windowed bundle. Build on Windows with Python 3.11.
from pathlib import Path
root = Path(SPECPATH)
a = Analysis([str(root / 'packaging' / 'entrypoint.py')],
             pathex=[str(root), str(root / 'packaging'), str(root / 'part2_preprocess' / 'src')],
             binaries=[], datas=[(str(root / 'part2_preprocess/config/default.toml'), 'part2_preprocess/config')],
             hiddenimports=['part4_ui', 'release_smoke', 'part3_smoke', 'thz_preprocessing',
                            'part3_reconstruction.gui_backend', 'part3_reconstruction.handoff'],
             hookspath=[], hooksconfig={}, runtime_hooks=[],
             excludes=['streamlit', 'plotly', 'pandas', 'pytest', 'pytestqt',
                       'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets',
                       'PySide6.QtWebEngineQuick', 'PySide6.QtPdf', 'PySide6.QtPdfWidgets'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='LFI Viewer',
          debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
          console=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='LFI Viewer')
