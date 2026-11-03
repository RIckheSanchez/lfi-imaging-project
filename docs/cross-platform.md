# macOS and Windows support

Run from source on either platform with `python -m part4_ui.app` after installing
`requirements.txt`. Windows packaging uses `LFI Viewer Windows.spec` (onedir,
windowed); the existing arm64 `LFI Viewer.spec` is unchanged. Build Windows
artifacts on Windows; PyInstaller does not cross-compile the application here.
The Windows distribution is the entire `dist/LFI Viewer/` folder.

## Shortcuts and file reveal

- Fit to window: **Cmd/Ctrl+0**.
- Direct CSV export: **Cmd/Ctrl+Shift+E**.
- 3D zoom: **Cmd/Ctrl+=** and **Cmd/Ctrl+-**.

The code uses Qt `Ctrl+` sequences: Command on macOS, Ctrl on Windows.
Open and Quit retain Qt's platform-standard shortcuts. The export reveal label is
translated in English/Chinese as Show in Finder (macOS), Show in Explorer
(Windows), or Show in folder (other platforms).

Reveal launches `open -R <path>` on macOS or
`explorer /select,<native path>` on Windows. Other platforms open the containing
folder with QDesktopServices. Launch exceptions and a rejected folder-open
request become status-bar messages, not unhandled exceptions. Commands use
argument lists, not shell interpolation; spaces remain part of the path.

## Portability audit

Checked and addressed:

- All executable/test `Meta+` key sequences: replaced by `Ctrl+`; current shortcut
  guidance uses Cmd/Ctrl notation. Historical native-macOS logs/screenshots are
  retained as evidence rather than rewritten as Windows results.
- Desktop export location: already uses QStandardPaths. Added a home-directory
  fallback when Desktop is unavailable, and ignore a remembered directory that
  no longer exists (including a path saved on a different OS).
- File import/recent files/folder identities: use pathlib, os.path and Qt dialogs;
  no desktop runtime path is hard-coded to a user's machine.
- The retained Streamlit placeholder: changed from a specific `/Users/...` path
  to a repository-relative example. Forward slashes in that example work with
  Python pathlib on both platforms.
- Path separators: Windows reveal explicitly normalizes to backslashes with
  ntpath; other paths remain handled by pathlib/Qt. Unicode and spaces are not
  passed through a shell.
- Fonts: no named macOS-only font family in the desktop UI. Qt uses the platform
  font; pyqtgraph changes font size and palette colors only. No font override
  was necessary.
- Test caches and missing-file paths: use tempfile/pytest tmp_path instead of a
  POSIX `/tmp` directory. Optional real-data tests skip explicitly if the local
  Git-ignored research files are absent; synthetic tests continue to execute.
- Test screenshots/timing JSON: chip and preprocessing captures now go to each
  test's tmp_path. Ordinary pytest runs never rewrite `docs/screenshots/`.
- Optional diagnostics: legacy viewer verification invokes system_profiler only
  on macOS.
- Known issues in teammate-owned modules (not changed here; reported to the
  module owners): some optional real-data tests in `part1_data/` and
  `part2_preprocess/tests/` create symlinks, which need administrator or
  developer mode on Windows; `part1_data/benchmark.py` imports the POSIX-only
  `resource` module. These tests skip in CI because the research data are not
  in the repository.
- Upload temporary files: already closed before h5py reopens them, and deleted
  in finally; no Windows file-locking change was necessary.
- Worker pools, backend adapters, image/CSV exports, OpenGL/offscreen gating and
  the bundled Part 2 TOML: inspected; no macOS-only runtime API was found there.
- `run.sh` and the macOS spec remain intentional POSIX/macOS conveniences;
  Windows instructions use the Python module and the separate Windows spec.

## CI and validation boundaries

`.github/workflows/ci.yml` installs runtime/development dependencies, runs the full
pytest suite offscreen on macos-latest and windows-latest with Python 3.11, and
checks that tests leave tracked files unchanged. Windows additionally builds the
onedir bundle and uploads it as `LFI-Viewer-Windows`.

Platform dispatch is tested by mocking sys.platform, subprocess and
QDesktopServices. Local tests cannot establish Windows-native behavior or a
successful Windows bundle build; those require the Windows CI/native run.
No historical macOS validation is presented as Windows validation.

## Native Windows verification

On 4 November 2026 the CI-built `LFI-Viewer-Windows` bundle (commit `9dcf8b6`,
GitHub Actions run 41655868750) was tested manually on a physical Windows 11
machine with Chinese regional settings. The full checklist passed: launch,
language switching, file open (dialog, drag and drop, recent files), 2D slice
navigation and display controls, pixel selection and waveform, preprocessing,
3D rendering, rotation and zoom (mouse, touchpad, buttons and shortcuts), image
and CSV export (including the CSV header), 3D view PNG export, Show in Explorer,
and the Ctrl+0 / Ctrl+Shift+E shortcuts. This testing found UI-004 (native
digit substitution in spin boxes), which is fixed and re-verified.


## Local verification

On the current macOS host, using Python 3.13.5 and the installed runtime/test
libraries (plus nibabel in a temporary dependency directory):

- Full pytest with local scans: **295 passed, 21 skipped, 0 failed**, 54 warnings,
  26.58 seconds.
- Full pytest in a temporary source-only checkout without `Realdata/`:
  **252 passed, 64 skipped, 0 failed**, 20 warnings, 12.76 seconds.
- Compared hashes for all tracked/intended project files and Git status before
  and after pytest: identical, including all tracked screenshot/timing evidence.
- Parsed workflow YAML and checked its OS matrix/Python 3.11 configuration.
  Checked the Windows spec's onedir/windowed/resource declarations without
  building it. The macOS spec's original hash is unchanged.

These are local macOS results, not Windows CI results. No Windows build,
GitHub workflow dispatch, commit, tag or merge was performed.

## Files changed in this request

- Runtime: `part4_ui/app.py`, `part4_ui/i18n.py`, `legacy/streamlit_app.py`.
- Packaging/CI: `LFI Viewer Windows.spec`, `.github/workflows/ci.yml`.
- Documentation: `README.md`, `docs/desktop-viewer.md`,
  `docs/usability-issues.md`, `docs/cross-platform.md`. Historical release notes
  (`docs/releases/v1.3-sus.md`) were deliberately left unchanged.
- UI/core/platform tests: `tests/conftest.py`, `tests/test_adapters.py`,
  `tests/test_core.py`, `tests/test_gui.py`, `tests/test_part3_integration.py`,
  `tests/test_platform_actions.py`.
- Diagnostics: `tests/verify_viewer.py`.

Pre-existing local edits to `docs/releases/v1.4-part3-integration.md` and
`docs/releases/v14-native-output.txt` were preserved byte-for-byte. The existing
macOS spec, scientific processing implementations and acquisition data were not
changed.


## Numeric locale policy and optional styles (UI-004)

The Windows spin-box issue was a numeric-locale problem, not a windows11-style
painting defect. With Chinese regional settings and native digit substitution,
Qt spin boxes displayed native numerals (sometimes rotated); switching to
Fusion did not fix it. Python-formatted labels were unaffected.

The normal entrypoint calls `QLocale.setDefault(QLocale.c())` before any widget
is created, on **every platform**. All QSpinBox/QDoubleSpinBox widgets also have
an explicit C locale, including pixel X/Y, slice index, pixel size and the 3D
percentile control. Qt numeric fields therefore use ASCII digits and `.` as
the decimal separator, even when MainWindow is created without main(). This
numeric policy does not change the app’s English/Chinese text selection.

The native application style is preserved on both Windows and macOS by default.
`select_qt_style` returns no style unless `LFI_QT_STYLE` has a nonblank value.
Fusion is **not** the Windows default. An explicit style override is still
available for optional appearance changes or diagnosis; it does not disable the
C numeric locale policy.

For an explicit style override in Windows PowerShell:

```powershell
$env:LFI_QT_STYLE = 'windows11'
python -m part4_ui.app
Remove-Item Env:LFI_QT_STYLE
```

To explicitly request Fusion on macOS:

```sh
LFI_QT_STYLE=Fusion python -m part4_ui.app
```

Use a style available on the host OS. Style selection tests cover native
defaults on all platforms and explicit overrides. App-level numeric-locale
tests set ar_EG and de_DE before constructing MainWindow, check every spin
box’s lineEdit().text(), and restore the default locale after each test.
The previous Fusion/dark-plot palette check remains an explicit override check,
not evidence that style selection fixes native digit substitution.

UI-004 remains **fixed, pending verification on Windows**. These locale tests
do not claim a native Windows reproduction or a confirmed Windows deployment.
