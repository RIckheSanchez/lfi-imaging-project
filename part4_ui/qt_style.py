"""Pure platform/default selection; Qt is applied by the application entrypoint."""


def select_qt_style(platform, override=None):
    """Preserve the native style except for the Windows spin-box workaround."""
    if override and override.strip():
        return override.strip()
    return 'Fusion' if platform == 'win32' else None
