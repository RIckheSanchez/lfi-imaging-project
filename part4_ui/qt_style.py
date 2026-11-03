"""Pure platform/default selection; Qt is applied by the application entrypoint."""


def select_qt_style(platform, override=None):
    """Preserve native styling on every platform unless explicitly overridden."""
    if override and override.strip():
        return override.strip()
    return None
