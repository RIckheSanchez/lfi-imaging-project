"""PyInstaller entry point; normal runs use only the desktop frontend."""
import os
from part4_ui.app import main

if os.environ.get('LFI_VALIDATE_BUNDLE'):
    from release_smoke import install
    install()

raise SystemExit(main())
