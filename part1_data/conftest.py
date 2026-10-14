"""Pytest bootstrap: allow flat imports when collecting from the repo root.

Enables::

    pytest part1_data          # from team repo root
    pytest -v tests/           # from inside part1_data/

without ``No module named 'loader'``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
