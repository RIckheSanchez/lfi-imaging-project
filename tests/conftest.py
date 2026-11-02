import os
import tempfile
from pathlib import Path
import pytest
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'lfi-mpl'))


@pytest.fixture(scope='session')
def real_file():
    """Local research fixtures are optional on clean checkouts and CI runners."""
    def locate(name):
        path = Path(__file__).resolve().parents[1] / 'Realdata' / name
        if not path.is_file():
            pytest.skip(f'Local real-data fixture unavailable: {name}')
        return path
    return locate
