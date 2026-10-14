"""Registered path loaders shared by every desktop opening action."""
from pathlib import Path
from typing import Protocol
from .data_types import LFIRawData
from .data_io import smart_load, adapt_raw_data, local_data_key


class Loader(Protocol):
    name: str
    path_types: tuple[str, ...]
    extensions: tuple[str, ...]

    def load(self, path: str) -> LFIRawData:
        """Read one path using public APIs; raise a readable error."""
        ...

    def make_data_key(self, path: str) -> tuple:
        """Identify every input and calibration resource affecting the result."""
        ...


class HDF5Loader:
    name = 'hdf5'
    path_types = ('file',)
    extensions = ('.h5', '.hdf5')

    def load(self, path):
        return smart_load(str(path))

    def make_data_key(self, path):
        return local_data_key(path)


class FrameFolderLoader:
    name = 'frame-folder'
    path_types = ('dir',)
    extensions = ('.h5', '.hdf5', '.mat')

    def load(self, path):
        from part1_data import load_volume
        return adapt_raw_data(load_volume(str(path), include_waveform=False))

    def make_data_key(self, path):
        # Include all data files, even nested/unused frames, and depth sidecars.
        entries = []
        for file in Path(path).rglob('*'):
            if file.is_file() and (file.suffix.lower() in self.extensions or
                                   file.name == 'experiment_details.txt'):
                resolved = file.resolve()
                stat = resolved.stat()
                entries.append((str(resolved), stat.st_mtime_ns, stat.st_size))
        return tuple(sorted(entries))


_loaders = {}


def register_loader(loader: Loader):
    """Register a named loader before creating windows; duplicate names fail.

    Extension matching is case-insensitive. Existing HDF5 behavior is preserved;
    plugins should claim distinct extensions/path types instead of patching it.
    """
    if not loader.name or loader.name in _loaders:
        raise ValueError(f'Loader name is empty or already registered: {loader.name}')
    if not loader.path_types or any(kind not in ('file', 'dir') for kind in loader.path_types):
        raise ValueError('Loader path_types must contain file and/or dir.')
    if not callable(getattr(loader, 'load', None)) or not callable(getattr(loader, 'make_data_key', None)):
        raise TypeError('Loader must implement load(path) and make_data_key(path).')
    if 'file' in loader.path_types and not loader.extensions:
        raise ValueError('File loaders must declare extensions.')
    _loaders[loader.name] = loader


def get_loaders():
    """Return an immutable snapshot of registered loader implementations."""
    return tuple(_loaders.values())


def loader_for(path):
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f'Path not found: {resolved}')
    kind = 'dir' if resolved.is_dir() else 'file'
    for loader in get_loaders():
        if kind in loader.path_types and (kind == 'dir' or
                resolved.suffix.lower() in tuple(ext.lower() for ext in loader.extensions)):
            return loader
    raise ValueError(f'No registered loader supports this path: {resolved}')


def load_path(path):
    """Dispatch on a worker; identity includes path, loader name and its inputs."""
    resolved = str(Path(path).expanduser().resolve())
    loader = loader_for(resolved)
    key = (resolved, loader.name, loader.make_data_key(resolved))
    return loader.load(resolved), key


register_loader(HDF5Loader())
register_loader(FrameFolderLoader())
