"""Frontend independent HDF5 loading and source identities."""
import os
import tempfile
import hashlib
from part1_data import load_single_scan
from .data_types import LFIRawData

def local_data_key(filepath):
    """Identify the resolved local file by path, modification time, and size."""
    path = os.path.realpath(os.path.expanduser(os.fspath(filepath)))
    return path, os.path.getmtime(path), os.path.getsize(path)


def upload_data_key(name, data):
    """Identify upload bytes independently of the temporary file lifetime."""
    return "upload", name, len(data), hashlib.sha1(data).hexdigest()


def adapt_raw_data(source):
    """Explicitly adapt Part 1's public result, retaining all metadata."""
    return LFIRawData(
        amplitude=source.amplitude,
        phase=source.phase,
        z_indices=list(source.z_indices),
        target_type=source.target_type,
        file_path=source.file_path,
        metadata=dict(source.metadata),
    )


def smart_load(filepath: str):
    """Load through Part 1's public API; propagate its errors without fallback."""
    return adapt_raw_data(load_single_scan(filepath, include_waveform=False))


def load_uploaded_bytes(name, data):
    """Load upload bytes and return a stable key; always clean up the temporary file."""
    data_key = upload_data_key(name, data)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5') as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(data)
        loaded_data = smart_load(tmp_path)
        if loaded_data.target_type == 'Unknown':
            known_types = ['Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom',
                           'Sound', 'Tomography', 'Beam Profile', 'Spectrum']
            loaded_data.target_type = next(
                (target for target in known_types if target.lower() in name.lower()), 'Unknown')
        return loaded_data, data_key
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
