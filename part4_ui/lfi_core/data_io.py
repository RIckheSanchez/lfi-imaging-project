"""Frontend independent HDF5 loading and source identities."""
import os
import tempfile
import hashlib
import h5py
import numpy as np
from .loader import load_single_scan
from .data_types import LFIRawData

def local_data_key(filepath):
    """Identify the resolved local file by path, modification time, and size."""
    path = os.path.realpath(os.path.expanduser(os.fspath(filepath)))
    return path, os.path.getmtime(path), os.path.getsize(path)


def upload_data_key(name, data):
    """Identify upload bytes independently of the temporary file lifetime."""
    return "upload", name, len(data), hashlib.sha1(data).hexdigest()


def _load_frames_format(filepath: str):
    """Fallback loader for HDF5 files using /frames/.../IMAGE_AMPLITUDE structure."""
    with h5py.File(filepath, 'r') as f:
        frames_group = f['frames']
        frame_keys   = sorted(frames_group.keys(), key=lambda x: int(x))

        amp_slices, phase_slices, z_indices = [], [], []
        for k in frame_keys:
            base = frames_group[k]['Image Data']['QCL0']['Left']
            amp_slices.append(base['IMAGE_AMPLITUDE'][:].astype(np.float64))
            phase_slices.append(base['IMAGE_PHASE'][:].astype(np.float64))
            z_indices.append(int(k))

        amplitude = np.stack(amp_slices,   axis=0)
        phase     = np.stack(phase_slices, axis=0)
        metadata  = {k: v for k, v in f.attrs.items()}

    stem   = os.path.splitext(os.path.basename(filepath))[0]
    known  = ['Chip', 'Leaf', 'Skin', 'Marble', 'Mushroom', 'Sound', 'Tomography']
    target = next((tp for tp in known if tp.lower() in stem.lower()), 'Unknown')

    return LFIRawData(
        amplitude=amplitude, phase=phase, z_indices=z_indices,
        target_type=target, file_path=filepath, metadata=metadata
    )

def smart_load(filepath: str):
    """Try standard loader first; fall back to frames-format loader."""
    try:
        return load_single_scan(filepath)
    except ValueError as e:
        if 'frames' in str(e) or 'amplitude' in str(e):
            return _load_frames_format(filepath)
        raise

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
