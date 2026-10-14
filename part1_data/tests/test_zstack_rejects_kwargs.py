"""z-stack must not silently ignore frame_ids / qcl / side."""

import h5py
import numpy as np
import pytest

from loader import load_single_scan


def _write_zstack(path) -> None:
    with h5py.File(path, "w") as f:
        amp = f.create_group("amplitude")
        ph = f.create_group("phase")
        for z in (1, 2):
            amp.create_dataset(str(z), data=np.ones((3, 4), dtype=np.float64))
            ph.create_dataset(str(z), data=np.linspace(-10, 10, 12).reshape(3, 4))


def test_zstack_rejects_frame_ids_qcl_side(tmp_path):
    fp = tmp_path / "Chip" / "sample.hdf5"
    fp.parent.mkdir()
    _write_zstack(fp)

    with pytest.raises(ValueError, match="hdf5_zstack"):
        load_single_scan(str(fp), frame_ids=[1])
    with pytest.raises(ValueError, match="hdf5_zstack"):
        load_single_scan(str(fp), qcl="QCL0")
    with pytest.raises(ValueError, match="hdf5_zstack"):
        load_single_scan(str(fp), side="Left")

    # Still loads when those kwargs are omitted
    with pytest.warns(UserWarning, match="unconfirmed"):
        data = load_single_scan(str(fp))
    assert data.metadata["file_format"] == "hdf5_zstack"
    assert data.n_slices == 2
