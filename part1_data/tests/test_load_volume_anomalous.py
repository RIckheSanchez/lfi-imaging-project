"""load_volume must skip anomalous single-frame files (Vegetables frame 20)."""

import warnings

import h5py
import numpy as np

from loader import load_volume
from output_interface import get_phase_radians


def _write_veg_frame(path, frame_id: int, h: int = 4, w: int = 5) -> None:
    """Minimal 2026 Vegetables single-frame HDF5."""
    name = f"frame_{frame_id}_qcl_0_left"
    amp = np.ones((h, w), dtype=np.float64)
    # Valid vegetables_int16 codes for normal frames; frame 20 content unused when skipped.
    phase = np.zeros((h, w), dtype=np.int16)
    if frame_id != 20:
        phase[:] = np.linspace(-100, 100, h * w, dtype=np.int16).reshape(h, w)
    else:
        phase[:] = -32000  # out of normal vegetables range
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.create_group("IMAGE_AMPLITUDE").create_dataset(name, data=amp)
        data.create_group("IMAGE_PHASE").create_dataset(name, data=phase)


def test_load_volume_skips_frame20_by_default(tmp_path):
    veg = tmp_path / "2026_Vegetables"
    veg.mkdir()
    for fid in (19, 20, 21):
        _write_veg_frame(veg / f"frame-{fid}.hdf5", fid)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vol = load_volume(str(veg))

    assert vol.amplitude.shape[0] == 2
    assert vol.z_indices == [19, 21]
    assert 20 in vol.metadata["anomalous_frame_ids"]
    assert vol.metadata["skipped_anomalous_frame_ids"] == [20]
    assert vol.metadata["phase_converted"] is True
    assert vol.metadata["phase_encoding_used"] == "vegetables_int16"
    # Phase is radians, not raw int16 codes
    rad = get_phase_radians(vol)
    assert rad.dtype == np.float64
    assert any("skip" in str(x.message).lower() for x in w)


def test_load_volume_include_anomalous_keeps_raw(tmp_path):
    veg = tmp_path / "2026_Vegetables"
    veg.mkdir()
    for fid in (19, 20, 21):
        _write_veg_frame(veg / f"frame-{fid}.hdf5", fid)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vol = load_volume(str(veg), exclude_anomalous=False)

    assert vol.amplitude.shape[0] == 3
    assert vol.z_indices == [19, 20, 21]
    assert 20 in vol.metadata["anomalous_frame_ids"]
