import numpy as np
from shared.data_types import LFIRawData, raw_to_dict, dict_to_raw


def test_raw_dict_roundtrip():
    raw = LFIRawData(amplitude=np.ones((2, 3, 4)), phase=np.zeros((2, 3, 4)),
                     z_coords=np.array([0.0, 0.1]), target_type="Leaf", file_path="scan.hdf5")
    d = raw_to_dict(raw)
    assert set(d) == {"amplitude", "phase", "x_coords", "y_coords", "z_coords", "target_type",
                      "scan_params", "timestamp", "file_path"}
    back = dict_to_raw(d)
    assert back.target_type == "Leaf" and back.file_path == "scan.hdf5"
    np.testing.assert_array_equal(back.z_coords, [0.0, 0.1])
    assert back.num_z_slices == 2
