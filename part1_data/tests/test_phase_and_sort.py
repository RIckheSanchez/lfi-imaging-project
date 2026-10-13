"""pytest: phase formulas, numeric sort, invalid inputs."""

import numpy as np
import pytest

from loader import sort_data_paths, frame_number_from_name
from phase_encoding import convert_phase, get_phase_encoding
from output_interface import get_phase_radians, PhaseNotConvertedError
from data_types import LFIRawData


def test_vegetables_and_leaf_formulas():
    codes = np.array([-8192, 0, 8191], dtype=np.int16)
    rad, enc, ok = convert_phase(codes, "vegetables_int16")
    assert ok and enc.name == "vegetables_int16"
    assert np.isclose(rad[0], -np.pi)
    assert np.isclose(rad[1], 0.0)

    leaf = np.array([0, 4096], dtype=np.int16)
    lrad, _, ok2 = convert_phase(leaf, "leaf_wrap_4096")
    assert ok2
    assert np.isclose(lrad[1], 0.0)


def test_unconfirmed_does_not_convert():
    codes = np.array([1, 2, 3], dtype=np.uint16)
    out, enc, ok = convert_phase(codes, "skin_unconfirmed")
    assert ok is False
    assert enc.confirmed is False
    np.testing.assert_array_equal(out, codes)


def test_numeric_sort_not_lexical():
    names = [
        "frame-100.hdf5",
        "frame-21.hdf5",
        "frame-110.hdf5",
        "frame-22.hdf5",
    ]
    got = [p.name for p in sort_data_paths(names)]
    assert got == [
        "frame-21.hdf5",
        "frame-22.hdf5",
        "frame-100.hdf5",
        "frame-110.hdf5",
    ]
    assert frame_number_from_name("frame-28.hdf5") == 28


def test_invalid_encoding_name():
    with pytest.raises(KeyError):
        get_phase_encoding("not_a_real_encoding")


def test_get_phase_radians_rejects_raw():
    data = LFIRawData(
        amplitude=np.zeros((1, 2, 3)),
        phase=np.ones((1, 2, 3), dtype=np.uint16),
        z_indices=[0],
        target_type="Skin",
        file_path="dummy.hdf5",
        metadata={
            "phase_converted": False,
            "phase_encoding_used": "skin_unconfirmed",
        },
    )
    with pytest.raises(PhaseNotConvertedError):
        get_phase_radians(data)


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        LFIRawData(
            amplitude=np.zeros((1, 2, 3)),
            phase=np.zeros((1, 4, 5)),
            z_indices=[0],
            target_type="Chip",
            file_path="x",
        )
