"""Sound .mat (mat_vars required) + metadata sidecar / HDF5 attrs."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from loader import load_single_scan
from output_interface import (
    write_metadata_sidecar,
    write_metadata_hdf5_attrs,
)


def test_sound_requires_mat_vars(tmp_path):
    """Sound has no verified layout — guessing is forbidden."""
    amp = np.random.randn(4, 5, 3).astype(np.float64)
    phase = np.random.randint(0, 100, size=(4, 5, 3)).astype(np.int16)
    fp = tmp_path / "Sound" / "sample.mat"
    fp.parent.mkdir()
    scipy.io.savemat(fp, {"S_A": amp, "S_P": phase})

    with pytest.raises(ValueError) as ei:
        load_single_scan(str(fp))
    msg = str(ei.value).lower()
    assert "mat_vars" in msg
    assert "unverified" in msg or "no verified" in msg


def test_sound_mat_loads_unconfirmed_with_mat_vars(tmp_path):
    amp = np.random.randn(4, 5, 3).astype(np.float64)
    phase = np.random.randint(0, 100, size=(4, 5, 3)).astype(np.int16)
    fp = tmp_path / "Sound" / "sample.mat"
    fp.parent.mkdir()
    scipy.io.savemat(fp, {"S_A": amp, "S_P": phase})

    data = load_single_scan(
        str(fp),
        mat_vars={"amplitude": "S_A", "phase": "S_P", "stack_axis": -1},
    )
    assert data.target_type == "Sound"
    assert data.metadata["phase_encoding_used"] == "sound_unconfirmed"
    assert data.metadata["phase_converted"] is False
    assert data.amplitude.shape == (3, 4, 5)


def test_sound_wrong_vars_lists_available(tmp_path):
    fp = tmp_path / "Sound" / "bad.mat"
    fp.parent.mkdir()
    scipy.io.savemat(fp, {"foo": np.ones((2, 2)), "bar": np.ones((2, 2)), "baz": np.ones((2, 2))})
    with pytest.raises(ValueError) as ei:
        load_single_scan(str(fp), mat_vars={"amplitude": "nope", "phase": "nope2"})
    msg = str(ei.value)
    assert "foo" in msg and "bar" in msg


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_single_scan("/tmp/definitely_missing_lfi_file_xyz.hdf5")


def test_unsupported_extension(tmp_path):
    fp = tmp_path / "x.txt"
    fp.write_text("nope")
    with pytest.raises(ValueError):
        load_single_scan(str(fp))


def test_leaf_skips_depth_metadata(tmp_path):
    """Leaf time series must not pick up experiment_details depth_mm."""
    amp = np.ones((2, 3, 4), dtype=np.float64)
    phase = np.zeros((2, 3, 4), dtype=np.int16)
    leaf_dir = tmp_path / "Leaf"
    leaf_dir.mkdir()
    (leaf_dir / "experiment_details.txt").write_text(
        "z_start_mm = 0.0\nz_step_mm = 0.1\nfirst_frame_id = 0\n",
        encoding="utf-8",
    )
    mat = leaf_dir / "tiny.mat"
    scipy.io.savemat(
        mat,
        {
            "L_A": np.moveaxis(amp, 0, -1),
            "L_P": np.moveaxis(phase, 0, -1),
        },
    )
    data = load_single_scan(str(mat))
    assert data.metadata["scan_mode"] == "time_series"
    assert data.metadata.get("depth_mm") is None
    assert "Karl, email, 2026-10-14" in (data.metadata.get("notes") or "")


def test_metadata_sidecar_requires_path(tmp_path):
    amp = np.ones((2, 3, 4), dtype=np.float64)
    phase = np.zeros((2, 3, 4), dtype=np.int16)
    mat = tmp_path / "Leaf" / "tiny.mat"
    mat.parent.mkdir()
    scipy.io.savemat(
        mat,
        {
            "L_A": np.moveaxis(amp, 0, -1),
            "L_P": np.moveaxis(phase, 0, -1),
        },
    )
    data = load_single_scan(str(mat))
    assert data.metadata["scan_mode"] == "time_series"

    with pytest.raises(ValueError, match="path"):
        write_metadata_sidecar(data)

    side = write_metadata_sidecar(data, path=str(tmp_path / "tiny.meta.json"))
    rec = json.loads(Path(side).read_text(encoding="utf-8"))
    assert rec["metadata"]["scan_mode"] == "time_series"
    assert "schema_version" in rec

    h5p = str(tmp_path / "tiny_meta.h5")
    write_metadata_hdf5_attrs(data, h5p)
    with h5py.File(h5p, "r") as f:
        assert f.attrs["target_type"] == "Leaf"
        assert "phase_encoding_used" in f.attrs
