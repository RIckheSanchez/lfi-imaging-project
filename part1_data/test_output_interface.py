"""
test_output_interface.py
tests for .mat support and output_interface.py

Usage:
    python3 test_output_interface.py
"""

import json
import os
import tempfile
import warnings
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from loader import load_single_scan, get_z_slices
from output_interface import (
    PhaseNotConvertedError,
    batch_export,
    export_scan,
    get_phase_radians,
    iter_scans,
    load_exported,
    scan_to_record,
)

_HERE = Path(__file__).resolve().parent
VEG_DIR = _HERE / "data" / "2026_Vegetables"
SKIN_FILE = os.environ.get("LFI_SKIN_FILE", "")


def _leaf_arrays(h=4, w=5, n=3, seed=0):
    rng = np.random.default_rng(seed)
    l_p = rng.integers(-8192, 8192, size=(h, w, n)).astype(np.int16)
    l_a = rng.random((h, w, n))
    return l_a, l_p


def _expected_leaf_phase(l_p):
    rad = l_p.astype(np.float64) * (np.pi / 4096.0) - np.pi
    return np.moveaxis((rad + np.pi) % (2 * np.pi) - np.pi, -1, 0)


def _section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def test_mat_v5_leaf():
    _section("test M1: Leaf .mat (v5) read + phase formula")
    l_a, l_p = _leaf_arrays()
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "Leaf" / "leaf_sample.mat"
        fp.parent.mkdir()
        scipy.io.savemat(fp, {"L_A": l_a, "L_P": l_p, "other": np.zeros(3)})

        data = load_single_scan(str(fp))
        m = data.metadata
        assert data.target_type == "Leaf"
        assert m["file_format"] == "mat"
        assert m["phase_encoding_used"] == "leaf_wrap_4096"
        assert m["phase_converted"] is True
        assert m["scan_mode"] == "time_series"
        assert data.amplitude.shape == (3, 4, 5)
        np.testing.assert_allclose(data.amplitude, np.moveaxis(l_a, -1, 0))
        np.testing.assert_allclose(data.phase, _expected_leaf_phase(l_p), atol=1e-12)
        assert get_z_slices(str(fp)) == [0, 1, 2]
        assert "other" in m["mat_variables"]
    print("\n✅ test_mat_v5_leaf passed!")


def test_mat_v73_leaf():
    _section("test M2: Leaf .mat (v7.3/HDF5) read; axis order matches v5")
    l_a, l_p = _leaf_arrays(seed=1)
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "Leaf" / "leaf_v73.mat"
        fp.parent.mkdir()
        # MATLAB v7.3 column-major; h5py axes appear transposed
        with h5py.File(fp, "w") as f:
            f["L_A"] = l_a.T
            f["L_P"] = l_p.T

        data = load_single_scan(str(fp))
        assert data.amplitude.shape == (3, 4, 5)
        np.testing.assert_allclose(data.amplitude, np.moveaxis(l_a, -1, 0))
        np.testing.assert_allclose(data.phase, _expected_leaf_phase(l_p), atol=1e-12)
        assert get_z_slices(str(fp)) == [0, 1, 2]
    print("\n✅ test_mat_v73_leaf passed!")


def test_mat_wrong_variable_names():
    _section("test M3: .mat errors clearly on bad names and lists variables")
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "Leaf" / "renamed.mat"
        fp.parent.mkdir()
        scipy.io.savemat(fp, {"amp": np.ones((4, 5)), "ph": np.ones((4, 5), dtype=np.int16)})
        try:
            load_single_scan(str(fp))
            raise AssertionError("mismatched variable names should error")
        except ValueError as e:
            assert "amp" in str(e) and "ph" in str(e)

        data = load_single_scan(str(fp), mat_vars={"amplitude": "amp", "phase": "ph"})
        assert data.amplitude.shape == (1, 4, 5)
    print("\n✅ test_mat_wrong_variable_names passed!")


def test_record_and_phase_guard():
    _section("test O1: scan_to_record is JSON-serializable; raw phase rejected")
    if not SKIN_FILE or not os.path.exists(SKIN_FILE):
        print("  skip: set LFI_SKIN_FILE to a local skin hdf5")
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skin = load_single_scan(SKIN_FILE, target_type="Skin")
    record = scan_to_record(skin)
    json.dumps(record)
    assert record["phase_units"] == "raw_code"
    try:
        get_phase_radians(skin)
        raise AssertionError("Skin phase unconfirmed; should reject")
    except PhaseNotConvertedError as e:
        assert "skin_unconfirmed" in str(e)
    print("\n✅ test_record_and_phase_guard passed!")


def test_export_roundtrip():
    _section("test O2: export_scan(arrays=True) → load_exported round-trip")
    l_a, l_p = _leaf_arrays()
    with tempfile.TemporaryDirectory() as tmp:
        fp = Path(tmp) / "Leaf" / "leaf_sample.mat"
        fp.parent.mkdir()
        scipy.io.savemat(fp, {"L_A": l_a, "L_P": l_p})
        data = load_single_scan(str(fp))

        paths = export_scan(data, Path(tmp) / "out", arrays=True)
        back = load_exported(paths["npz"])
        np.testing.assert_array_equal(back.amplitude, data.amplitude)
        np.testing.assert_array_equal(back.phase, data.phase)
        assert back.z_indices == data.z_indices
        assert back.metadata["phase_encoding_used"] == "leaf_wrap_4096"
        assert get_phase_radians(back) is back.phase

        on_disk = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
        assert on_disk["schema_version"] == "1.1"
        assert on_disk["shape"] == [3, 4, 5]
    print("\n✅ test_export_roundtrip passed!")


def test_batch_export_real_vegetables():
    _section("test O3: batch_export real Vegetables(frame 20 + 21)")
    f20, f21 = VEG_DIR / "frame-20.hdf5", VEG_DIR / "frame-21.hdf5"
    if not (f20.exists() and f21.exists()):
        print(f"  skip: {VEG_DIR} has no frame-20/21")
        return
    with tempfile.TemporaryDirectory() as tmp:
        sub = Path(tmp) / "2026_Vegetables"
        sub.mkdir()
        os.symlink(f20, sub / f20.name)
        os.symlink(f21, sub / f21.name)
        out = Path(tmp) / "exported"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            manifest = batch_export(str(sub), str(out))

        assert manifest["n_ok"] == 2 and manifest["n_failed"] == 0
        assert (out / "manifest.json").exists()
        assert (out / "metadata_index.csv").exists()
        rows = {Path(r["source"]).name: r for r in manifest["files"]}
        assert rows["frame-20.hdf5"]["anomalous_frame_ids"] == "[20]"
        assert rows["frame-20.hdf5"]["phase_units"] == "raw_code"
        assert rows["frame-21.hdf5"]["phase_units"] == "rad"

        rec21 = json.loads((out / "2026_Vegetables_frame-21.json").read_text(encoding="utf-8"))
        assert "mcp" in rec21["metadata"]["acquisition"]

        # iter_scans excludes anomalous by default: only frame 21; frame 20 recorded as error
        errors = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scans = list(iter_scans(str(sub), errors=errors))
        assert [s.z_indices for s in scans] == [[21]]
        assert len(errors) == 1 and "frame-20" in errors[0][0]
    print("\n✅ test_batch_export_real_vegetables passed!")


def test_iter_scans_filters_before_loading():
    _section("test O4: iter_scans(target_type=...) filters by path before reading other datasets")
    with tempfile.TemporaryDirectory() as tmp:
        leaf = Path(tmp) / "Leaf"
        leaf.mkdir()
        (leaf / "broken.mat").write_bytes(b"not a mat file")
        veg = Path(tmp) / "2026_Vegetables"
        veg.mkdir()
        if (VEG_DIR / "frame-21.hdf5").exists():
            os.symlink(VEG_DIR / "frame-21.hdf5", veg / "frame-21.hdf5")

        errors = []
        scans = list(iter_scans(tmp, target_type="Vegetables", errors=errors))
        assert errors == [], errors
        assert all(s.target_type == "Vegetables" for s in scans)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            list(iter_scans(tmp, target_type="Leaf", errors=errors))
        assert len(errors) == 1 and "broken.mat" in errors[0][0]
    print("\n✅ test_iter_scans_filters_before_loading passed!")


if __name__ == "__main__":
    test_mat_v5_leaf()
    test_mat_v73_leaf()
    test_mat_wrong_variable_names()
    test_record_and_phase_guard()
    test_export_roundtrip()
    test_batch_export_real_vegetables()
    test_iter_scans_filters_before_loading()
    print("\n🎉 All tests passed!")
