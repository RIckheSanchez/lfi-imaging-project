import h5py
import numpy as np
import pytest
from part3_reconstruction.audit import audit_file, audit_directory
from part3_reconstruction import DataLoader


def make_file(path):
    with h5py.File(path, "w") as f:
        for key in ("0010", "0002"):
            f[f"amplitude/{key}"] = np.arange(12).reshape(3, 4).astype("uint16")
            f[f"phase/{key}"] = np.full((3, 4), 180.)


def test_pairing_and_numeric_order(tmp_path):
    p = tmp_path / "x.h5"; make_file(p)
    r = audit_file(p)
    assert [x["key"] for x in r["pairs"]] == ["0002", "0010"]
    assert r["frame_id_gaps"] == [[2, 10]]
    d = DataLoader.load_grouped_frame(p, "0002", axis_order="xy", phase_unit="degrees")
    assert d.phase.shape == (4, 3)
    np.testing.assert_allclose(d.phase, np.pi)
    np.testing.assert_equal(d.amplitude, np.arange(12).reshape(3, 4).T)


def test_mismatched_frames_fail(tmp_path):
    p = tmp_path / "x.h5"; make_file(p)
    with h5py.File(p, "a") as f: del f["phase/0010"]
    assert audit_file(p)["status"] == "error"
    with pytest.raises(ValueError): DataLoader.load_grouped_frame(p, "0002", axis_order="yx", phase_unit="rad")


def test_corrupt_file_does_not_stop_batch(tmp_path):
    make_file(tmp_path / "good.h5")
    (tmp_path / "bad.h5").write_bytes(b"not hdf5")
    r = audit_directory(tmp_path, tmp_path / "report.json")
    assert r["summary"] == {"files": 2, "pairs": 2, "errors": 1, "review": 1}


def test_uq_ignores_bitmap_and_waveform(tmp_path):
    p = tmp_path / "uq.h5"
    with h5py.File(p, "w") as f:
        f["data/IMAGE_AMPLITUDE/frame_1_qcl_1_left"] = np.array([[65535, 0]], dtype="uint16")
        f["data/IMAGE_PHASE/frame_1_qcl_1_left"] = np.array([[1., np.nan]])
        f["data/IMAGE_AMPLITUDE/frame_1_qcl_1_left_BMP"] = np.zeros((1, 2))
        f.create_dataset("data/RX_LINE_OF_BEST_FIT/huge", shape=(100, 100, 10000), dtype="int16", chunks=True)
    r = audit_file(p)
    assert len(r["pairs"]) == 1 and r["status"] == "review"
    assert r["pairs"][0]["amplitude"]["dtype_max_fraction"] == .5


def test_empty_audit_fails(tmp_path):
    with pytest.raises(ValueError): audit_directory(tmp_path, tmp_path / "out.json")
