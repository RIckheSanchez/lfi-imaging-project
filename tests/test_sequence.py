import json
import h5py
import nibabel as nib
import numpy as np
import pytest
from part3_reconstruction.sequence import load_sequence, reconstruct_manifest


def fixture_manifest(tmp_path):
    with h5py.File(tmp_path / "scan.h5", "w") as f:
        for i, z in enumerate([0., 2., 5.]):
            yy, xx = np.mgrid[:8, :9]
            f[f"amplitude/{i:04d}"] = xx+yy+z
            f[f"phase/{i:04d}"] = (2*xx+yy+z)*180/np.pi
    m = {"schema_version": 1, "sample_id": "synthetic-linear", "calibration_source": "analytic synthetic coordinates",
         "spacing_xy": [1, 1], "origin_xy": [10, 20], "spatial_unit": "mm",
         "axis_order": "yx", "phase_unit": "degrees", "phase_state": "unwrapped",
         "frames": [{"path": "scan.h5", "amplitude_key": f"amplitude/{i:04d}",
                     "phase_key": f"phase/{i:04d}", "z": z} for i, z in [(2, 5), (0, 0), (1, 2)]]}
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m))
    return p, m


def test_sorted_sequence_and_provenance(tmp_path):
    p, _ = fixture_manifest(tmp_path)
    v = load_sequence(p)
    np.testing.assert_equal(v.z, [0, 2, 5])
    assert v.data.amplitude.shape == (3, 8, 9)
    assert v.data.phase[2, 3, 4] == pytest.approx(16, abs=2e-6)
    assert v.data.metadata["sorted_sources"][0]["amplitude_key"] == "amplitude/0000"


def test_complete_pipeline_readback(tmp_path):
    p, _ = fixture_manifest(tmp_path)
    out = tmp_path / "output"
    r = reconstruct_manifest(p, out, spacing=1)
    assert r["readback_verified"] and r["shape_zyx"] == [6, 8, 9]
    image = nib.load(out / "phase.nii.gz")
    np.testing.assert_allclose(image.affine @ [4, 3, 5, 1], [14, 23, 5, 1])
    assert image.get_fdata()[4, 3, 5] == pytest.approx(16, abs=2e-6)
    assert (out / "middle_xy.png").is_file()
    with pytest.raises(ValueError, match="empty"): reconstruct_manifest(p, out, spacing=1)


@pytest.mark.parametrize("case", ["duplicate_source", "duplicate_z", "wrong_key", "missing_file", "size", "uncalibrated", "units", "unknown_phase"])
def test_reject_invalid_manifest(tmp_path, case):
    p, m = fixture_manifest(tmp_path)
    if case == "duplicate_source": m["frames"][1] = m["frames"][0]
    elif case == "duplicate_z": m["frames"][1]["z"] = 5
    elif case == "wrong_key": m["frames"][0]["phase_key"] = "phase/0001"
    elif case == "missing_file": m["frames"][0]["path"] = "missing.h5"
    elif case == "size":
        with h5py.File(tmp_path / "scan.h5", "a") as f:
            del f["phase/0002"]; f["phase/0002"] = np.zeros((7, 9))
    elif case == "uncalibrated": m["phase_unit"] = "raw_counts"
    elif case == "units": m["spatial_unit"] = "pixels"
    elif case == "unknown_phase": m["phase_state"] = "unknown"
    p.write_text(json.dumps(m))
    with pytest.raises((ValueError, OSError)):
        reconstruct_manifest(p, tmp_path / "output", spacing=1)
    assert not (tmp_path / "output").exists()


def test_memory_and_irregular_export_guard(tmp_path):
    p, _ = fixture_manifest(tmp_path)
    with pytest.raises(ValueError, match="max_voxels"): load_sequence(p, max_voxels=2)
    with pytest.raises(ValueError, match="irregular"): reconstruct_manifest(p, tmp_path / "out")


def test_explicit_integer_calibration(tmp_path):
    p, m = fixture_manifest(tmp_path)
    m.update(phase_unit="raw_counts", phase_scale_to_rad=.01, phase_offset_rad=-1.)
    p.write_text(json.dumps(m))
    v = load_sequence(p)
    assert v.data.phase[0, 0, 0] == -1
    assert v.data.phase[2, 3, 4] == pytest.approx(16*180/np.pi*.01-1, abs=1e-6)
