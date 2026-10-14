"""Integration tests for the adapter to the shared data-import loader."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from thz_preprocessing import api, load_rdm_hdf5_frame, load_skin_hdf5
from thz_preprocessing import shared_loader

ROOT = Path(__file__).resolve().parents[1]
SKIN_HDF5 = ROOT / "data/skin/2025-11-28_10-14-26.594507929.hdf5"
VEGETABLE_FRAME = ROOT / "data/vegetables/frame-21.hdf5"

pytestmark = pytest.mark.skipif(
    not (shared_loader.VENDOR_DIR / "loader.py").exists(),
    reason="vendored shared loader not present",
)


def test_zstack_phase_unit_is_unconfirmed_upstream_and_withheld(tmp_path):
    path = tmp_path / "Chip" / "board.hdf5"
    path.parent.mkdir()
    rng = np.random.default_rng(0)
    amp = rng.random((3, 6, 8))
    phase = rng.uniform(-np.pi, np.pi, (3, 6, 8))
    with h5py.File(path, "w") as f:
        for i, z in enumerate([530, 531, 532]):
            f["amplitude/%d" % z] = amp[i]
            f["phase/%d" % z] = phase[i]
    data = shared_loader.load_scan(path)
    assert data.metadata["loader"] == "shared_data_import"
    assert data.target_type == "Chip"
    assert data.z_indices == [530, 531, 532]
    # Upstream v3 treats Chip/Marble z-stack phase as unit-unconfirmed (looks like degrees).
    assert data.metadata["phase_encoding_status"] == "unconfirmed"
    assert data.metadata["phase_units"] == "raw_codes"
    np.testing.assert_array_equal(data.amplitude, amp)
    np.testing.assert_array_equal(data.phase, phase)


def test_shared_loader_does_not_pollute_sys_path(tmp_path):
    import sys

    shared_loader._upstream()
    assert str(shared_loader.VENDOR_DIR) not in sys.path


@pytest.mark.skipif(not SKIN_HDF5.exists(), reason="Skin pilot HDF5 not present")
def test_skin_via_shared_loader_matches_local_amplitude_and_withholds_phase():
    shared = shared_loader.load_scan(SKIN_HDF5, target_type="Skin", qcl="QCL0", side="Left")
    local = load_skin_hdf5(SKIN_HDF5)
    assert shared.metadata["loader"] == "shared_data_import"
    assert shared.target_type == "Skin"
    assert shared.z_indices == local.z_indices
    np.testing.assert_array_equal(shared.amplitude, local.amplitude)
    # Upstream keeps unconfirmed Skin phase as raw uint16 codes.
    assert shared.metadata["phase_units"] == "raw_codes"
    assert shared.metadata["phase_encoding_status"] == "unconfirmed"
    assert shared.phase.max() > 2 * np.pi

    out = api.run_pipeline(shared)
    assert out.metadata["phase_unwrapping_status"]["status"] == "withheld_unconfirmed_encoding"
    np.testing.assert_array_equal(out.phase, shared.phase)
    products = out.metadata["spatial_products"]
    assert not products["phase_correction_available"].any()
    reference = api.run_pipeline(local).metadata["spatial_products"]
    np.testing.assert_array_equal(products["amplitude_corrected"], reference["amplitude_corrected"])


@pytest.mark.skipif(not VEGETABLE_FRAME.exists(), reason="2026 vegetable frame not present")
def test_2026_layout_via_shared_loader_matches_local_loader():
    data = shared_loader.load_scan(VEGETABLE_FRAME, target_type="Vegetables")
    assert data.metadata["loader"] == "shared_data_import"
    assert data.target_type == "Vegetable"
    assert data.metadata["phase_encoding_status"] == "explicit_validated"
    local = load_rdm_hdf5_frame(VEGETABLE_FRAME)
    assert data.z_indices == local.z_indices
    np.testing.assert_array_equal(data.amplitude, local.amplitude)
    np.testing.assert_allclose(data.phase, local.phase, atol=1e-12)
    out = api.run_pipeline(data).metadata["spatial_products"]
    ref = api.run_pipeline(local).metadata["spatial_products"]
    np.testing.assert_array_equal(out["amplitude_corrected"], ref["amplitude_corrected"])


def test_unsupported_hdf5_layout_falls_back_and_records_gap(tmp_path, monkeypatch):
    path = tmp_path / "odd.hdf5"
    with h5py.File(path, "w") as f:
        f["something/else"] = np.zeros((2, 2))
    sentinel = object()
    monkeypatch.setattr(shared_loader.io, "load_rdm_hdf5_frame",
                        lambda p, **k: type("D", (), {"metadata": {}, "tag": sentinel})())
    data = shared_loader.load_scan(path)
    assert data.metadata["loader"] == "local_fallback"
    assert "not supported by the shared loader" in data.metadata["shared_loader_gap"]


@pytest.mark.skipif(not VEGETABLE_FRAME.exists(), reason="2026 vegetable frames not present")
def test_load_volume_orders_frames_numerically_skips_frame_20_and_adds_z(tmp_path):
    folder = tmp_path / "2026_Vegetables"
    folder.mkdir()
    source = VEGETABLE_FRAME.parent
    wanted = [n for n in (20, 21, 22, 100) if (source / ("frame-%d.hdf5" % n)).exists()]
    if not {20, 21, 22, 100} <= set(wanted):
        pytest.skip("needs local frames 20, 21, 22 and 100")
    for n in wanted:
        (folder / ("frame-%d.hdf5" % n)).symlink_to(source / ("frame-%d.hdf5" % n))
    (folder / "experiment_details.txt").symlink_to(source / "experiment_details.txt")

    volume = shared_loader.load_volume(folder)
    assert volume.z_indices == [21, 22, 100]          # numeric, not "100" < "21"
    assert volume.shape == (3, 400, 1300)
    assert len(volume.metadata["skipped_frames"]) == 1
    assert "frame-20" in volume.metadata["skipped_frames"][0]["path"]
    assert volume.metadata["z_positions_mm"] == [30.15, 30.25, 38.05]
    assert volume.target_type == "Vegetable"

    from thz_preprocessing import load_rdm_hdf5_volume
    local = load_rdm_hdf5_volume([source / ("frame-%d.hdf5" % n) for n in (100, 22, 21)])
    np.testing.assert_array_equal(volume.amplitude, local.amplitude)
    np.testing.assert_allclose(volume.phase, local.phase, atol=1e-12)
