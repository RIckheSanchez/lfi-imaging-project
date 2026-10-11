"""Smoke + integration tests for the pre-processing pipeline.

Covers the deliverables expected for the Progress Seminar:
* config loads and merges per-target overrides
* loaders produce well-shaped :class:`LFIRawData`
* pipeline runs end-to-end on synthetic, Skin (real), and Leaf (real)
* repeatability: same input on two runs produces byte-identical output
  (proposal §3.2.4 repeatability criterion).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from thz_preprocessing import (
    api,
    clean_fitted_waveform,
    config,
    evaluate_volume_preservation,
    load_leaf_mat,
    load_rdm_fitted_waveform,
    load_rdm_hdf5_frame,
    load_rdm_hdf5_volume,
    load_skin_hdf5,
)
from thz_preprocessing.io import (
    LFIRawData,
    LFIWaveformData,
    _i16_phase_to_radians,
    _signed_14bit_phase_to_radians,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIN_HDF5 = REPO_ROOT / "data" / "skin" / "2025-11-28_10-14-26.594507929.hdf5"
LEAF_MAT = REPO_ROOT / "data" / "leaf" / "leaf1.mat"
VEGETABLE_IMAGE_HDF5 = REPO_ROOT / "data" / "vegetables" / "frame-21.hdf5"
VEGETABLE_WAVEFORM_HDF5 = REPO_ROOT / "data" / "vegetables" / "frame-20.hdf5"
VEGETABLE_VOLUME_FRAMES = [
    REPO_ROOT / "data" / "vegetables" / "frame-21.hdf5",
    REPO_ROOT / "data" / "vegetables" / "frame-22.hdf5",
]


def _write_minimal_rdm_frame(path: Path, frame_id: int, shape=(4, 5)) -> None:
    import h5py

    key = f"frame_{frame_id}_qcl_1_left"
    with h5py.File(path, "w") as f:
        f.create_dataset(
            f"data/IMAGE_AMPLITUDE/{key}",
            data=np.full(shape, frame_id, dtype=np.uint16),
        )
        f.create_dataset(
            f"data/IMAGE_PHASE/{key}",
            data=np.full(shape, frame_id * 10, dtype=np.int16),
        )


def _synthetic_scan(n_z: int = 4, h: int = 32, w: int = 32) -> LFIRawData:
    rng = np.random.default_rng(seed=0)
    amplitude = rng.normal(loc=100.0, scale=10.0, size=(n_z, h, w))
    phase = rng.uniform(low=-np.pi, high=np.pi, size=(n_z, h, w))
    return LFIRawData(
        amplitude=amplitude,
        phase=phase,
        z_indices=list(range(n_z)),
        target_type="Synthetic",
        metadata={"source": "synthetic"},
    )


# ─────────────────────────────────────────────────────────────────────
# config
# ─────────────────────────────────────────────────────────────────────


def test_config_defaults_load():
    cfg = config.load()
    assert "noise_reduction" in cfg
    assert "normalisation" in cfg
    assert "phase_unwrapping" in cfg


def test_config_target_override_skin_vs_leaf():
    skin_cfg = config.load(target_type="Skin")
    leaf_cfg = config.load(target_type="Leaf")
    assert skin_cfg["noise_reduction"]["method"] == "gaussian2d"
    assert leaf_cfg["noise_reduction"]["method"] == "none"
    assert leaf_cfg["normalisation"]["method"] == "none"
    assert leaf_cfg["phase_unwrapping"]["method"] == "skimage_2d"
    assert leaf_cfg["spatial_correction"]["enabled"] is True


# ─────────────────────────────────────────────────────────────────────
# io
# ─────────────────────────────────────────────────────────────────────


def test_confirmed_signed_phase_encodings():
    rdm_codes = np.asarray([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
    rdm_expected = np.asarray(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi * 32767 / 32768]
    )
    np.testing.assert_allclose(_i16_phase_to_radians(rdm_codes), rdm_expected)

    leaf_codes = np.asarray([-8192, -4096, 0, 4096, 8191], dtype=np.float64)
    leaf_expected = np.asarray(
        [-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi * 8191 / 8192]
    )
    rdm_14bit_codes = leaf_codes.astype(np.int16)
    np.testing.assert_allclose(
        _i16_phase_to_radians(rdm_14bit_codes, "signed_14bit_pi"),
        leaf_expected,
    )
    np.testing.assert_allclose(
        _signed_14bit_phase_to_radians(leaf_codes), leaf_expected
    )


@pytest.mark.skipif(not SKIN_HDF5.exists(), reason="Skin pilot HDF5 not present")
def test_load_skin_hdf5_shape_and_phase_range():
    data = load_skin_hdf5(SKIN_HDF5)
    assert data.target_type == "Skin"
    assert data.amplitude.shape == data.phase.shape
    assert data.amplitude.shape[0] == 1, "Skin pilot is single-frame"
    assert data.amplitude.ndim == 3
    assert -np.pi - 1e-9 <= data.phase.min() <= data.phase.max() < np.pi + 1e-9
    assert len(data.z_indices) == 1


@pytest.mark.skipif(not LEAF_MAT.exists(), reason="Leaf .mat not present")
def test_load_leaf_mat_small_slice():
    data = load_leaf_mat(
        LEAF_MAT,
        spatial_slice=(slice(0, 16), slice(0, 16)),
        z_slice=slice(0, 32),
    )
    assert data.target_type == "Leaf"
    assert data.amplitude.shape == (32, 16, 16)
    assert data.amplitude.shape == data.phase.shape
    assert data.amplitude.dtype == np.float64
    assert -np.pi - 1e-9 <= data.phase.min() <= data.phase.max() < np.pi + 1e-9
    assert data.metadata["phase_encoding"] == "leaf_reference_pi4096_offset_pi"
    assert data.metadata["phase_encoding_status"] == "reference_equivalence_validated"
    assert len(data.z_indices) == 32


@pytest.mark.skipif(not VEGETABLE_IMAGE_HDF5.exists(), reason="2026 vegetable HDF5 not present")
def test_load_rdm_hdf5_frame_2026_schema():
    data = load_rdm_hdf5_frame(
        VEGETABLE_IMAGE_HDF5,
        spatial_slice=(slice(0, 16), slice(0, 16)),
    )
    assert data.target_type == "Vegetable"
    assert data.amplitude.shape == (1, 16, 16)
    assert data.amplitude.shape == data.phase.shape
    assert data.z_indices == [21]
    assert data.metadata["source_format"] == "hdf5_rdm_2026_frame"
    assert data.metadata["dataset_key"] == "frame_21_qcl_1_left"
    assert data.metadata["waveform_shape"] == (400, 1300, 184)
    assert data.metadata["phase_encoding_status"] == "explicit_validated"
    assert data.metadata["phase_encoding"] == "signed_14bit_pi"
    assert -np.pi - 1e-9 <= data.phase.min() <= data.phase.max() < np.pi + 1e-9


def test_rdm_loader_rejects_automatic_phase_scale(tmp_path):
    frame = tmp_path / "frame-21.hdf5"
    _write_minimal_rdm_frame(frame, 21)
    with pytest.raises(ValueError, match="Automatic int16 phase-scale inference"):
        load_rdm_hdf5_frame(frame, phase_encoding="auto")


def test_load_rdm_hdf5_volume_sorts_frames_and_maps_z(tmp_path):
    frame_21 = tmp_path / "frame-21.hdf5"
    frame_22 = tmp_path / "frame-22.hdf5"
    details = tmp_path / "experiment_details.txt"
    _write_minimal_rdm_frame(frame_21, 21)
    _write_minimal_rdm_frame(frame_22, 22)
    details.write_text(
        "start z position (mm): 30.05\n"
        "stop z position (mm): 44.67\n"
        "imcrements (mm): 0.1\n",
        encoding="utf-8",
    )

    volume = load_rdm_hdf5_volume(
        [frame_22, frame_21],
        target_type="Vegetable",
        experiment_details_path=details,
        z_reference_frame_id=20,
    )
    assert volume.shape == (2, 4, 5)
    assert volume.z_indices == [21, 22]
    np.testing.assert_allclose(volume.metadata["z_positions_mm"], [30.15, 30.25])
    assert volume.metadata["frame_count"] == 2


def test_load_rdm_hdf5_volume_rejects_incompatible_shapes(tmp_path):
    frame_20 = tmp_path / "frame-20.hdf5"
    frame_21 = tmp_path / "frame-21.hdf5"
    _write_minimal_rdm_frame(frame_20, 20, shape=(2, 5))
    _write_minimal_rdm_frame(frame_21, 21, shape=(4, 5))

    with pytest.raises(ValueError, match="incompatible spatial shapes"):
        load_rdm_hdf5_volume([frame_20, frame_21])


def test_load_rdm_hdf5_rejects_codes_outside_confirmed_signed14(tmp_path):
    import h5py

    frame_20 = tmp_path / "frame-20.hdf5"
    frame_21 = tmp_path / "frame-21.hdf5"
    _write_minimal_rdm_frame(frame_20, 20)
    _write_minimal_rdm_frame(frame_21, 21)
    with h5py.File(frame_20, "a") as f:
        f["data/IMAGE_PHASE/frame_20_qcl_1_left"][0, 0] = 20000

    with pytest.raises(ValueError, match="outside the confirmed signed-14"):
        load_rdm_hdf5_volume([frame_20, frame_21])


@pytest.mark.skipif(not VEGETABLE_WAVEFORM_HDF5.exists(), reason="2026 vegetable HDF5 not present")
def test_load_rdm_fitted_waveform_semantics_and_shape():
    data = load_rdm_fitted_waveform(
        VEGETABLE_WAVEFORM_HDF5,
        spatial_slice=(slice(320, 336), slice(0, 16)),
    )
    assert isinstance(data, LFIWaveformData)
    assert data.shape == (16, 16, 184)
    assert data.n_samples == 184
    assert data.metadata["waveform_kind"] == "fitted_line_of_best_fit"
    flags = data.metadata["acquisition_flags"]
    assert flags["en_lobf"] == 1
    assert flags["log_lobf"] == 1
    assert flags["log_rx_raw"] == 0
    assert flags["log_rx_avg"] == 0
    assert flags["log_rx_dec"] == 0
    assert flags["log_rx_sub"] == 0


@pytest.mark.skipif(not VEGETABLE_WAVEFORM_HDF5.exists(), reason="2026 vegetable HDF5 not present")
def test_clean_real_fitted_waveform_preserves_trace_shape():
    raw = load_rdm_fitted_waveform(
        VEGETABLE_WAVEFORM_HDF5,
        spatial_slice=(slice(320, 336), slice(0, 16)),
    )
    out = clean_fitted_waveform(raw)
    assert out.shape == raw.shape
    metrics = out.metadata["preservation_metrics"]
    assert metrics["valid_trace_count"] > 0
    assert metrics["median_correlation"] > 0.90
    assert metrics["roughness_ratio"] < 1.0


# ─────────────────────────────────────────────────────────────────────
# pipeline — synthetic
# ─────────────────────────────────────────────────────────────────────


def test_pipeline_synthetic_returns_same_shape_and_records_provenance():
    data = _synthetic_scan(n_z=4, h=32, w=32)
    out = api.run_pipeline(data)
    assert isinstance(out, LFIRawData)
    assert out.amplitude.shape == data.amplitude.shape
    assert out.phase.shape == data.phase.shape
    assert out.z_indices == data.z_indices
    assert out.target_type == data.target_type
    assert "pipeline" in out.metadata
    stages = [list(d.keys())[0] for d in out.metadata["pipeline"]]
    assert stages == ["noise_reduction", "normalisation", "phase_unwrapping", "spatial_correction"]
    assert "timings_seconds" in out.metadata
    assert out.metadata["timings_seconds"]["total"] > 0


def test_pipeline_repeatability_byte_identical():
    """Same input + same config → byte-identical output. Proposal §3.2.4."""
    data = _synthetic_scan(n_z=4, h=32, w=32)
    a = api.run_pipeline(data)
    b = api.run_pipeline(data)
    np.testing.assert_array_equal(a.amplitude, b.amplitude)
    np.testing.assert_array_equal(a.phase, b.phase)


def test_pipeline_normalisation_zeros_mean_baseline():
    """Baseline subtraction should drop the floor to near-zero per slice."""
    data = _synthetic_scan(n_z=4, h=32, w=32)
    out = api.run_pipeline(
        data,
        config_overrides={
            "noise_reduction": {"method": "none"},
            "normalisation": {"method": "baseline", "baseline_percentile": 5.0},
            "phase_unwrapping": {"enabled": False},
        },
    )
    per_slice_min = out.amplitude.reshape(out.amplitude.shape[0], -1).min(axis=1)
    assert np.all(per_slice_min < 0.0)  # 5th percentile floor is now negative


def test_volume_preservation_metrics_respect_scale_and_phase_cycles():
    raw = _synthetic_scan(n_z=3, h=16, w=16)
    processed = LFIRawData(
        amplitude=raw.amplitude * 2.0 + 5.0,
        phase=raw.phase + 2.0 * np.pi,
        z_indices=list(raw.z_indices),
        target_type=raw.target_type,
        metadata={"timings_seconds": {"total": 0.1}},
    )
    metrics = evaluate_volume_preservation(raw, processed)
    assert metrics["summary"]["median_amplitude_pearson"] == pytest.approx(1.0)
    assert metrics["summary"]["max_amplitude_normalized_rmse"] < 1e-12
    assert metrics["summary"]["min_phase_modulo_consistency"] == pytest.approx(1.0)
    assert metrics["summary"]["max_phase_rewrap_rmse_radians"] < 1e-12


def test_spatial_unwrap_quality_gate_accepts_smooth_wrapped_ramp():
    y, x = np.mgrid[:32, :48]
    true_phase = 0.15 * x + 0.08 * y
    wrapped = np.angle(np.exp(1j * true_phase))[None, :, :]
    raw = LFIRawData(
        amplitude=np.ones_like(wrapped),
        phase=wrapped,
        z_indices=[0],
        target_type="Vegetable",
    )
    out = api.run_pipeline(
        raw,
        config_overrides={
            "noise_reduction": {"method": "none"},
            "normalisation": {"method": "none"},
        },
    )
    offset_cycles = round(np.median(out.phase[0] - true_phase) / (2.0 * np.pi))
    np.testing.assert_allclose(
        out.phase[0] - offset_cycles * 2.0 * np.pi,
        true_phase,
        atol=1e-10,
    )
    assert out.metadata["phase_unwrapping_status"]["status"] == "unwrapped"


def test_spatial_unwrap_quality_gate_keeps_high_residue_phase_wrapped():
    rng = np.random.default_rng(12)
    wrapped = rng.uniform(-np.pi, np.pi, size=(1, 32, 48))
    raw = LFIRawData(
        amplitude=np.ones_like(wrapped),
        phase=wrapped,
        z_indices=[0],
        target_type="Vegetable",
    )
    out = api.run_pipeline(
        raw,
        config_overrides={
            "noise_reduction": {"method": "none"},
            "normalisation": {"method": "none"},
        },
    )
    np.testing.assert_array_equal(out.phase, raw.phase)
    assert (
        out.metadata["phase_unwrapping_status"]["status"]
        == "kept_wrapped_low_quality"
    )


# ─────────────────────────────────────────────────────────────────────
# pipeline — real datasets
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.skipif(not SKIN_HDF5.exists(), reason="Skin pilot HDF5 not present")
def test_pipeline_on_real_skin():
    data = load_skin_hdf5(SKIN_HDF5)
    out = api.run_pipeline(data)
    assert out.amplitude.shape == data.amplitude.shape
    assert out.phase.shape == data.phase.shape
    assert out.target_type == "Skin"
    used = out.metadata["config_resolved"]["noise_reduction"]["method"]
    assert used == "gaussian2d", "Skin target should pick the 2D-image operator"
    # Signed-14 decoding leaves the signal region almost residue free, so the
    # strict gate releases a masked unwrap; outside the sample it is NaN.
    status = out.metadata["phase_unwrapping_status"]
    assert status["status"] == "unwrapped"
    diagnostics = status["quality_diagnostics"][0]
    assert diagnostics["residue_fraction"] <= 0.001 < diagnostics["residue_fraction_whole_image"]
    products = out.metadata["spatial_products"]
    mask = products["signal_mask"][0]
    assert 0.3 < mask.mean() < 0.6
    assert np.isfinite(out.phase[0][mask]).all() and np.isnan(out.phase[0][~mask]).all()
    np.testing.assert_allclose(
        np.exp(1j * out.phase[0][mask]), np.exp(1j * data.phase[0][mask]), atol=1e-9
    )
    assert products["phase_correction_available"].all()
    assert np.isfinite(products["phase_corrected"][0][mask]).all()
    assert np.isfinite(products["amplitude_corrected"]).all()


@pytest.mark.skipif(not SKIN_HDF5.exists(), reason="Skin pilot HDF5 not present")
def test_skin_phase_wraps_are_two_pi_steps_only_under_signed14_scale():
    """The evidence behind the default Skin decoding (numerical, not a sign-off)."""
    new = load_skin_hdf5(SKIN_HDF5).phase[0]
    old = load_skin_hdf5(SKIN_HDF5, phase_encoding="0_2pi")
    assert old.metadata["phase_encoding_status"] == "unconfirmed"
    assert new.min() < -3.1 and new.max() > 3.1
    assert np.abs(old.phase).max() < np.pi / 4 + 1e-9

    def large_jumps(phase):
        steps = np.abs(np.diff(phase, axis=1))
        return steps[steps > 1.0]

    jumps_new, jumps_old = large_jumps(new), large_jumps(old.phase[0])
    # Under the signed-14 scale most large steps are genuine 2*pi wraps ...
    assert np.mean(jumps_new > 5.5) > 0.5
    # ... under the old scale they are pi/2 steps that no wrap can produce.
    assert np.mean((jumps_old > 1.3) & (jumps_old < 1.8)) > 0.8
    assert not np.any(jumps_old > 3.0)


@pytest.mark.skipif(not LEAF_MAT.exists(), reason="Leaf .mat not present")
def test_pipeline_on_real_leaf_small_slice():
    data = load_leaf_mat(
        LEAF_MAT,
        spatial_slice=(slice(0, 32), slice(0, 32)),
        z_slice=slice(0, 64),
    )
    out = api.run_pipeline(data)
    assert out.amplitude.shape == data.amplitude.shape
    used = out.metadata["config_resolved"]["noise_reduction"]["method"]
    assert used == "none", "Leaf must not assume a waveform or time coordinate"
    assert out.metadata["spatial_correction_status"]["enabled"]
    # Spatial unwrapping either applies integer cycles or retains wrapped data.
    assert (
        out.phase.max() - out.phase.min() >= data.phase.max() - data.phase.min() - 1e-9
    )


@pytest.mark.skipif(not VEGETABLE_IMAGE_HDF5.exists(), reason="2026 vegetable HDF5 not present")
def test_pipeline_on_real_vegetable_small_slice():
    data = load_rdm_hdf5_frame(
        VEGETABLE_IMAGE_HDF5,
        spatial_slice=(slice(0, 32), slice(0, 32)),
    )
    out = api.run_pipeline(data)
    assert out.amplitude.shape == data.amplitude.shape
    assert out.phase.shape == data.phase.shape
    assert out.target_type == "Vegetable"
    used = out.metadata["config_resolved"]["noise_reduction"]["method"]
    assert used == "gaussian2d", "Vegetable frame should pick the 2D-image operator"
    assert out.metadata["phase_unwrapping_status"]["status"] in {
        "unwrapped",
        "kept_wrapped_low_quality",
    }
    assert out.metadata["phase_unwrapping_status"]["quality_diagnostics"]


@pytest.mark.skipif(
    not all(path.exists() for path in VEGETABLE_VOLUME_FRAMES),
    reason="2026 vegetable volume frames not present",
)
def test_pipeline_on_real_vegetable_volume_small_slice():
    raw = load_rdm_hdf5_volume(
        VEGETABLE_VOLUME_FRAMES,
        target_type="Vegetable",
        spatial_slice=(slice(0, 32), slice(0, 32)),
    )
    out = api.run_pipeline(raw)
    metrics = evaluate_volume_preservation(raw, out)
    assert raw.shape == (2, 32, 32)
    assert raw.z_indices == [21, 22]
    assert raw.metadata["phase_encoding"] == "signed_14bit_pi"
    assert out.metadata["phase_unwrapping_status"]["status"] in {
        "unwrapped",
        "kept_wrapped_low_quality",
    }
    assert metrics["summary"]["min_phase_modulo_consistency"] > 0.999999
    assert metrics["summary"]["max_phase_rewrap_rmse_radians"] < 1e-12
    assert metrics["summary"]["median_amplitude_pearson"] > 0.8


@pytest.mark.skipif(
    not all(path.exists() for path in VEGETABLE_VOLUME_FRAMES),
    reason="2026 vegetable volume frames not present",
)
def test_full_vegetable_frames_respect_residue_gate():
    raw = load_rdm_hdf5_volume(
        VEGETABLE_VOLUME_FRAMES,
        target_type="Vegetable",
    )
    # Even inside the signal mask frames 21/22 exceed the strict gate, so
    # the default keeps the measured wrapped phase untouched.
    strict = api.run_pipeline(raw)
    np.testing.assert_array_equal(strict.phase, raw.phase)
    status = strict.metadata["phase_unwrapping_status"]
    assert status["status"] == "kept_wrapped_low_quality"
    for item in status["quality_diagnostics"]:
        assert item["mask"] == "amplitude_otsu"
        assert item["max_residue_fraction"] == 0.001
        assert item["residue_fraction"] > item["max_residue_fraction"]
        # Residues concentrate where there is no signal.
        assert item["residue_fraction"] < item["residue_fraction_whole_image"]
    assert not strict.metadata["spatial_products"]["phase_correction_available"].any()

    # A loose gate would release a masked unwrap that only differs by cycles.
    out = api.run_pipeline(
        raw, config_overrides={"phase_unwrapping": {"max_residue_fraction": 0.06}}
    )
    assert out.metadata["phase_unwrapping_status"]["status"] == "unwrapped"
    inside = np.isfinite(out.phase)
    assert 0.02 < inside.mean() < 0.9
    np.testing.assert_allclose(
        np.angle(np.exp(1j * (out.phase[inside] - raw.phase[inside]))), 0, atol=1e-9
    )
