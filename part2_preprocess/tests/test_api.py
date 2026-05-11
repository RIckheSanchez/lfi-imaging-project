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

from thz_preprocessing import api, config, load_leaf_mat, load_skin_hdf5
from thz_preprocessing.io import LFIRawData

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIN_HDF5 = REPO_ROOT / "data" / "skin" / "2025-11-28_10-14-26.594507929.hdf5"
LEAF_MAT = REPO_ROOT / "data" / "leaf" / "leaf1.mat"


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
    assert skin_cfg["noise_reduction"]["method"] == "median2d"
    assert leaf_cfg["noise_reduction"]["method"] == "savgol_1d"
    assert leaf_cfg["normalisation"]["method"] == "zscore"


# ─────────────────────────────────────────────────────────────────────
# io
# ─────────────────────────────────────────────────────────────────────


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
    assert len(data.z_indices) == 32


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
    assert stages == ["noise_reduction", "normalisation", "phase_unwrapping"]
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
    assert used == "median2d", "Skin target should pick the 2D-image operator"


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
    assert used == "savgol_1d", "Leaf target should pick the sweep-axis operator"
    # Itoh-1D unwrap is enabled for leaf; output phase can exceed [-pi, pi).
    assert (
        out.phase.max() - out.phase.min() >= data.phase.max() - data.phase.min() - 1e-9
    )
