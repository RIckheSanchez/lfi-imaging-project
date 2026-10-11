import json
import numpy as np
import pytest
from part3_reconstruction import ImageGenerator
from part3_reconstruction.handoff import load_frame, load_volume, resample_depth, pixel_aspect


def _frame(path, shape, *, available, frame_id=40):
    rng = np.random.default_rng(0)
    amp = rng.random((1, *shape))
    mask = np.zeros((1, *shape), bool)
    mask[:, : shape[0] // 2] = True
    corrected = np.where(mask, 1.5, np.nan) if available else np.full((1, *shape), np.nan)
    np.savez(path, amplitude_corrected=amp, amplitude_detail=amp - .5, raw_amplitude=amp * 10,
             raw_phase=np.full((1, *shape), -3.0), phase_corrected=corrected, signal_mask=mask,
             phase_correction_available=np.array([available]), z_indices=np.array([frame_id]))
    return amp[0]


def test_skin_frame_uses_corrected_phase_and_sidecar_aspect(tmp_path):
    path = tmp_path / "skin_frame40_spatial.npz"
    amp = _frame(path, (4, 13), available=True)
    (tmp_path / "skin_frame40_spatial_metadata.json").write_text(
        json.dumps({"metadata": {"display_aspect_ratio": 2.0}}), encoding="utf-8")
    d = load_frame(path)
    np.testing.assert_allclose(d.amplitude, amp, rtol=1e-6)
    assert d.metadata["phase_state"] == "unwrapped" and d.metadata["phase_channel"] == "phase_corrected"
    assert np.isnan(d.phase[2:]).all() and np.all(d.phase[:2] == 1.5)
    assert d.metadata["pixel_aspect_yx"] == pytest.approx(13 / (4 * 2.0))
    assert d.metadata["signal_fraction"] == pytest.approx(.5)


def test_vegetable_frame_falls_back_to_wrapped_raw_phase(tmp_path):
    path = tmp_path / "vegetable_frame130_spatial.npz"
    _frame(path, (4, 13), available=False, frame_id=130)
    d = load_frame(path, amplitude_channel="raw_amplitude")
    assert d.metadata["phase_state"] == "wrapped" and np.all(d.phase == -3.0)
    assert d.metadata["frame_id"] == 130 and d.metadata["amplitude_channel"] == "raw_amplitude"
    ImageGenerator(d).generate_phase("cyclic")  # wrapped radians are valid for cyclic display


def test_leaf_is_transposed_to_long_axis_x(tmp_path):
    path = tmp_path / "leaf_frame045_spatial.npz"
    amp = _frame(path, (12, 4), available=True)
    d = load_frame(path)
    assert d.amplitude.shape == (4, 12)
    np.testing.assert_allclose(d.amplitude, amp.T, rtol=1e-6)
    assert d.metadata["pixel_aspect_yx"] == pytest.approx(12 / (4 * 1.5))


def test_frame_errors(tmp_path):
    with pytest.raises(ValueError): load_frame(tmp_path / "unknown.npz")
    path = tmp_path / "skin_frame40_spatial.npz"
    np.savez(path, amplitude_corrected=np.ones((1, 2, 2)))
    with pytest.raises(ValueError, match="lacks fields"): load_frame(path)
    with pytest.raises(ValueError): load_frame(path, amplitude_channel="phase")


def test_volume_and_depth_resampling(tmp_path):
    z = np.array([30.15, 30.25, 30.35, 31.35, 32.35])
    amp = (np.arange(5, dtype=float)[:, None, None] * np.ones((5, 3, 4)))
    path = tmp_path / "vegetables_volume.npz"
    np.savez(path, amplitude_corrected=amp.astype(np.float32), raw_amplitude=amp, raw_phase=np.zeros_like(amp),
             z_mm=z, frame_ids=np.array([21, 22, 23, 33, 43]), signal_mask=amp > 1)
    data, z_mm, ids, mask = load_volume(path)
    assert data.metadata["phase_state"] == "wrapped" and ids.tolist() == [21, 22, 23, 33, 43]
    out, grid = resample_depth(data.amplitude, z_mm, 0.1)
    assert len(grid) == 23 and grid[-1] == pytest.approx(32.35)
    # Piecewise-linear reference along z for every pixel.
    np.testing.assert_allclose(out[:, 0, 0], np.interp(grid, z, np.arange(5)), atol=1e-5)
    # Reverse-ordered input gives the same result.
    out_rev, _ = resample_depth(data.amplitude[::-1], z_mm[::-1], 0.1)
    np.testing.assert_allclose(out_rev, out)


def test_depth_resampling_keeps_missing_and_rejects_bad_input():
    v = np.ones((3, 2, 2), np.float32)
    v[1, 0, 0] = np.nan
    out, grid = resample_depth(v, [0, 1, 2], .5)
    assert np.isnan(out[1:4, 0, 0]).all() and np.isfinite(out[:, 1, 1]).all()
    with pytest.raises(ValueError): resample_depth(v, [0, 0, 1], .5)
    with pytest.raises(ValueError): resample_depth(v, [0, 1, 2], 0)
    with pytest.raises(ValueError): resample_depth(v[0], [0], .5)


def test_pixel_aspect_drives_figure_shape():
    from part3_reconstruction import ImagingData
    d = ImagingData(np.ones((4, 13)), np.zeros((4, 13)), {"pixel_aspect_yx": pixel_aspect((4, 13), 1.9651163)})
    fig = ImageGenerator(d).generate_amplitude()
    assert fig.axes[0].get_aspect() == pytest.approx(13 / (4 * 1.9651163))
