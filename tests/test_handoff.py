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


def test_mirror_geometry_reproduces_stored_ratio(tmp_path):
    import h5py
    from part3_reconstruction.handoff import mirror_geometry
    path = tmp_path / "frame-21.hdf5"
    values = {"imageColumns": 1300, "imageRows": 800, "columnStep": 26, "columnStepFract256": -26,
              "rowStepSize": 43, "rowStepFract256": 0}
    with h5py.File(path, "w") as f:
        for k, v in values.items():
            f["thz/info/mcp/" + k] = np.int64(v)
        f["thz/info/aspect_ratio_computed"] = np.float32(1300 * 26 / (400 * 43))
    g = mirror_geometry(path)
    assert g["matches_recorded"] and g["aspect_from_integer_steps"] == pytest.approx(1.9651163)
    assert g["aspect_with_fractional_steps"] == pytest.approx(1300 * (26 - 26 / 256) / (400 * 43))
    assert g["pixel_aspect_yx"] == pytest.approx(43 / 26)
    with h5py.File(path, "a") as f:
        f["thz/info/aspect_ratio_computed"][()] = np.float32(1.5)
    assert not mirror_geometry(path)["matches_recorded"]


def test_aperture_aspect_recovers_frame_ratio():
    from part3_reconstruction.handoff import aperture_aspect
    ny, nx = 200, 1000                               # frame ratio 2.5 -> pixel dy/dx = 1000 / (200 * 2.5) = 2
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    for cy in (100.0, 95.0):                         # radius 300 px = 150 rows: cut off at top and bottom
        disk = ((xx - 480) / 300) ** 2 + ((yy * 2 - 2 * cy) / 300) ** 2 <= 1
        result = aperture_aspect(np.where(disk, 1.0, 0.1))
        assert len(result["variants"]) == 6
        for v in result["variants"]:
            assert v["frame_aspect"] == pytest.approx(2.5, rel=0.03)
            assert v["semi_axes_px_xy"][0] == pytest.approx(300, rel=0.02)
    with pytest.raises(ValueError): aperture_aspect(np.full((50, 50), np.nan))
    with pytest.raises(ValueError): aperture_aspect(np.ones((50, 50)))


def test_html_report_is_self_contained(tmp_path):
    from matplotlib.figure import Figure
    from part3_reconstruction.report_html import write_html_report
    fig = Figure(figsize=(2, 1)); fig.subplots().plot([0, 1])
    fig.savefig(tmp_path / "handoff_veg_depth_profile.png")
    sample = {"frames": 2, "shape": [4, 13], "phase_available": 1, "signal_fraction_range": [.1, .2],
              "max_identity_error_pre_spatial": 1e-13, "max_identity_error_detail": 2e-13,
              "phase_values_outside_mask": 0, "stripe_index_raw_median": .01,
              "stripe_index_corrected_median": .007, "stripe_reduced_frames": 2}
    report = {"handoff": "x<y", "files_read": 2, "files_with_missing_fields": [], "elapsed_seconds": 1.0,
              "scope": "test", "summary_by_sample": {"vegetable": sample}, "render_seconds": {},
              "depth_resampling": {"input_shape_zyx": [2, 4, 13], "z_mm_input": [1.0, 2.0], "output_shape_zyx": [11, 4, 13],
                                   "output_spacing_mm": .1, "output_megabytes": .01, "resample_seconds": .01,
                                   "strongest_frame": 2, "note": "n", "profile": [
                                       {"frame_id": 1, "z_mm": 1.0, "raw_mean": 2.0, "signal_fraction": .1,
                                        "stripe_index_corrected": .01}]},
              "geometry": {}, "frames": [{"file": "a.npz", "missing": ["raw_phase"]}]}
    page = write_html_report(report, tmp_path).read_text(encoding="utf-8")
    assert page.count("data:image/png;base64,") == 1          # only the figure that exists
    assert "Figure 1 missing" in page and "x&lt;y" in page       # missing figures flagged, text escaped
    assert "mirror settings not checked" in page
    assert "<th>Stripe index raw &rarr; corrected (median)</th>" in page
