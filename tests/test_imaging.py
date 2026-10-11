import json
import numpy as np
import pytest
import h5py
from PIL import Image
from part3_reconstruction import ImagingData, DataLoader, ImageGenerator
from part3_reconstruction.colormap_engine import normalise, fold_phase
from part3_reconstruction.validation import image_metrics, dice_score


@pytest.mark.parametrize("strategy", ["linear", "percentile", "log"])
def test_normalise_negative_missing_and_constant(strategy):
    source = np.array([[-2., 0., 4., np.nan, np.inf]])
    backup = source.copy()
    out = normalise(source, strategy)
    assert out.dtype == np.float32
    assert out[0, 0] == 0 and out[0, 2] == 1
    assert np.isnan(out[0, 3:]).all()
    np.testing.assert_equal(source, backup)
    np.testing.assert_equal(normalise(np.ones((2, 3)), strategy), 0)


def test_linear_and_log_known_reference():
    np.testing.assert_allclose(normalise([0, 1, 2], "linear"), [0, .5, 1])
    np.testing.assert_allclose(normalise([0, 1, 3], "log"), [0, .5, 1])
    np.testing.assert_allclose(normalise([1, 2], "linear", limits=(0, 4)), [.25, .5])


@pytest.mark.parametrize("kwargs", [{"strategy": "bad"}, {"pct_lo": 99, "pct_hi": 2}, {"limits": (2, 1)}])
def test_bad_normalisation(kwargs):
    with pytest.raises(ValueError): normalise([1, 2], **kwargs)


def test_phase_periodicity():
    a = np.array([-8244., -np.pi, 0., np.pi, 8244.])
    np.testing.assert_allclose(fold_phase(a + 2*np.pi), fold_phase(a), atol=1e-11)
    assert fold_phase(np.pi) == -np.pi


def test_axis_mapping_and_data_validation():
    a = np.arange(24).reshape(2, 3, 4)
    d = DataLoader.from_arrays(a, a, axis_order="zxy")
    assert d.amplitude.shape == (2, 4, 3)
    assert d.amplitude[1, 2, 0] == a[1, 0, 2]
    with pytest.raises(ValueError): ImagingData(np.zeros((2, 3)), np.zeros((3, 2)))
    with pytest.raises(ValueError): ImagingData(np.full((2, 2), np.nan), np.zeros((2, 2)))
    with pytest.raises(ValueError): DataLoader.from_arrays(a, a, axis_order="xyzx")


@pytest.mark.parametrize("ext", ["npz", "hdf5", "npy"])
def test_loader_roundtrip(tmp_path, ext):
    a = np.arange(24).reshape(4, 6)
    path = tmp_path / ("input." + ext)
    kw = {}
    if ext == "npz": np.savez(path, amplitude=a, phase=-a)
    elif ext == "hdf5":
        with h5py.File(path, "w") as f:
            f["amplitude"] = a; f["phase"] = -a
        assert len(DataLoader.inspect_hdf5(path)) == 2
    else:
        np.save(path, a); np.save(tmp_path / "phase.npy", -a)
        kw["phase_path"] = tmp_path / "phase.npy"
    d = DataLoader.load(path, axis_order="yx", **kw)
    np.testing.assert_equal(d.phase, -a)


@pytest.mark.parametrize("mode", ["amplitude", "cyclic", "sequential", "side_by_side", "hsl_fusion"])
def test_export_modes(tmp_path, mode):
    a = np.arange(100).reshape(10, 10).astype(float); a[0, 0] = np.nan
    gen = ImageGenerator(a, a*10, {"spacing_xy": (.1, .2), "spatial_unit": "mm"})
    path = tmp_path / (mode + ".png")
    if mode == "amplitude": fig = gen.generate_amplitude(export=path)
    elif mode in {"cyclic", "sequential"}: fig = gen.generate_phase(mode, export=path)
    else: fig = gen.generate_comparison(mode, export=path)
    assert Image.open(path).width > 500
    assert json.loads(path.with_suffix(".png.json").read_text())["shape_yx"] == [10, 10]
    assert "mm" in fig.axes[0].get_xlabel()


def test_phase_fixed_scale_and_tiff(tmp_path):
    gen = ImageGenerator(np.ones((8, 8)), np.full((8, 8), .1))
    fig = gen.generate_phase(export=tmp_path / "phase.tiff")
    assert fig.axes[0].images[0].get_clim() == (-np.pi, np.pi)
    assert Image.open(tmp_path / "phase.tiff").format == "TIFF"


def test_metrics_against_known_values():
    a = np.zeros((8, 8)); b = np.full((8, 8), .1)
    assert image_metrics(a, b, data_range=1)["psnr"] == pytest.approx(20)
    assert image_metrics(a, a, data_range=1)["ssim"] == 1
    assert dice_score([1, 1, 0], [0, 1, 1]) == .5
    assert dice_score([0], [0]) == 1


def test_uq_schema_calibration_guard(tmp_path):
    path = tmp_path / "frame.hdf5"
    a = np.arange(64).reshape(8, 8)
    with h5py.File(path, "w") as f:
        f["data/IMAGE_AMPLITUDE/frame_20_qcl_1_left"] = a.astype("uint16")
        f["data/IMAGE_PHASE/frame_20_qcl_1_left"] = (a-30).astype("int16")
        f["data/IMAGE_PHASE/frame_20_qcl_1_left_BMP"] = a.astype("uint8")
    data = DataLoader.load_uq_frame(path)
    assert data.metadata["phase_unit"] == "raw_counts"
    with pytest.raises(ValueError): ImageGenerator(data).generate_phase()
    with pytest.raises(ValueError): ImageGenerator(data).generate_comparison("hsl_fusion")
    ImageGenerator(data).generate_phase("sequential")
    calibrated = DataLoader.load_uq_frame(path, phase_scale=.01, phase_offset=.2)
    np.testing.assert_allclose(calibrated.phase, (a-30)*.01+.2, atol=1e-7)
    ImageGenerator(calibrated).generate_phase()


def test_legacy_api_is_explicit():
    from part3_reconstruction import reconstruct_image, reconstruct_3d, calculate_quality_score
    with pytest.raises(TypeError): reconstruct_image(np.ones((8, 8)))
    d = ImagingData(np.ones((8, 8)), np.zeros((8, 8)))
    assert len(reconstruct_image(d).axes) == 4
    assert reconstruct_3d([d, d], [0, 1], spacing_xy=(1, 1)).data.amplitude.shape == (2, 8, 8)
    with pytest.raises(ValueError): calculate_quality_score(d.amplitude)


def test_physical_scale_bar():
    a = np.ones((10, 10))
    g = ImageGenerator(a, a, {"spacing_xy": (.2, .3), "spatial_unit": "mm"}, scale_bar_length=.5)
    assert len(g.generate_amplitude().axes[0].artists) == 1
    with pytest.raises(ValueError): ImageGenerator(a, a, scale_bar_length=1).generate_amplitude()


def test_hsl_fusion_keeps_hue_at_peak_amplitude():
    amplitude = np.tile(np.linspace(0, 1, 20), (20, 1))
    phase = np.full((20, 20), 1.0)
    def peak_rgb(**kw):
        fig = ImageGenerator(amplitude, phase).generate_comparison("hsl_fusion", norm="linear", **kw)
        return fig.axes[0].images[0].get_array()[0, -1, :3]
    capped, white = peak_rgb(), peak_rgb(max_lightness=1.0)
    assert np.ptp(capped) == pytest.approx(1.0)   # fully saturated colour
    np.testing.assert_allclose(white, 1.0)        # old behaviour: phase lost
    with pytest.raises(ValueError): ImageGenerator(amplitude, phase).generate_comparison("hsl_fusion", max_lightness=0)
