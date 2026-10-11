import numpy as np
import pytest

from thz_preprocessing import api, load_leaf_mat
from thz_preprocessing.io import LFIRawData, _leaf_reference_phase_to_radians
from thz_preprocessing.masking import signal_mask
from thz_preprocessing.spatial_correction import (
    _notch_filter,
    correct_image,
    estimate_line_offsets,
    gaussian_background,
)


def test_leaf_reference_decoder_is_not_rdm_signed14_scale():
    codes = np.array([-8192, -4096, 0, 4096, 8191], dtype=float)
    expected = np.angle(np.exp(1j * (codes * np.pi / 4096 - np.pi)))
    np.testing.assert_allclose(np.exp(1j * _leaf_reference_phase_to_radians(codes)),
                               np.exp(1j * expected), atol=1e-14)
    with pytest.raises(ValueError, match="finite codes"):
        _leaf_reference_phase_to_radians(np.array([8192]))


def test_spatial_decomposition_is_additive_and_does_not_mutate_input():
    rng = np.random.default_rng(4)
    image = rng.normal(size=(64, 96))
    before = image.copy()
    parts = correct_image(image, {})
    np.testing.assert_allclose(sum(parts.values()), image, atol=1e-14)
    np.testing.assert_array_equal(image, before)
    for value in parts.values():
        assert value.shape == image.shape
        assert np.isfinite(value).all()


def test_known_background_and_row_offsets_are_removed_without_erasing_local_feature():
    y, x = np.mgrid[:128, :192]
    clean = 8 * np.exp(-((x - 90)**2 + (y - 60)**2) / 18)
    background = 100 + 0.2 * x + 0.1 * y
    stripe = 2 * np.sin(2 * np.pi * y / 8)
    image = clean + background + stripe
    corrected = correct_image(image, {"background_sigma_pixels": 16})["corrected"]
    before_error = np.sqrt(np.mean((image - clean - np.median(background))**2))
    after_error = np.sqrt(np.mean((corrected - clean)**2))
    assert after_error < before_error * 0.15
    assert corrected[60, 90] > 0.9 * clean[60, 90]


def test_notch_rejects_selected_frequency_and_preserves_other_direction():
    y, x = np.mgrid[:128, :128]
    stripe = np.sin(2 * np.pi * y * 8 / 128)
    detail = np.sin(2 * np.pi * x * 13 / 128)
    corrected = _notch_filter(stripe + detail, [[8 / 128, 0]], 0.005)
    np.testing.assert_allclose(corrected, detail, atol=1e-10)


@pytest.mark.parametrize("params", [
    {"background_sigma_pixels": 0}, {"background_sigma_pixels": float("nan")},
    {"stripe_axis": "diagonal"}, {"fft_notches": [[0, 0]]},
    {"fft_notches": [[0.7, 0]]}, {"fft_notches": [[0.1]]},
    {"background_boundary_mode": "invalid"}, {"background_truncate": 0},
])
def test_invalid_spatial_parameters_fail(params):
    with pytest.raises(ValueError):
        correct_image(np.ones((16, 16)), params)


def test_wrapped_phase_not_background_filtered_and_measurements_retained():
    rng = np.random.default_rng(2)
    raw = LFIRawData(rng.normal(size=(2, 32, 48)),
                     rng.uniform(-np.pi, np.pi, size=(2, 32, 48)),
                     [10, 11], "Leaf")
    out = api.run_pipeline(raw)
    np.testing.assert_array_equal(out.amplitude, raw.amplitude)
    np.testing.assert_array_equal(out.phase, raw.phase)
    products = out.metadata["spatial_products"]
    assert not products["phase_correction_available"].any()
    assert np.isnan(products["phase_corrected"]).all()
    np.testing.assert_array_equal(products["raw_phase"], raw.phase)


def test_spatial_output_is_independent_of_neighbouring_frames():
    rng = np.random.default_rng(7)
    amp = rng.normal(size=(2, 64, 96))
    phase = np.zeros_like(amp)
    first = api.run_pipeline(LFIRawData(amp, phase, [3, 4], "Leaf"))
    amp[1] += 10000
    second = api.run_pipeline(LFIRawData(amp, phase, [3, 4], "Leaf"))
    np.testing.assert_array_equal(first.metadata["spatial_products"]["amplitude_corrected"][0],
                                  second.metadata["spatial_products"]["amplitude_corrected"][0])


def test_validated_reference_requires_modulo_match_and_provenance():
    y, x = np.mgrid[:32, :48]
    reference = (0.2 * x + 0.1 * y)[None, :, :]
    wrapped = np.angle(np.exp(1j * reference))
    raw = LFIRawData(np.ones_like(wrapped), wrapped, [0], "Leaf",
                     {"phase_reference": reference, "phase_reference_source": "test reference"})
    override = {"phase_unwrapping": {"method": "validated_reference"}}
    out = api.run_pipeline(raw, config_overrides=override)
    np.testing.assert_array_equal(out.phase, reference)
    assert out.metadata["spatial_products"]["phase_correction_available"].all()
    raw.metadata["phase_reference"] = reference + 0.1
    with pytest.raises(ValueError, match="does not match"):
        api.run_pipeline(raw, config_overrides=override)


def test_real_leaf_reference_equivalence_and_source_indices():
    from pathlib import Path
    from scipy.io import loadmat

    source = Path(__file__).resolve().parents[1] / "data/leaf/leaf1.mat"
    if not source.exists():
        pytest.skip("Leaf data not present")
    raw = load_leaf_mat(source, spatial_slice=(slice(0, 16), slice(0, 16)),
                        z_slice=slice(0, 91, 45))
    assert raw.z_indices == [0, 45, 90]
    reference = loadmat(source, variable_names=["L_P_U"])["L_P_U"]
    expected = reference[:16, :16, [0, 45, 90]].transpose(2, 0, 1)
    np.testing.assert_allclose(np.exp(1j * raw.phase), np.exp(1j * expected), atol=3e-14)


def test_real_leaf_u_m_is_exactly_reproduced_by_background_kernel():
    from pathlib import Path
    from scipy.io import loadmat
    from scipy.ndimage import gaussian_filter

    source = Path(__file__).resolve().parents[1] / "data/leaf/leaf1.mat"
    if not source.exists():
        pytest.skip("Leaf data not present")
    references = loadmat(source, variable_names=["L_P_U", "L_P_U_M"])
    for index in [0, 45, 90]:
        u = references["L_P_U"][:, :, index]
        m = references["L_P_U_M"][:, :, index]
        np.testing.assert_allclose(u - gaussian_filter(u, 20, mode="nearest", truncate=2),
                                   m, atol=3e-13, rtol=0)


def test_unconfirmed_phase_scale_withholds_phase_spatial_correction():
    raw = LFIRawData(np.ones((1, 16, 24)), np.zeros((1, 16, 24)), [0], "Skin",
                     {"phase_encoding_status": "unconfirmed"})
    out = api.run_pipeline(raw)
    assert out.metadata["phase_unwrapping_status"]["status"] == "withheld_unconfirmed_encoding"
    assert not out.metadata["spatial_products"]["phase_correction_available"].any()


def test_line_offsets_remove_random_scan_line_offsets_but_keep_real_bands():
    rng = np.random.default_rng(11)
    y, x = np.mgrid[:240, :320]
    scene = 50 * np.exp(-((x - 160) ** 2 + (y - 120) ** 2) / 4000.0)
    scene[100:112, :] += 30.0          # real band: full width, 12 lines thick
    scene[60:63, 40:150] += 30.0       # real thin feature covering a third of a line
    offsets = rng.normal(0, 4.0, size=240)
    image = scene + offsets[:, None] + rng.normal(0, 1.0, size=scene.shape)

    estimate = estimate_line_offsets(image, "rows", window=15)
    assert np.std(offsets - estimate) < 0.45 * np.std(offsets)
    # Neither real structure is mistaken for line offsets: on the clean scene
    # the estimate stays below 2% of the 30-unit band height everywhere.
    clean_estimate = estimate_line_offsets(scene, "rows", window=15)
    assert np.abs(clean_estimate).max() < 0.6
    assert np.abs(clean_estimate[100:112]).max() < 0.6

    with pytest.raises(ValueError, match="odd integer"):
        estimate_line_offsets(image, "rows", window=4)


def test_amplitude_corrected_keeps_object_contrast_and_detail_is_separate():
    rng = np.random.default_rng(3)
    y, x = np.mgrid[:160, :240]
    disc = (x - 120) ** 2 + (y - 80) ** 2 < 55 ** 2
    amplitude = np.where(disc, 200.0, 1500.0) + rng.normal(0, 5.0, size=disc.shape)
    amplitude += rng.normal(0, 8.0, size=160)[:, None]
    raw = LFIRawData(amplitude[None], np.zeros((1, 160, 240)), [0], "Vegetable")
    out = api.run_pipeline(raw, config_overrides={"noise_reduction": {"method": "none"}})
    products = out.metadata["spatial_products"]
    corrected = products["amplitude_corrected"][0]

    contrast_in = amplitude[~disc].mean() - amplitude[disc].mean()
    contrast_out = corrected[~disc].mean() - corrected[disc].mean()
    assert abs(contrast_out - contrast_in) < 0.01 * contrast_in

    # Two exact identities: line offsets, then the optional detail split.
    np.testing.assert_allclose(
        corrected + products["amplitude_stripe"][0] + products["amplitude_notch_removed"][0],
        amplitude, atol=1e-9)
    mask = products["signal_mask"][0]
    assert mask[~disc].mean() > 0.95 and mask[disc].mean() < 0.05
    np.testing.assert_allclose(
        (products["amplitude_detail"][0] + products["amplitude_background"][0])[mask],
        corrected[mask], atol=1e-9)
    assert np.isnan(products["amplitude_detail"][0][~mask]).all()
    assert abs(np.nanmean(products["amplitude_detail"][0])) < 0.02 * contrast_in


def test_masked_unwrap_releases_phase_only_inside_the_signal_region():
    rng = np.random.default_rng(5)
    y, x = np.mgrid[:120, :160]
    disc = (x - 80) ** 2 + (y - 60) ** 2 < 45 ** 2
    true_phase = 0.25 * x + 0.004 * (y - 60) ** 2      # tens of radians across the disc
    wrapped = np.angle(np.exp(1j * np.where(disc, true_phase, rng.uniform(-np.pi, np.pi, disc.shape))))
    amplitude = np.where(disc, 1000.0, 20.0) + rng.normal(0, 2.0, disc.shape)
    raw = LFIRawData(amplitude[None], wrapped[None], [0], "Skin")
    out = api.run_pipeline(raw, config_overrides={"noise_reduction": {"method": "none"}})

    status = out.metadata["phase_unwrapping_status"]
    assert status["status"] == "unwrapped"
    quality = status["quality_diagnostics"][0]
    assert quality["residue_fraction"] == 0.0 < quality["residue_fraction_whole_image"]
    mask = out.metadata["spatial_products"]["signal_mask"][0]
    inner = mask & disc
    assert inner.sum() > 0.9 * disc.sum()
    assert np.isnan(out.phase[0][~mask]).all()
    difference = out.phase[0][inner] - true_phase[inner]
    np.testing.assert_allclose(difference, np.round(difference[0] / (2 * np.pi)) * 2 * np.pi, atol=1e-9)

    products = out.metadata["spatial_products"]
    assert products["phase_correction_available"].all()
    rebuilt = (products["phase_corrected"] + products["phase_background"]
               + products["phase_stripe"] + products["phase_notch_removed"])[0]
    np.testing.assert_allclose(rebuilt[mask], out.phase[0][mask], atol=1e-9)

    # Without the mask the noise outside the sample fails the strict gate.
    whole = api.run_pipeline(raw, config_overrides={
        "noise_reduction": {"method": "none"}, "phase_unwrapping": {"mask": "none"}})
    assert whole.metadata["phase_unwrapping_status"]["status"] == "kept_wrapped_low_quality"
    np.testing.assert_array_equal(whole.phase, raw.phase)


def test_signal_mask_and_masked_background():
    image = np.full((40, 60), 10.0)
    image[:, 30:] = 500.0
    mask = signal_mask(image)
    assert mask[:, 35:].all() and not mask[:, :25].any()
    assert signal_mask(image, "none").all()
    with pytest.raises(ValueError):
        signal_mask(image, "threshold")
    # Normalised convolution ignores values outside the mask entirely.
    background = gaussian_background(image, 6.0, valid=mask)
    np.testing.assert_allclose(background[mask], 500.0, atol=1e-9)
    assert np.isnan(background[~mask]).all()


def test_real_leaf_independent_unwrap_disagrees_with_saved_reference():
    """Why the gate is strict: two unwrappers place cycles differently on Leaf."""
    from pathlib import Path
    from scipy.io import loadmat
    from skimage.restoration import unwrap_phase

    source = Path(__file__).resolve().parents[1] / "data/leaf/leaf1.mat"
    if not source.exists():
        pytest.skip("Leaf data not present")
    raw = load_leaf_mat(source, z_slice=slice(45, 46))
    saved = loadmat(source, variable_names=["L_P_U"])["L_P_U"][:, :, 45]
    cycles = np.rint((unwrap_phase(raw.phase[0], rng=0) - saved) / (2 * np.pi)).astype(int)
    _, counts = np.unique(cycles, return_counts=True)
    assert counts.max() / cycles.size < 0.5
    out = api.run_pipeline(raw)
    assert out.metadata["phase_unwrapping_status"]["status"] == "kept_wrapped_low_quality"
