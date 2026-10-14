from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from part3_reconstruction import ImageGenerator
from part3_reconstruction.pipeline import slice_from_part2, run_part1_to_part3

RAW_FRAME = Path(__file__).resolve().parents[1] / "data" / "vegetables" / "frame-21.hdf5"


def _part2_output(target, shape, *, available, units="radians", aspect=None):
    n = 2
    amp = np.random.default_rng(1).random((n, *shape))
    mask = np.zeros((n, *shape), bool)
    mask[:, : shape[0] // 2] = True
    products = {"amplitude_corrected": amp, "amplitude_detail": amp - .5, "raw_amplitude": amp * 3,
                "raw_phase": np.full((n, *shape), -2.0), "signal_mask": mask,
                "phase_corrected": np.where(mask, 1.0, np.nan),
                "phase_correction_available": np.array([available, False])}
    upstream = {} if aspect is None else {"aspect_ratio_computed": aspect}
    return SimpleNamespace(target_type=target, z_indices=[40, 41],
                           metadata={"spatial_products": products, "phase_units": units,
                                     "source_path": "scan.hdf5", "upstream_metadata": upstream})


def test_released_phase_and_vegetable_aspect():
    d = slice_from_part2(_part2_output("Vegetable", (4, 13), available=True))
    assert d.metadata["phase_state"] == "unwrapped" and d.metadata["frame_id"] == 40
    assert d.metadata["pixel_aspect_yx"] == pytest.approx(13 / (4 * 1.9651163))
    assert np.isnan(d.phase[2:]).all() and "in memory" in d.metadata["provenance"]


def test_withheld_phase_and_raw_codes():
    out = _part2_output("Skin", (4, 13), available=False, units="raw_codes", aspect=2.0)
    d = slice_from_part2(out, 1, amplitude_channel="raw_amplitude")
    assert d.metadata["phase_unit"] == "raw_counts" and d.metadata["phase_state"] == "unknown"
    assert d.metadata["frame_aspect_ratio"] == 2.0
    with pytest.raises(ValueError): ImageGenerator(d).generate_phase("cyclic")   # no radians, no cyclic map
    ImageGenerator(d).generate_phase("sequential")


def test_leaf_is_transposed_and_errors():
    d = slice_from_part2(_part2_output("Leaf", (12, 4), available=True))
    assert d.amplitude.shape == (4, 12)
    with pytest.raises(IndexError): slice_from_part2(_part2_output("Leaf", (12, 4), available=True), 5)
    out = _part2_output("Leaf", (12, 4), available=True)
    out.metadata.pop("spatial_products")
    with pytest.raises(ValueError, match="spatial_products"): slice_from_part2(out)
    unknown = slice_from_part2(_part2_output("Chip", (4, 4), available=False))
    assert "pixel_aspect_yx" not in unknown.metadata                 # no aspect is invented


@pytest.mark.skipif(not RAW_FRAME.exists(), reason="real Vegetables frame not available")
def test_real_frame_through_parts_1_2_3(tmp_path):
    data = run_part1_to_part3(RAW_FRAME)
    assert data.amplitude.shape == (400, 1300) and data.metadata["frame_id"] == 21
    assert data.metadata["phase_unit"] == "rad" and data.metadata["phase_state"] == "wrapped"
    assert data.metadata["pixel_aspect_yx"] == pytest.approx(1300 / (400 * 1.9651163))
    ImageGenerator(data).generate_comparison(export=tmp_path / "frame21.png")
    assert (tmp_path / "frame21.png").stat().st_size > 10_000
